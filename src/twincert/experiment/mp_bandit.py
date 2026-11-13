"""The bandit layer: the dosing policies on Tan's ODE, and the information the model carries.

The truth is Tan's saturable ODE and the model is a different, simpler ODE, so nothing here is
a free parameter:

    pi*      is a policy, the dose arm that maximises this patient's expected reward,
    pi_hat   is a policy, the arm the deployed model recommends for them,
    R_mech   = I(pi*; pi_hat) is measured from the joint law of those two policies,
    B_mu     is the gap between the two models' per-arm values (Eq. 3), also measured,
    sigma_h, L_h, d_Z  are measured, so C(B_mu) (Eq. 4) and B_crit (Eq. 8) are computed.

The protocol (see `twincert.experiment.runin`): cycle 1 is the body-surface-area dose plus
one plasma concentration test, identical for every policy and not scored. The mechanistic
model is inverted on that test and the scored cycles start at cycle 2.

The policies share the identical cycle 1 and differ only in the scored cycles:

    bsa           the same body-surface-area dose, never changed. Standard of care.
    uninformed    Thompson sampling on Beta counts. Learning, with no model.
    mechanistic   the dose the model currently recommends, the model refit from a fresh
                  blood draw every cycle. The model, with no learning from reward feedback.
    mech_once     "ODE, one solve": the model is solved once on the lead-in draw and the arm
                  it recommends there, pi_hat, is played every scored cycle. No reward is
                  read, no further draw is taken, the model is never re-solved. This is the
                  hybrid's information with the learning removed.
    hybrid        Theorem 2: Thompson sampling over the latent best arm, whose
                  prior is P(pi* | pi_hat) built from the cycle-1 recommendation. The model
                  is solved once, at cycle 1, and the per-round observation is the binary
                  reward alone.

The reward is r = 1{realised AUC in the window}, so it is itself an exposure readout. What
the hybrid does not read is the plasma concentration `c_obs` that the mechanistic policy
re-fits on. The reward is computed from the true realised exposure, with no measurement
error, while `c_obs` carries the assay error.

The hybrid prior is a conditional table, not the model's self-assessment: P(pi* | pi_hat) is
estimated on a disjoint calibration cohort against ground truth (`run_study`), so the model
contributes its information and not its error. This makes H(pi* | pi_hat) = H(mu) - R_mech
hold by construction, which is the hypothesis of Theorem 2.
"""
from __future__ import annotations


import numpy as np
from scipy.special import betaincinv, ndtr

from twincert.generator import folfox as F


#: The policies, in the order every script prints and writes them. `mech_once` is
#: "ODE, one solve".
POLICIES = ("bsa", "uninformed", "mechanistic", "mech_once", "hybrid")


# --------------------------------------------------------------------------- #
# the reward: what cycle-to-cycle variability does to a window indicator
# --------------------------------------------------------------------------- #
def arm_reward_probs(spec: F.FolfoxSpec, auc) -> np.ndarray:
    """P(the realised 46-h exposure lands in the therapeutic window), per patient per arm.

    The reward is r = 1{AUC in [lo, hi]}, random because the same patient on the same dose
    does not get the same exposure twice; with log-normal cycle noise the probability is a
    difference of two normal CDFs. Closed form, never sampled: pi* is an argmax over K close
    values, and sampling would flip it for some patients and corrupt every information
    quantity.
    """
    a = np.maximum(np.asarray(auc, float), 1e-9)
    return (ndtr(np.log(spec.auc_hi / a) / spec.sigma_cycle)
            - ndtr(np.log(spec.auc_lo / a) / spec.sigma_cycle))


# --------------------------------------------------------------------------- #
# what the model believes, and the prior that follows from it
# --------------------------------------------------------------------------- #
def log_cl_estimate(log_cl_sum, n_obs: float) -> np.ndarray:
    """The estimate of theta = log CL from the blood draws alone: the mean of the inverted
    values.

    There is no population term: the deployed model is `CL = gamma * R/Css`, the
    calibration enters through `gamma` in `fit_clearance`, and the
    covariate-adjusted population value is what the model recommends before any draw
    (`covariate_prior`). `log_cl_sum` and `n_obs` are the sufficient statistics.
    """
    n = float(n_obs)
    if n <= 0:
        raise ValueError("log_cl_estimate needs at least one draw")
    return np.asarray(log_cl_sum, float) / n


def hybrid_prior_table(pi_star: np.ndarray, pi_hat: np.ndarray, K: int) -> np.ndarray:
    """P(pi* = j | pi_hat = i) -> (K, K), estimated on a calibration cohort against truth.

    The model is used only to partition patients; where the prior sits inside each cell comes
    from data. When the model is useless its recommendation is constant, the table has one
    populated row, and that row is the cohort marginal. Laplace (add-one) smoothing keeps an
    unobserved (recommendation, truth) pair off probability zero.

    What it costs: pi* must be known for the calibration patients, i.e. a prior dose-ranging
    study established their optimal doses.
    """
    T = np.ones((K, K), float)
    np.add.at(T, (np.asarray(pi_hat, int), np.asarray(pi_star, int)), 1.0)
    return T / T.sum(axis=1, keepdims=True)


def covariate_prior(spec: F.FolfoxSpec, cl_pop_hat, bsa) -> np.ndarray:
    """The model's clearance prior for a patient before any measurement -> (n,).

    A population-PK model's prior for an individual is the typical value scaled by that
    person's covariates, not the bare population mean.
    """
    return (np.asarray(cl_pop_hat, float)
            * (np.asarray(bsa, float) / spec.bsa_ref) ** F.covariate_exponent(spec))


def shape_matrix(reward_obs: np.ndarray, pi_star: np.ndarray, K: int) -> np.ndarray:
    """Q[k, j] = P(reward = 1 | arm j played, this patient's best arm is k) -> (K, K).

    Row k is the dose-response curve of a patient whose optimum is arm k: it peaks at j = k
    and falls away on both sides, so a failure at arm j is evidence about how far the truth
    is from j. Estimated from observed binary outcomes in the calibration study, never from
    the plant's exact probabilities. Rows with fewer than 20 patients fall back to the cohort
    mean row, and entries are clipped to [0.02, 0.98].
    """
    pi_star = np.asarray(pi_star, int)
    rows = [reward_obs[pi_star == k].mean(axis=0) if (pi_star == k).sum() > 20
            else reward_obs.mean(axis=0) for k in range(K)]
    return np.clip(np.asarray(rows, float), 0.02, 0.98)


def model_shape_matrix(spec: F.FolfoxSpec) -> np.ndarray:
    """Q[k, j] from the deployed one-compartment model alone -> (K, K).

    If arm k is a patient's optimum, the model puts the exposure at dose D_k at the centre of
    the therapeutic window. Dose proportionality then puts the exposure at arm j at
    target * D_j / D_k, and the between-cycle noise gives the probability that it lands in
    the window. No study patient, no optimal arm and no counterfactual outcome is used.
    """
    target = np.sqrt(spec.auc_lo * spec.auc_hi)
    d = spec.dose_levels
    return np.clip(arm_reward_probs(spec, target * d[None, :] / d[:, None]), 0.02, 0.98)


def run_study(spec: F.FolfoxSpec, auc: np.ndarray,
              rng: np.random.Generator, runin, q_source: str = "oracle") -> dict:
    """The offline calibration study -> {"Q", "table", "R_mech"}.

    Everything the hybrid knows before it treats anyone comes out of here: the returned dict
    is a pure function of the study cohort, and the evaluation cohort is not an argument. The
    study runs the same protocol that is deployed: a lead-in cycle at the BSA dose with one
    blood draw (`runin` is that lead-in, run on the study cohort), the model solved once on
    it, and the resulting recommendation calibrated against the study patients' known
    optimal arms.
    """
    ns, K = auc.shape
    p_study = arm_reward_probs(spec, auc)
    pi_star = p_study.argmax(axis=1)

    if q_source == "oracle":
        # One binary outcome per (patient, arm), grouped by the true optimal arm: the
        # construction of the submitted paper, and the default.
        Q = shape_matrix((rng.random((ns, K)) < p_study).astype(float), pi_star, K)
    elif q_source == "model":
        # The model's own dose response: needs no outcome data.
        Q = model_shape_matrix(spec)
    else:
        raise ValueError(f"q_source must be 'model' or 'oracle', got {q_source!r}")

    return {"Q": Q, "table": hybrid_prior_table(pi_star, runin.pi_hat, K),
            "R_mech": information_quantities(pi_star, runin.pi_hat, K)["R_mech"]}


def model_arm_values(spec: F.FolfoxSpec, bsa, cl_hat) -> np.ndarray:
    """The model's own prediction of each arm's reward probability -> (n, K).

    This is the attainment vector B_mu compares against the truth: K independent
    probabilities, not a distribution over arms.
    """
    pred = F.auc_model(spec, spec.dose_levels[None, :], np.asarray(bsa, float)[:, None],
                       np.asarray(cl_hat, float)[:, None])
    return arm_reward_probs(spec, pred)


def model_recommendation(spec: F.FolfoxSpec, bsa, cl_hat) -> np.ndarray:
    """The arm the mechanistic model recommends: the argmax of its own predicted reward.

    Nothing here assumes the reward is symmetric: a different reward would change the
    recommendation through `arm_reward_probs` and through nothing else. One degenerate case
    needs a rule: a patient the model places entirely outside the window has a row of
    probabilities that underflow to exact zeros, where the argmax would return arm 0. Those
    rows fall back to the arm whose predicted exposure is closest to the window as an
    interval.
    """
    pred = F.auc_model(spec, spec.dose_levels[None, :], np.asarray(bsa, float)[:, None],
                       np.asarray(cl_hat, float)[:, None])
    prob = arm_reward_probs(spec, pred)
    arm = prob.argmax(axis=1)
    dead = prob.max(axis=1) == 0.0
    if dead.any():
        lp = np.log(np.maximum(pred[dead], 1e-300))
        gap = np.maximum(np.maximum(np.log(spec.auc_lo) - lp, lp - np.log(spec.auc_hi)), 0.0)
        arm[dead] = gap.argmin(axis=1)
    return arm


# --------------------------------------------------------------------------- #
# the information the model carries, and its bias (Eqs. 3, 4, 8)
# --------------------------------------------------------------------------- #
def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def information_quantities(pi_star: np.ndarray, pi_hat: np.ndarray, K: int) -> dict:
    """H(mu), R_mech = I(pi*; pi_hat) and H_mech = H(pi* | pi_hat), all plug-in estimates.

    The plug-in mutual information is biased upward by roughly (K-1)^2 / (2n) nats, so the
    same statistic is also computed after shuffling pi_hat (`R_mech_null`). H_mech is returned
    as H(mu) - R_mech so the identity holds exactly in the reported numbers.
    """
    J = np.zeros((K, K), float)
    np.add.at(J, (pi_star, pi_hat), 1.0)
    H_mu = _entropy(J.sum(axis=1))
    R = max(H_mu + _entropy(J.sum(axis=0)) - _entropy(J.ravel()), 0.0)
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(20):
        Js = np.zeros((K, K), float)
        np.add.at(Js, (pi_star, rng.permutation(pi_hat)), 1.0)
        nulls.append(_entropy(Js.sum(axis=1)) + _entropy(Js.sum(axis=0)) - _entropy(Js.ravel()))
    return {"H_mu": H_mu, "R_mech": R, "H_mech": H_mu - R,
            "R_mech_null": float(np.mean(nulls))}


def occupancy_weights(pi_star: np.ndarray, K: int) -> np.ndarray:
    """mu(pi_k): the fraction of the cohort whose true optimal arm is k."""
    w = np.bincount(np.asarray(pi_star, int), minlength=K).astype(float)
    return w / w.sum()


def occupancy_weighted_bias_per_patient(mu_true: np.ndarray, mu_model: np.ndarray,
                                        w: np.ndarray) -> float:
    """Eq. (3) without the centring: differenced per patient, then pooled by RMS.

    Reported alongside the centred form because the choice moves the certificate; the
    centred form is the definition of Eq. (3).
    """
    d = np.asarray(mu_true, float) - np.asarray(mu_model, float)
    return float(np.sqrt(np.mean(np.sum(np.asarray(w, float) * d ** 2, axis=1))))


def occupancy_weighted_bias_centred(mu_true: np.ndarray, mu_model: np.ndarray,
                                    w: np.ndarray) -> float:
    """B_mu (Eq. 3): centred per-patient reward gap, in expectation over patients.

        B_mu^2 = E_{M*~rho, nu} [ sum_k w_k ( (J(pi_k;M*) - Jbar(M*))
                                            - (J(pi_k;Mhat) - Jbar(Mhat)) )^2 ],
        Jbar(M) = sum_k w_k J(pi_k; M).

    A common offset of the reward vector scores zero because it does not change the
    recommendation. This is also the norm the L_h construction uses, so the chain
    E||dh||^2 >= L_h^2 B_mu^2 is uniform.
    """
    w = np.asarray(w, float)
    d = np.asarray(mu_true, float) - np.asarray(mu_model, float)
    d = d - (w[None, :] * d).sum(axis=1, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(w * d ** 2, axis=1))))


def capacity_nu(sigma_h: float, sigma_nu: float, d_Z: int = 1,
                H_mu: float | None = None) -> float:
    """The estimator-free channel ceiling of Proposition 1:

        C_nu = min{ H(mu), (d_Z/2) * log(1 + sigma_h^2 / (d_Z * sigma_nu^2)) }.

    It needs only the population spread of the measured quantity (sigma_h) and the
    measurement noise (sigma_nu), and holds for any Mhat = f(Z) by data processing. The min
    with H(mu) caps the bound at the total prior entropy.

    The d_Z convention, shared with `capacity_bound`: the denominator of the log is the total
    noise power E||nu||^2. Here nu ~ N(0, sigma_nu^2 I) over d_Z coordinates, so
    E||nu||^2 = d_Z sigma_nu^2. In `capacity_bound` the same slot is filled by Assumption 3's
    lower bound E||nu||^2 >= L_h^2 B_mu^2, which is already a total power, so no d_Z divides
    there. The two agree whenever L_h B_mu = sqrt(d_Z) sigma_nu, at any d_Z.
    """
    c = 0.5 * d_Z * np.log(1.0 + sigma_h ** 2 / (d_Z * max(sigma_nu, 1e-12) ** 2))
    return float(min(c, H_mu) if H_mu is not None else c)


def measurement_moments(spec: F.FolfoxSpec, coh: F.Cohort, conc: np.ndarray,
                        p_true: np.ndarray, rel: float = 0.02) -> dict:
    """(sigma_h, sigma_nu, L_h) for the pre-trial measurement Z = h(M*) + nu, measured on
    this plant.

    On this plant the measurement is the lead-in blood test: h(M*) is the noise-free log
    plateau concentration at the BSA dose, nu is the cycle plus assay noise, and d_Z = 1 (one
    draw). sigma_h^2 = tr(Cov(h(M*))) is the variance of that measurement across the cohort.
    L_h is Assumption 3's lower Lipschitz constant tying the reward-space model norm to
    measurement space; the tightest valid constant is 1 / sup ||dJ/dz||, estimated by
    perturbing the only free direction of M*, the patient's Vmax at fixed body surface area.

    Two norms, two constants, both returned:

        L_h     = 1 / sup_i max_k |dJ_k/dz|          (sup norm)
        L_h_mu  = 1 / sup_i ||sqrt(w) (dJ/dz - c)||  (weighted, centred: the norm of Eq. 3)

    The weighted norm is never larger than the sup norm, so L_h_mu >= L_h and the weighted
    certificate C(B_mu) is tighter. Assumption 2 states the weighted norm and Eq. 3's B_mu uses
    it, so `L_h_mu` is the primary constant downstream; `L_h` is the sup-norm robustness line.
    """
    arm = spec.bsa_arm
    z = np.log(np.maximum(conc[:, arm], 1e-300))
    auc2, conc2 = F.arm_exposures(spec, F.Cohort(coh.bsa, coh.vmax * (1.0 + rel)))
    dz = np.log(np.maximum(conc2[:, arm], 1e-300)) - z
    dJ = (arm_reward_probs(spec, auc2) - p_true) / np.where(np.abs(dz) > 1e-12, dz,
                                                            1e-12)[:, None]     # (n, K)
    slope = np.abs(dJ).max(axis=1)                           # ||dJ/dz||_inf, per patient
    w = occupancy_weights(p_true.argmax(axis=1), p_true.shape[1])
    centred = dJ - (w[None, :] * dJ).sum(axis=1, keepdims=True)
    slope_mu = np.sqrt((w[None, :] * centred ** 2).sum(axis=1))
    return {"sigma_h": float(z.std()),
            "sigma_nu": float(np.sqrt(spec.sigma_cycle ** 2 + spec.sigma_assay ** 2)),
            "L_h": float(1.0 / slope.max()),
            "L_h_mu": float(1.0 / slope_mu.max()), "g_mu": float(slope_mu.max())}


def capacity_bound(B_mu: float, sigma_h: float, L_h: float, d_Z: int = 1) -> float:
    """Proposition 1's ceiling in its bias form (Eq. 4):

        C(B_mu) = (d_Z / 2) * log( 1 + sigma_h^2 / (L_h^2 B_mu^2) )

    The mechanistic model acts as a noisy Gaussian channel carrying the pre-trial
    measurement: signal variance sigma_h^2, model bias contributing noise variance
    L_h^2 B_mu^2. See `capacity_nu` for the shared d_Z convention.
    """
    B = max(float(B_mu), 1e-12)
    return float(0.5 * d_Z * np.log(1.0 + sigma_h ** 2 / (L_h ** 2 * B ** 2)))


def critical_bias_at(target_nats: float, sigma_h: float, L_h: float, d_Z: int = 1) -> float:
    """Theorem 3's critical bias: the bias at which the channel delivers exactly
    `target_nats`. Inverts `capacity_bound`:

        B_crit(s) = (sigma_h / L_h) / sqrt( exp( (2/d_Z) * (1 - 1/s) * H(mu) ) - 1 )

    with target_nats = (1 - 1/s) H(mu). The general form is kept because the target is a
    choice; the reported working point is `critical_bias`.
    """
    return float((sigma_h / L_h) / np.sqrt(np.expm1(2.0 * target_nats / d_Z)))


def critical_bias(H_mu: float, sigma_h: float, L_h: float, n_cycles: int,
                  d_Z: int = 1) -> float:
    """Theorem 3 at the one-cycle working point s = N/(N-1), where (1 - 1/s)H = H/N."""
    return critical_bias_at(H_mu / n_cycles, sigma_h, L_h, d_Z)


# --------------------------------------------------------------------------- #
# the policies
# --------------------------------------------------------------------------- #
def _thompson_argmax(alpha: np.ndarray, beta: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Beta posterior draw by inverse CDF on a pre-drawn uniform: deterministic given `u`.

    Not `rng.beta`: rejection sampling consumes a parameter-dependent number of raw uniforms,
    so two policies' random streams would desynchronise and the paired comparison would
    silently become unpaired.
    """
    return betaincinv(np.maximum(alpha, 1e-6), np.maximum(beta, 1e-6),
                      np.clip(u, 1e-12, 1 - 1e-12)).argmax(axis=1)


def draw_environment(rng: np.random.Generator, n: int, n_cycles: int, k: int,
                     spec: F.FolfoxSpec) -> dict:
    """Pre-draw every random quantity a policy will consume, so all policies face identical
    patients and identical noise.

    `cycle_noise` is biological: it moves the exposure the patient actually received, so it
    moves the reward. `assay_noise` is measurement only: it moves the number the laboratory
    reports, corrupting the model's estimate while leaving the reward untouched.
    """
    return {"cycle_noise": rng.normal(0.0, spec.sigma_cycle, (n, n_cycles)),
            "assay_noise": rng.normal(0.0, spec.sigma_assay, (n, n_cycles)),
            "ts_uniform": rng.random((n, n_cycles, k))}


def run_policy(policy: str, spec: F.FolfoxSpec, coh: F.Cohort, auc_true: np.ndarray,
               conc_true: np.ndarray, p_true: np.ndarray, pi_star: np.ndarray, env: dict,
               n_cycles: int, cl_pop_hat: float, gamma: float = 1.0,
               study: dict | None = None, runin=None):
    """Run one policy over the `n_cycles` scored cycles; return (regret, arm trace).

    Regret is against each patient's own best arm in expected reward (Definition 1), summed
    over the scored cycles; the shared lead-in's cost is the caller's to report.

    The hybrid is Theorem 2 literally: the arm is sampled from the posterior over the latent
    best arm, and that posterior sees the prior plus the binary rewards only. The lead-in
    draw enters exactly once, through the prior, and the concentration reading `c_obs` is
    never read again.

    `runin` is the lead-in's `RunInState`: the model's sufficient statistics after the
    cycle-1 blood test. `mechanistic` continues that posterior, refitting from a fresh draw
    every cycle. `hybrid` freezes the model there: its prior is the row of the study table
    selected by the cycle-1 recommendation, and its rewards enter through the study's Q
    matrix. `mech_once` freezes it there too and then stops: it plays `runin.pi_hat` in every
    scored cycle, updates nothing, and consumes no random number, so its arm trace is a pure
    function of the lead-in.

    Order within a cycle: choose an arm, accrue regret, realise the cycle noise, observe the
    reward and the concentration, update.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if policy == "hybrid" and (study is None or runin is None):
        raise ValueError("the hybrid needs a `study` from run_study() and a lead-in `runin`")
    if policy == "mech_once" and runin is None:
        raise ValueError("mech_once plays the lead-in recommendation: it needs a `runin`")
    n, k = p_true.shape
    idx = np.arange(n)
    best = p_true[idx, pi_star]

    alpha = np.zeros((n, k))               # observed successes  (uninformed TS)
    beta = np.zeros((n, k))                # observed failures   (uninformed TS)
    loglik = np.zeros((n, k))              # log P(rewards | pi* = k)       (hybrid)

    # Where the model starts: the lead-in's posterior if there was one, the covariate-only
    # prior otherwise. Sufficient statistics are carried, not the point estimate, so the
    # mechanistic policy continues the same posterior rather than restarting one.
    if runin is None:
        log_cl_sum, n_obs = np.zeros(n), 0.0
        cl_hat = covariate_prior(spec, cl_pop_hat, coh.bsa)
    else:
        log_cl_sum, n_obs = runin.log_cl_sum.copy(), runin.n_obs
        cl_hat = runin.cl_hat

    if policy == "hybrid":
        # The prior is mu_hyb itself: the table row the cycle-1 recommendation selects, fixed
        # for the whole horizon. Laplace smoothing inside `hybrid_prior_table` keeps every
        # entry positive, so no floor is mixed in here.
        log_prior = np.log(study["table"][runin.pi_hat])

    regret = np.zeros(n)
    trace = np.zeros((n, n_cycles), dtype=int)

    for t in range(n_cycles):
        # ---- choose an arm -------------------------------------------------------
        if policy == "bsa":
            arm = np.full(n, spec.bsa_arm)
        elif policy == "uninformed":
            arm = _thompson_argmax(1.0 + alpha, 1.0 + beta, env["ts_uniform"][:, t])
        elif policy == "mechanistic":
            arm = model_recommendation(spec, coh.bsa, cl_hat)
        elif policy == "mech_once":
            # The single solve's recommendation, held for the whole course; `env` is not
            # touched, so this policy cannot shift any other policy's noise or uniforms.
            arm = runin.pi_hat
        else:
            # Posterior over the latent best arm: prior + reward evidence, each entering
            # exactly once. Theorem 2 literally: sample pi* from it (probability matching).
            post = log_prior + loglik
            post = np.exp(post - post.max(axis=1, keepdims=True))
            post /= post.sum(axis=1, keepdims=True)
            u = env["ts_uniform"][:, t, 0]
            arm = (post.cumsum(axis=1) > u[:, None]).argmax(axis=1)

        trace[:, t] = arm
        regret += best - p_true[idx, arm]

        # ---- realise the cycle ---------------------------------------------------
        # One biological draw drives both the exposure the patient received and the reading
        # taken from it; the assay error is layered on the reading alone.
        cycle = np.exp(env["cycle_noise"][:, t])
        auc_real = auc_true[idx, arm] * cycle
        reward = ((auc_real >= spec.auc_lo) & (auc_real <= spec.auc_hi)).astype(float)
        c_obs = conc_true[idx, arm] * cycle * np.exp(env["assay_noise"][:, t])

        # ---- update --------------------------------------------------------------
        if policy == "uninformed":
            alpha[idx, arm] += reward
            beta[idx, arm] += 1.0 - reward
        elif policy == "mechanistic":
            # Refit the model on (dose given, concentration measured), every cycle.
            new_cl = F.fit_clearance(spec, spec.dose_levels[arm], coh.bsa, c_obs, gamma)
            log_cl_sum += np.log(np.maximum(new_cl, 1e-9))
            n_obs += 1.0
            cl_hat = np.exp(log_cl_estimate(log_cl_sum, n_obs))
        elif policy == "hybrid":
            # The reward, through the study's dose-response rows. `c_obs` is deliberately
            # untouched: the theory's per-round observation is the one bit.
            q = study["Q"][:, arm].T                       # (n, K)
            loglik += reward[:, None] * np.log(q) + (1.0 - reward)[:, None] * np.log(1.0 - q)

    return regret, trace
