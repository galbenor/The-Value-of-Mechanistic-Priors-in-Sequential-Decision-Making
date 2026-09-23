"""The scalar end-to-end channel ablation.

The full pipeline reduced to a scalar channel plus the dose grid, with the Tan-sourced
constants of `twincert.literature`:

    1.  theta* = log-clearance deviation ~ N(0, sigma_h^2).
    2.  Z = theta* + nu,  nu ~ N(0, sigma_nu^2)   (the pre-dose limited-sampling estimate).
    3.  M_hat is the model with log-clearance Z. Both models map dose to exposure linearly,
        AUC(dose, theta) = dose / (CL_pop e^theta), with CL_pop chosen so the BSA dose hits
        the window centre at theta = 0. pi_hat = the arm with the best model-predicted
        attainment; pi* = the arm with the best true attainment.
    4.  The realised I(pi*; pi_hat) is computed by Monte Carlo over patients and compared with
        the estimator-free ceiling C_nu = 0.5 ln(1 + sigma_h^2/sigma_nu^2). Here h is the
        identity in parameter units, so the two forms of Proposition 1 coincide numerically,
        a property of this scalar construction and not an identity in general.
    5.  Thompson sampling with the induced mu_hyb = P(pi* | pi_hat) runs against the same
        baselines as the plant experiment: uninformed TS and fixed BSA dosing.

Attainment is computed against multiplicative cycle noise (sigma_cycle, the intra-patient
figure of Li et al. 2023); rewards are Bernoulli draws of that attainment. Regret is
Definition 1's expected-reward regret against the patient's own best arm, summed over N = 12
cycles.

Writes outputs/channel_ablation.csv and prints the certificate comparison.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import norm

from twincert.experiment.mp_bandit import hybrid_prior_table, information_quantities
from twincert.experiment.stats import summ
from twincert.literature import SIGMA_H_LIT, SIGMA_NU_LIT

K = 8

# Every constant below comes from a published value; the two Tan-sourced ones and their
# derivation are declared in `twincert.literature`.
SIGMA_H = SIGMA_H_LIT       # the population spread (sqrt of Tan's Omega^2(Vmax))
SIGMA_NU = SIGMA_NU_LIT     # the LSS precision (Tan Sec. 3.4 / Sec. 3.5, Fig. 5)
SIGMA_CYCLE = 0.20          # Li et al. 2023: intra-patient cycle-to-cycle AUC variability
AUC_LO, AUC_HI = 20.0, 30.0                     # therapeutic window (Kaldate et al.)
DOSES = np.linspace(1600.0, 3600.0, K)          # mg/m^2, Kaldate's adjustment range
DOSE_BSA = 2400.0                               # standard BSA-derived dose (mg/m^2)
CL_POP = DOSE_BSA / (0.5 * (AUC_LO + AUC_HI))   # BSA dose -> window centre at theta = 0
BSA_ARM = int(np.argmin(np.abs(DOSES - DOSE_BSA)))
N_CYCLES = 12


def attainment(theta: np.ndarray) -> np.ndarray:
    """P(AUC_k e^eps in window) for every arm -> (n, K).

    The scalar plant's own exposure model, AUC(dose, theta) = dose / (CL_pop e^theta), rather
    than the ODE-integrated exposure of `mp_bandit.arm_reward_probs`; the window probability
    is the same difference of two normal CDFs.
    """
    log_auc = np.log(DOSES)[None, :] - np.log(CL_POP) - theta[:, None]
    z_hi = (np.log(AUC_HI) - log_auc) / SIGMA_CYCLE
    z_lo = (np.log(AUC_LO) - log_auc) / SIGMA_CYCLE
    return norm.cdf(z_hi) - norm.cdf(z_lo)


# --------------------------------------------------------------------------- #
# the two Bernoulli learners of this ablation, vectorised over patients
#
#     ts_uninformed   standard Beta(1,1) Thompson sampling, one independent posterior per
#                     arm. Learning, with no model: it never sees pi_hat.
#     ts_hybrid       Thompson sampling over the latent best arm with prior
#                     mu_hyb(. | pi_hat), Theorem 2's algorithm. The per-arm reward means
#                     carry independent Beta(1,1) posteriors; the posterior over the best arm
#                     is P(k | data) ~ mu_hyb(k) * P(arm k has the largest mean | posteriors),
#                     the second factor estimated each round from `s_samples` joint draws.
#                     At a uniform prior this reduces to standard probability-matching TS.
#
# The draw order is part of the result: every reported number is a function of a seed and of
# the sequence of calls made on that generator, so the order inside these loops, and the
# order in which `main` invokes them, must not be rearranged.
#
# Regret is Definition 1's regret in expected reward against the patient's own best arm, so
# no Bernoulli draw enters it; the draws only drive what the learner observes.
# --------------------------------------------------------------------------- #
def ts_uninformed(rng: np.random.Generator, p_true: np.ndarray, pi_star: np.ndarray,
                  n_cycles: int) -> np.ndarray:
    """Standard Beta(1,1) Thompson sampling, vectorised over trials -> regret (m,)."""
    n, K_ = p_true.shape
    wins = np.zeros((n, K_))
    losses = np.zeros((n, K_))
    rows = np.arange(n)
    best = p_true[rows, pi_star]
    regret = np.zeros(n)
    for _ in range(n_cycles):
        arm = rng.beta(1.0 + wins, 1.0 + losses).argmax(axis=1)
        regret += best - p_true[rows, arm]
        r = rng.random(n) < p_true[rows, arm]
        wins[rows, arm] += r
        losses[rows, arm] += ~r
    return regret


def ts_hybrid(rng: np.random.Generator, p_true: np.ndarray, pi_star: np.ndarray,
              prior_rows: np.ndarray, n_cycles: int,
              s_samples: int = 128) -> np.ndarray:
    """Latent-best-arm TS with the prior mu_hyb(. | pi_hat) -> regret (m,).

    `prior_rows` is (m, K): patient i's prior over which arm is best, i.e. the row of the
    mu_hyb table that its own model recommendation selects.
    """
    n, K_ = p_true.shape
    wins = np.zeros((n, K_))
    losses = np.zeros((n, K_))
    rows = np.arange(n)
    best = p_true[rows, pi_star]
    regret = np.zeros(n)
    for _ in range(n_cycles):
        th = rng.beta(1.0 + wins[:, None, :], 1.0 + losses[:, None, :], (n, s_samples, K_))
        am = th.argmax(axis=2)
        counts = np.zeros((n, K_))
        np.add.at(counts, (np.repeat(rows, s_samples), am.ravel()), 1.0)
        post = prior_rows * counts
        z = post.sum(axis=1, keepdims=True)
        post = np.where(z > 0, post / np.maximum(z, 1e-300), prior_rows)
        arm = (post.cumsum(axis=1) > rng.random(n)[:, None]).argmax(axis=1)
        regret += best - p_true[rows, arm]
        r = rng.random(n) < p_true[rows, arm]
        wins[rows, arm] += r
        losses[rows, arm] += ~r
    return regret


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m", type=int, default=6000, help="patients per policy run")
    ap.add_argument("--m-cal", type=int, default=400000,
                    help="patients for the channel Monte Carlo (I(pi*;pi_hat), mu_hyb)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    c_nu = 0.5 * np.log(1.0 + (SIGMA_H / SIGMA_NU) ** 2)

    # ---- the channel, by Monte Carlo ---------------------------------------------
    rng = np.random.default_rng(a.seed)
    th_cal = rng.normal(0.0, SIGMA_H, a.m_cal)
    z_cal = th_cal + rng.normal(0.0, SIGMA_NU, a.m_cal)
    pi_star_cal = attainment(th_cal).argmax(axis=1)
    pi_hat_cal = attainment(z_cal).argmax(axis=1)
    # H(mu) and the realised R_mech = I(pi*; pi_hat), from the same plug-in estimator the
    # ODE tier uses.
    info = information_quantities(pi_star_cal, pi_hat_cal, K)
    H_mu, R_real = info["H_mu"], info["R_mech"]

    # The induced prior mu_hyb = P(pi* | pi_hat), the empirical conditional of the channel,
    # with the same Laplace (add-one) smoothing as the ODE tier, so no arm has zero prior mass.
    mu_hyb = hybrid_prior_table(pi_star_cal, pi_hat_cal, K)

    print("Scalar end-to-end channel, Tan-sourced constants")
    print(f"  sigma_h = {SIGMA_H:.3f} (= sqrt(0.10), Tan Table 2), "
          f"sigma_nu = {SIGMA_NU:.3f} (Tan Sec 3.4 LSS), d_Z = 1, K = {K}")
    print(f"  robustness line (estimator-free)  C_nu = "
          f"0.5 ln(1+(sigma_h/sigma_nu)^2) = {c_nu:.3f} nats")
    print(f"  H(mu)      (realized, MC)  = {H_mu:.3f} nats   (uniform ln 8 = {np.log(K):.3f})")
    print(f"  R_mech     (realized, MC)  = {R_real:.3f} nats")
    print(f"  R_mech {R_real:.3f} <= C_nu {c_nu:.3f}  "
          f"{'HOLDS' if R_real <= c_nu else 'VIOLATED'}")

    # ---- the bandit --------------------------------------------------------------
    rng = np.random.default_rng(a.seed + 1)
    theta = rng.normal(0.0, SIGMA_H, a.m)
    z = theta + rng.normal(0.0, SIGMA_NU, a.m)
    p_true = attainment(theta)
    pi_star = p_true.argmax(axis=1)
    pi_hat = attainment(z).argmax(axis=1)
    rows = np.arange(a.m)
    best = p_true[rows, pi_star]

    res = {}
    res["bsa"] = summ((best - p_true[:, BSA_ARM]) * N_CYCLES)
    res["ts_uninformed"] = summ(ts_uninformed(np.random.default_rng(a.seed + 3),
                                              p_true, pi_star, N_CYCLES))
    res["ts_hybrid"] = summ(ts_hybrid(np.random.default_rng(a.seed + 4), p_true, pi_star,
                                      mu_hyb[pi_hat], N_CYCLES))

    print(f"\n  regret over N = {N_CYCLES} cycles (mean +- 95% CI, m = {a.m:,})")
    for k, (m_, c_) in res.items():
        print(f"    {k:14s} {m_:6.3f} +- {c_:5.3f}")
    print(f"    uninf/hyb ratio {res['ts_uninformed'][0] / res['ts_hybrid'][0]:.2f}x, "
          f"bsa/hyb {res['bsa'][0] / res['ts_hybrid'][0]:.2f}x")

    with open(out / "channel_ablation.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["quantity", "value", "ci95"])
        wr.writerow(["sigma_h", SIGMA_H, ""])
        wr.writerow(["sigma_nu", SIGMA_NU, ""])
        wr.writerow(["C_nu_holds", round(c_nu, 4), ""])
        wr.writerow(["H_mu_realized", round(H_mu, 4), ""])
        wr.writerow(["R_mech_realized", round(R_real, 4), ""])
        for k, (m_, c_) in res.items():
            wr.writerow([f"regret_{k}", round(m_, 4), round(c_, 4)])
    print(f"\nwrote {out}/channel_ablation.csv")


if __name__ == "__main__":
    main()
