"""The 5-FU example: an ODE used as a prior, and the certificate measured on it.

Reproduces every number of the ODE instantiation in one run:

    protocol      cycle 1 is the standard body-surface-area dose plus one plasma concentration
                  test, identical for every policy and not scored; the mechanistic model is
                  inverted on that test; cycles 2..N+1 are scored.
    policies      1. BSA static (standard of care)      2. uninformed Thompson sampling
                  3. mechanistic ODE, re-solved every cycle
                  4. ODE, one solve: the cycle-1 recommendation played every scored cycle,
                     the hybrid's information with no learning at all
                  5. hybrid: the ODE solved once at cycle 1 and used as a prior
    certificate   H(mu), R_mech = I(pi*; pi_hat) (Definition 2), B_mu (Eq. 3, both forms),
                  sigma_h / L_h / d_Z (Section 2.1, measured), C(B_mu) (Eq. 4),
                  B_crit(N) (Eq. 8), and the literature tier from the published constants.

Three-cohort discipline: the hybrid's prior is built entirely on a study cohort (its own seed,
its own noise streams); every regret is scored on a disjoint evaluation cohort. The ODE is
integrated once per cohort. Every certificate quantity is measured on the study patients and
their lead-in, so the whole chain is checkable before any evaluation patient is treated.

Outputs: a plain-text report on stdout, plus `outputs/example_summary.csv` (the certificate
and protocol quantities) and `outputs/example_policies.csv` (per-policy regret and ratios).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from scipy import stats

from twincert.generator import folfox as F
from twincert.experiment.mp_bandit import (
    POLICIES, capacity_bound, capacity_nu,
    covariate_prior, critical_bias,
    information_quantities, measurement_moments, model_arm_values, model_recommendation,
    occupancy_weights, occupancy_weighted_bias_centred,
    occupancy_weighted_bias_per_patient, run_policy)
from twincert.experiment.protocol import build_eval, build_study
from twincert.experiment.runin import RunInSpec
from twincert.experiment.stats import boot_indices, boot_ratio_from_indices, summ
from twincert.literature import LSS_MAPE, MEDIAN_AUC, OMEGA2_VMAX, SIGMA_H_LIT, SIGMA_NU_LIT

LABEL = {"bsa": "1. BSA static", "uninformed": "2. uninformed TS",
         "mechanistic": "3. mechanistic ODE", "mech_once": "4. ODE, one solve",
         "hybrid": "5. hybrid (Thm 2: TS, ODE prior)"}

#: The alternative reference exposure of the literature tier: Tan Fig. 5, the 2,800 mg/m2
#: panel (mg*h/L).
REF_AUC_ALT = 25.0

#: The operating point of this script, recorded in `example_summary.csv`: the deployed
#: model's population clearance is fitted on the whole study cohort, which makes the side
#: vacuous. Miscalibration is `run_bias_sweep.py`'s experiment.
CALIB_FRAC, CALIB_SIDE = 1.0, "low"


def literature_tier(K: int = 8) -> dict:
    """The literature-tier certificate, plus what each derivation assumption is worth.

    sigma_h is derived from one quoted number under the log-normal eta convention; sigma_nu
    from two, via (abs error / reference exposure) * sqrt(pi/2), where the last factor
    converts a mean absolute error to a standard deviation assuming a zero-mean Gaussian
    error. Both steps are assumptions, so each variant is reported.
    """
    sigma_h = SIGMA_H_LIT
    gauss = float(np.sqrt(np.pi / 2))
    variants = {
        # label -> sigma_nu under that assumption
        "as printed": SIGMA_NU_LIT,
        "no MAE->SD factor": LSS_MAPE / MEDIAN_AUC,
        "reference AUC 25": LSS_MAPE / REF_AUC_ALT * gauss,
        "both inputs 2-digit": round(SIGMA_NU_LIT, 2),
    }
    out = {"sigma_h": sigma_h, "H_uniform": float(np.log(K))}
    for name, s_nu in variants.items():
        sh = round(sigma_h, 2) if name == "both inputs 2-digit" else sigma_h
        out[name] = {"sigma_nu": float(s_nu),
                     "C": float(capacity_nu(sh, s_nu, 1, np.log(K)))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=10000, help="evaluation patients")
    ap.add_argument("--n-study", type=int, default=2000,
                    help="calibration-study patients; the size of a real dose-ranging "
                         "study, and the prior is only as good as this estimate")
    ap.add_argument("--cycles", type=int, default=11,
                    help="scored cycles; 1 lead-in + 11 scored = the 12-cycle FOLFOX course")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--study-seed", type=int, default=2024, help="must differ from --seed")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    if a.seed == a.study_seed:
        raise SystemExit("--seed and --study-seed must differ: the study cohort is disjoint")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    spec = F.FolfoxSpec()
    K, k_bsa, N = spec.n_arms, spec.bsa_arm, a.cycles
    rs = RunInSpec(n_cycles=1)

    # --- the study cohort: everything the hybrid knows before treating anyone ------
    # The deployed model's population clearance comes from the study cohort, calibrated on
    # the whole cohort (gamma = 1). The miscalibration axis belongs to `run_bias_sweep.py`.
    sc, dep = build_study(spec, a.n_study, a.study_seed, rs=rs)
    s_coh, s_conc, s_p, s_pi_star = sc.coh, sc.conc, sc.p, sc.pi_star
    cl_pop_hat, gamma, s_state, study = (dep.cl_pop_hat, dep.gamma, dep.state, dep.study)

    # --- the evaluation cohort, its shared lead-in, and the truth about them --------
    ec, state = build_eval(spec, a.n, a.seed, N, gamma, rs)
    coh, auc_true, conc_true = ec.coh, ec.auc, ec.conc
    p_true, pi_star, env = ec.p, ec.pi_star, ec.env

    print(f"study n={a.n_study} (seed {a.study_seed})   evaluation n={a.n} (seed {a.seed})   "
          f"K={K} arms {spec.dose_min:.0f}-{spec.dose_max:.0f} mg/m2")
    print(f"protocol: 1 lead-in cycle at {spec.dose_levels[k_bsa]:.0f} mg/m2 with one blood "
          f"test (unscored, identical for all), then {N} scored cycles")

    # --- what the lead-in test is worth (Definition 2, measured) -------------------
    # Every certificate quantity below is measured on the study cohort; the regret table
    # alone reads the evaluation cohort.
    pre = model_recommendation(spec, s_coh.bsa, covariate_prior(spec, cl_pop_hat, s_coh.bsa))
    i_pre = information_quantities(s_pi_star, pre, K)
    i_post = information_quantities(s_pi_star, s_state.pi_hat, K)
    print("\nthe mechanistic information of the prior, on the study cohort  "
          "(R_mech = I(pi*; pi_hat), Def. 2)")
    print(f"  before any test:   R_mech {i_pre['R_mech']:.3f} nats   "
          f"P(pi_hat=pi*) {np.mean(pre == s_pi_star):.3f}   "
          f"distinct doses recommended {len(np.unique(pre))}")
    print(f"  after one test:    R_mech {i_post['R_mech']:.3f} nats   "
          f"P(pi_hat=pi*) {np.mean(s_state.pi_hat == s_pi_star):.3f}   "
          f"distinct doses recommended {len(np.unique(s_state.pi_hat))}")
    print(f"  H(mu) = {i_post['H_mu']:.4f} nats (uniform ln {K} = {np.log(K):.4f});  "
          f"shuffled-estimator null {i_post['R_mech_null']:.4f} nats")

    # --- the certificate, every constant measured on the plant ---------------------
    # The primary constant is the weighted-norm L_h_mu (the norm of Eq. 3's B_mu, which
    # Assumption 2 states); the sup-norm variant is reported as a robustness line.
    w = occupancy_weights(s_pi_star, K)
    mom = measurement_moments(spec, s_coh, s_conc, s_p)
    d_Z = rs.n_cycles
    L_h = mom["L_h_mu"]
    mv = model_arm_values(spec, s_coh.bsa, s_state.cl_hat)   # the model the hybrid deploys
    B_pp = occupancy_weighted_bias_per_patient(s_p, mv, w)
    # B_mu is the centred, expectation form (Eq. 3); the uncentred variant is reported
    # alongside it.
    B_cen = occupancy_weighted_bias_centred(s_p, mv, w)
    C_cen = capacity_bound(B_cen, mom["sigma_h"], L_h, d_Z)
    C_pp_sup = capacity_bound(B_cen, mom["sigma_h"], mom["L_h"], d_Z)
    # The estimator-free ceiling of Proposition 1: needs only the population spread of the
    # measurement and the measurement noise, and holds for any estimator.
    C_nu = capacity_nu(mom["sigma_h"], mom["sigma_nu"], d_Z, i_post["H_mu"])
    # Theorem 3's working point is one cycle out of the full 12-cycle FOLFOX course: the
    # lead-in cycle plus the N scored ones.
    n_course = N + rs.n_cycles                       # 11 scored + 1 lead-in = 12
    B_crit_N = critical_bias(i_post["H_mu"], mom["sigma_h"], L_h, n_course, d_Z)
    print("\nthe certificate  (constants measured on the plant, Section 2.1; study cohort "
          f"n={a.n_study}; no evaluation patient enters any line below)")
    print(f"  sigma_h = {mom['sigma_h']:.4f}   sigma_nu = {mom['sigma_nu']:.4f}   "
          f"L_h = {L_h:.4f} (weighted norm: 1/sup||dJ/dz||_mu = 1/{mom['g_mu']:.3f})   "
          f"d_Z = {d_Z}")
    print(f"  B_mu (Eq. 3, centred): {B_cen:.4f}   "
          f"[uncentred per-patient {B_pp:.4f}]")
    print(f"  C_nu (Prop 1, estimator-free) = min(H, 0.5*ln(1+sigma_h^2/sigma_nu^2)) "
          f"= {C_nu:.3f} nats")
    print(f"  C(B_mu) (Prop 1, bias form) = {C_cen:.3f} nats;  C_nu <= C(B_mu): "
          f"{'yes' if C_nu <= C_cen + 1e-9 else 'NO'}")
    # What the centring of Eq. 3 is worth on this plant: centring can only shrink B_mu,
    # hence only widen the ceiling.
    C_unc = capacity_bound(B_pp, mom["sigma_h"], L_h, d_Z)
    print(f"  [centring of Eq. 3: uncentred B_mu = {B_pp:.4f} would give "
          f"C_uncentred(B_mu) = {C_unc:.3f} nats vs centred {C_cen:.3f}]")
    print(f"  [robustness, sup-norm L_h = {mom['L_h']:.4f}: C = {C_pp_sup:.3f} nats "
          f"(robustness line; the weighted norm is the primary constant)]")
    print(f"  bound check:  R_mech {i_post['R_mech']:.3f}  <=  C_nu {C_nu:.3f}  <=  "
          f"C(B_mu) {C_cen:.3f}   "
          f"{'HOLDS' if i_post['R_mech'] <= C_nu + 1e-9 else 'VIOLATED'}")
    # The working point the clinic can act on: can this model be expected to save one cycle
    # out of N? Theorem 3 at s = N/(N-1).
    verdict = "data-efficient" if B_cen < B_crit_N else "baseline"
    print(f"  B_crit (Thm 3) at the one-cycle point s=N/(N-1), N={n_course} "
          f"(the full course: {rs.n_cycles} lead-in + {N} scored): {B_crit_N:.3f}")
    print(f"    B_mu {B_cen:.3f} vs {B_crit_N:.3f}  ->  {verdict} regime "
          f"(margin B_crit/B_mu {B_crit_N / B_cen:.2f}x)")

    # --- the literature tier, and what its derivation assumptions cost ---------------
    lit = literature_tier()
    print("\nthe literature tier  (derived from published constants, no simulation)")
    print(f"  sigma_h = sqrt({OMEGA2_VMAX}) = {lit['sigma_h']:.4f}  [derived from Tan Table 2]")
    for name in ("as printed", "no MAE->SD factor", "reference AUC 25", "both inputs 2-digit"):
        v = lit[name]
        print(f"    {name:<20s} sigma_nu = {v['sigma_nu']:.4f}   C = {v['C']:.3f} nats")
    print("  (the spread across these rows is the precision of the literature tier)")

    print(f"\ncontext: P(in window | BSA arm) = {p_true[:, k_bsa].mean():.3f} "
          f"(literature p_BSA = 0.203)")

    # --- the policies over the scored cycles ---------------------------------------
    res, raw, traces = {}, {}, {}
    for pol in POLICIES:
        regret, trace = run_policy(
            pol, spec, coh, auc_true, conc_true, p_true, pi_star, env, N,
            cl_pop_hat, gamma, study=study if pol == "hybrid" else None, runin=state)
        res[pol], raw[pol], traces[pol] = summ(regret), regret, trace

    lead = float(state.regret.mean())
    # The lead-in draw is taken from every patient; what differs is whether a policy reads
    # it, what it reads each scored cycle, and how often the model is re-solved.
    leadin = {"bsa": "no", "uninformed": "no", "mechanistic": "yes", "mech_once": "yes",
              "hybrid": "yes"}
    readout = {"bsa": "none", "uninformed": "binary", "mechanistic": "continuous",
               "mech_once": "none", "hybrid": "binary"}
    solves = {"bsa": 0, "uninformed": 0, "mechanistic": 1 + N, "mech_once": 1, "hybrid": 1}
    print(f"\ncumulative regret, mean +- 95% CI over {a.n} patients  "
          f"(per cycle: P(in window | best arm) - P(in window | arm given))")
    print(f"  {'policy':26s} {'scored (' + str(N) + ' cyc)':>16s} "
          f"{'total (+lead-in)':>17s} {'lead-in':>9s} {'per-cycle':>11s} "
          f"{'ODE solves':>11s}")
    for pol in POLICIES:
        r = res[pol]
        print(f"  {LABEL[pol]:34s} {r.mean:8.3f} +-{r.ci:5.3f} {r.mean + lead:11.3f}   "
              f"{leadin[pol]:>9s} {readout[pol]:>11s} {solves[pol]:11d}")
    print(f"  shared lead-in cost: {lead:.3f} per patient, identical for every policy")

    # One resample, shared by every policy and every denominator below, so the ratios are
    # paired with each other as well as with their denominator; the printed intervals are
    # the CSV's.
    bidx = boot_indices(a.n, np.random.default_rng(123))

    print("\npaired ratios (comparator / hybrid; above 1 = the hybrid is better)")
    for other in ("bsa", "uninformed", "mechanistic", "mech_once"):
        d = raw[other] - raw["hybrid"]
        p = stats.ttest_rel(raw[other], raw["hybrid"])[1]
        print(f"  {LABEL[other]:34s} {res[other].mean / res['hybrid'].mean:6.2f}x   "
              f"paired diff {d.mean():+.3f} +- {1.96 * d.std(ddof=1) / np.sqrt(a.n):.3f}   "
              f"p = {p:.2e}" if p > 1e-300 else
              f"  {LABEL[other]:34s} {res[other].mean / res['hybrid'].mean:6.2f}x   "
              f"paired diff {d.mean():+.3f}   p < 1e-300")

    # `mech_once` has exactly the hybrid's information and no learning, so mech_once / hybrid
    # prices Theorem 2's reward feedback alone, while mech_once against the two model-free
    # references prices the prior alone.
    print("\nthe ODE solved once, played for the course (mech_once): ratios of means "
          "with paired-bootstrap 95% intervals")
    for den in ("hybrid", "bsa", "uninformed"):
        lo, hi = boot_ratio_from_indices(raw["mech_once"], raw[den], bidx)
        print(f"  mech_once / {den:11s} {res['mech_once'].mean / res[den].mean:6.2f}x   "
              f"[{lo:.2f}, {hi:.2f}]")

    # --- machine-readable outputs ---------------------------------------------------
    summary = {
        "n_eval": a.n, "n_study": a.n_study, "K": K, "n_scored": N, "n_leadin": rs.n_cycles,
        "calib_frac": CALIB_FRAC, "calib_side": CALIB_SIDE, "gamma": gamma,
        "H_mu": i_post["H_mu"], "R_mech_pre": i_pre["R_mech"], "R_mech": i_post["R_mech"],
        # Every certificate column below is a study-cohort quantity (`accuracy` included).
        "R_mech_null": i_post["R_mech_null"],
        "accuracy": float(np.mean(s_state.pi_hat == s_pi_star)),
        "sigma_h": mom["sigma_h"], "L_h": mom["L_h"], "d_Z": d_Z,
        # L_h_mu (weighted) is the primary constant and feeds every C / B_crit column below;
        # L_h (sup norm) is the robustness reference.
        "L_h_mu": mom["L_h_mu"], "g_mu": mom["g_mu"], "C_B_sup_norm": C_pp_sup,
        "sigma_nu": mom["sigma_nu"], "C_nu": C_nu,
        "B_mu_per_patient": B_pp, "B_mu_centred": B_cen,
        "C_B_per_patient": C_cen,
        "C_B_uncentred": C_unc,
        "n_course": n_course, "B_crit_N": B_crit_N,
        # Literature tier: derived from the quoted Tan constants alone, no plant involved.
        "lit_sigma_h": lit["sigma_h"],
        **{f"lit_{k}_{n.replace(' ', '_').replace('->', '_to_')}": v[k]
           for n, v in lit.items() if isinstance(v, dict) for k in ("sigma_nu", "C")},
        "p_bsa_arm": float(p_true[:, k_bsa].mean()), "leadin_regret": lead,
    }
    with open(out / "example_summary.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(summary.keys())
        wr.writerow(f"{v:.6g}" if isinstance(v, float) else v for v in summary.values())
    with open(out / "example_policies.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        # The three ratio triples share the one resample drawn above, so every ratio in this
        # file is paired with every other.
        wr.writerow(["policy", "regret_scored", "ci95", "regret_total", "reads_leadin",
                     "per_cycle_readout", "ode_solves", "ratio_vs_hybrid", "ratio_lo",
                     "ratio_hi", "ratio_vs_bsa", "ratio_vs_bsa_lo", "ratio_vs_bsa_hi",
                     "ratio_vs_uninformed", "ratio_vs_uninformed_lo",
                     "ratio_vs_uninformed_hi"])
        for pol in POLICIES:
            r_lo, r_hi = boot_ratio_from_indices(raw[pol], raw["hybrid"], bidx)
            b_lo, b_hi = boot_ratio_from_indices(raw[pol], raw["bsa"], bidx)
            u_lo, u_hi = boot_ratio_from_indices(raw[pol], raw["uninformed"], bidx)
            # 6 decimals, so that a 2-decimal display downstream is a single rounding of the
            # raw value.
            wr.writerow([pol, f"{res[pol].mean:.6f}", f"{res[pol].ci:.6f}",
                         f"{res[pol].mean + lead:.6f}", leadin[pol], readout[pol],
                         solves[pol],
                         f"{res[pol].mean / res['hybrid'].mean:.6f}",
                         f"{r_lo:.6f}", f"{r_hi:.6f}",
                         f"{res[pol].mean / res['bsa'].mean:.6f}",
                         f"{b_lo:.6f}", f"{b_hi:.6f}",
                         f"{res[pol].mean / res['uninformed'].mean:.6f}",
                         f"{u_lo:.6f}", f"{u_hi:.6f}"])
    print(f"\nwrote {out}/example_summary.csv and {out}/example_policies.csv")


if __name__ == "__main__":
    main()
