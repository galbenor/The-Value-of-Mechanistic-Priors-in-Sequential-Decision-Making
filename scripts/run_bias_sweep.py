"""The dense calibration-bias sweep behind the by-bias figure and tables.

For every calibration slice (the fraction of the study cohort the deployed model's population
clearance is fitted on, from everyone down to the slowest/fastest 20%, in steps of 0.05) the
protocol is re-run -- calibration, lead-in, policies -- and the measured centred bias B_mu is
recorded together with the regret of the three model-dependent policies:

    hybrid        Theorem 2's algorithm (TS with the mu_hyb prior, rewards only). Which
                  population its prior table is fitted on is `--fit-table-on-slice`: with the
                  flag the table is re-fitted on each slice, so the study and the deployment
                  share one population model; without it the
                  table is fitted once, at the deployed operating point (gamma = 1), and held
                  fixed across the sweep, so only the mechanistic model is biased. The
                  `table_fitted_on` column records which run a file is.
    mechanistic   the per-cycle controller (model re-fitted on a fresh draw every cycle).
    mech_once     "ODE, one solve": the deployed model solved once on the lead-in draw, its
                  recommendation played every scored cycle. It has exactly the hybrid's
                  information and no learning, so the gap between the two columns is what
                  Theorem 2's reward feedback buys at each bias, and the gap to the
                  mechanistic column is what re-solving the biased model every cycle costs.

All three policies see the same deployed (biased) model, the same `env` and the same lead-in
state, so the comparison is exactly paired. The model-free policies (uninformed TS and BSA)
never read the model, so their regret is a single number across the whole sweep; it is
computed once and stored with every row as a reference.

Writes outputs/bias_sweep.csv, which scripts/plot_bias_figures.py reads. `run_example.py`
produces the gamma = 1 row of this sweep and nothing else.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from twincert.generator import folfox as F
from twincert.experiment import mp_bandit as MB
from twincert.experiment.protocol import (
    deploy_model, eval_cohort, eval_leadin, study_cohort)
from twincert.experiment.runin import RunInSpec
from twincert.experiment.stats import (
    boot_indices, boot_ratio_from_indices, summ)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--n-study", type=int, default=2000)
    ap.add_argument("--cycles", type=int, default=11)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--study-seed", type=int, default=2024)
    ap.add_argument("--frac-step", type=float, default=0.05)
    ap.add_argument("--frac-min", type=float, default=0.20)
    ap.add_argument("--fit-table-on-slice", action="store_true",
                    help="fit the hybrid's prior table on the calibration slice of the "
                         "deployed model. Default: the table fitted on the full study cohort "
                         "(gamma = 1) is used for every slice. Only the hybrid column "
                         "changes; the table_fitted_on column records the choice")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    spec = F.FolfoxSpec()
    K, N = spec.n_arms, a.cycles
    rs = RunInSpec(n_cycles=1)

    # Both cohorts are drawn once (integrating the ODE is the expensive part) and only the
    # calibration, the lead-in and the policies are re-run per slice.
    sc = study_cohort(spec, a.n_study, a.study_seed, rs)
    s_coh, s_p, s_pi_star = sc.coh, sc.p, sc.pi_star
    # B_mu and R_mech are study-cohort quantities, so the figure's x-axis is measurable
    # before any evaluation patient is treated. Only the regret columns use `coh`.
    s_w = MB.occupancy_weights(s_pi_star, K)

    ec = eval_cohort(spec, a.n, a.seed, N, rs)
    coh, auc, conc, p_true, pi_star, env = ec.coh, ec.auc, ec.conc, ec.p, ec.pi_star, ec.env

    # The model-free references, once (they cannot depend on the calibration slice).
    # `calibrate` with its defaults is the everyone slice (gamma = 1): the deployed operating
    # point. Without --fit-table-on-slice this is the one place the hybrid's prior table is
    # fitted, and that table is carried unchanged into every biased slice below; with the
    # flag each slice fits its own.
    dep_ref = deploy_model(spec, sc, a.study_seed, rs=rs)
    cl_pop_ref, study = dep_ref.cl_pop_hat, dep_ref.study
    state_ref = eval_leadin(spec, ec, 1.0, rs)
    refs = {}
    for pol in ("uninformed", "bsa"):
        r, _ = MB.run_policy(pol, spec, coh, auc, conc, p_true, pi_star, env, N,
                             cl_pop_ref, 1.0, runin=state_ref)
        refs[pol] = summ(r)

    fracs = [round(f, 2) for f in np.arange(1.0, a.frac_min - 1e-9, -a.frac_step)]
    settings = [("low", 1.0)] + [(s, f) for f in fracs if f < 1.0 for s in ("low", "high")]

    rows = []
    for side, frac in settings:
        # Under --fit-table-on-slice this slice fits its own table, on the lead-in of the
        # model it deploys; otherwise the hybrid carries the gamma = 1 table computed above
        # and the dose-ranging study's outcome draws are skipped (`with_table=False`).
        dep = deploy_model(spec, sc, a.study_seed, frac=frac, side=side, rs=rs,
                           with_table=a.fit_table_on_slice)
        cl_pop, gamma, s_state = dep.cl_pop_hat, dep.gamma, dep.state
        state = eval_leadin(spec, ec, gamma, rs)
        # Measured on the study cohort under the deployed (biased) model: `s_state` already
        # carries this slice's cl_pop / gamma.
        B = MB.occupancy_weighted_bias_centred(
            s_p, MB.model_arm_values(spec, s_coh.bsa, s_state.cl_hat), s_w)
        info = MB.information_quantities(s_pi_star, s_state.pi_hat, K)
        table = dep.study if a.fit_table_on_slice else study
        if frac == 1.0 and a.fit_table_on_slice:
            # The everyone slice is the gamma = 1 deployment, so the flag must be a no-op
            # here; Q is drawn from its own generator and never reads the lead-in.
            assert np.array_equal(table["table"], study["table"]), (
                "the everyone slice must reproduce the gamma = 1 prior table exactly")
            assert np.array_equal(table["Q"], study["Q"]), (
                "the dose-ranging outcomes must not depend on which lead-in fitted the table")
        hyb_x = MB.run_policy("hybrid", spec, coh, auc, conc, p_true, pi_star, env, N,
                              cl_pop, gamma, study=table, runin=state)[0]
        mech_x = MB.run_policy("mechanistic", spec, coh, auc, conc, p_true, pi_star,
                               env, N, cl_pop, gamma, runin=state)[0]
        # The same deployed model, solved once on the lead-in and then played: no reward, no
        # further reading, no random number, so this column cannot disturb the two above.
        once_x = MB.run_policy("mech_once", spec, coh, auc, conc, p_true, pi_star,
                               env, N, cl_pop, gamma, runin=state)[0]
        hyb, mech, once = summ(hyb_x), summ(mech_x), summ(once_x)
        # One resample per slice, shared by both ratios against the hybrid, so they are
        # paired with each other too.
        bidx = boot_indices(a.n, np.random.default_rng(a.seed + 1000))
        r_lo, r_hi = boot_ratio_from_indices(mech_x, hyb_x, bidx)
        o_lo, o_hi = boot_ratio_from_indices(once_x, hyb_x, bidx)
        rows.append({"side": side, "frac": frac, "gamma": round(gamma, 4),
                     "B_mu_centred": round(float(B), 4),
                     "R_mech": round(float(info["R_mech"]), 4),
                     "hybrid": round(hyb[0], 4), "hybrid_ci95": round(hyb[1], 4),
                     "mechanistic": round(mech[0], 4),
                     "mechanistic_ci95": round(mech[1], 4),
                     "ratio_mech_hyb": round(mech[0] / hyb[0], 4),
                     "ratio_lo": round(r_lo, 4), "ratio_hi": round(r_hi, 4),
                     "uninformed": round(refs["uninformed"][0], 4),
                     "bsa": round(refs["bsa"][0], 4),
                     "ratio_hyb_bsa": round(hyb[0] / refs["bsa"][0], 4),
                     "ratio_hyb_uninformed": round(hyb[0] / refs["uninformed"][0], 4),
                     "mech_once": round(once[0], 4),
                     "mech_once_ci95": round(once[1], 4),
                     "ratio_mech_once_hyb": round(once[0] / hyb[0], 4),
                     "ratio_mech_once_lo": round(o_lo, 4),
                     "ratio_mech_once_hi": round(o_hi, 4),
                     "ratio_mech_once_bsa": round(once[0] / refs["bsa"][0], 4),
                     "ratio_mech_once_uninformed":
                         round(once[0] / refs["uninformed"][0], 4),
                     # Which population fitted the hybrid's prior table: the two runs differ
                     # in the hybrid column and nothing else.
                     "table_fitted_on": "slice" if a.fit_table_on_slice else "full"})
        print(f"{side:4s} {frac:.2f}: gamma={gamma:.3f} B={B:.4f} "
              f"hybrid={hyb[0]:.3f} mech_once={once[0]:.3f} mech={mech[0]:.3f}", flush=True)

    with open(out / "bias_sweep.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {out}/bias_sweep.csv ({len(rows)} slices)")


if __name__ == "__main__":
    main()
