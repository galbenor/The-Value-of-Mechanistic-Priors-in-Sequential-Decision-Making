"""The hybrid without an oracle: a Gaussian prior calibrated on a titrated cohort.


The submitted hybrid takes two inputs from an offline study (Appendix H.6):

  the prior table  P(pi* | pi_hat)   needs every study patient's TRUE optimal arm;
  the matrix Q                       needs, in addition, one outcome at EVERY arm for every
                                     study patient (counterfactuals).

Neither exists in an observational cohort. This script replaces both and measures the cost.

  Q      comes from the deployed one-compartment model alone. If arm k is the optimum, dose
         proportionality puts the exposure at arm j at target * D_j / D_k, and the
         between-cycle variability gives the probability that it lands in the window.
         No patient data.

  prior  is a Gaussian on the patient's log clearance, centred on the model's estimate from
         the lead-in blood draw. Its two parameters, a centre offset and a width, are
         estimated on a small therapeutic-drug-monitoring (TDM) cohort.

What the TDM cohort is. No real cohort is available here, so what such a cohort records is
simulated: each study patient is titrated by the published dose-adjustment table of Kaldate
et al. (The Oncologist 17:296-302, 2012, Table 5) applied to assay-noisy AUC readings, and
the dose at which the course settles is kept. That settled dose is a noisy proxy for the
optimal arm, obtained without a true optimum and without a counterfactual. The calibration
then compares it with what the model recommended for the same patient:

    e_i = log(clearance at which the settled dose would be the model's optimum)
          - log(the model's clearance estimate from the lead-in draw)
    offset = mean(e),   width = std(e)

What the plant is used for: scoring regret, and nothing else that enters the method.

The TDM cohorts are drawn independently of the paper's study cohort and of the evaluation
cohort. The study cohort is used only to define the paper's calibration slices and to run the
submitted (oracle) construction for comparison.

Reported: regret over 11 scored cycles at correct calibration and on the paper's nine
calibration slices, averaged over evaluation seeds and over independent draws of the
calibration cohort (default: 300 patients). A second table crosses the two priors (the
paper's oracle table, the calibrated Gaussian) with the two likelihoods (the paper's oracle
Q, the model's Q), to separate what each replacement costs.

Requires `model_shape_matrix` in mp_bandit.py and the `q_source` argument of `deploy_model`
in protocol.py; both Q matrices are requested from the package explicitly, so the results do
not depend on the package default.

Usage, from the repository root:
    python scripts/run_calibrated_gaussian.py
    python scripts/run_calibrated_gaussian.py --n-cal 300 --out outputs
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# Kaldate et al. (2012), Table 5: (lowest integer AUC of the bin in mg.h/L, dose change in
# mg/m2). The printed table gives magnitudes per bin and "no change" for 20-30; the paper's
# text states a maximum increase of 727 mg/m2 for AUC of 10 or below. Readings below 8 are
# not tabulated and are given that maximum increase here.
KALDATE_TABLE_5 = ((40, -727.0), (37, -582.0), (34, -436.0), (31, -291.0), (20, 0.0),
                   (17, 291.0), (14, 436.0), (11, 582.0), (-10 ** 9, 727.0))


def main() -> None:
    here = Path(__file__).resolve().parent
    default_src = here.parent / "src"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-src", default=str(default_src),
                    help="the repository's src/ directory (default: ../src)")
    ap.add_argument("--n", type=int, default=10000, help="evaluation patients per seed")
    ap.add_argument("--n-study", type=int, default=2000, help="the paper's study cohort")
    ap.add_argument("--n-cal", type=int, default=300, help="patients in the TDM cohort")
    ap.add_argument("--cal-draws", type=int, default=5,
                    help="independent draws of the TDM cohort")
    ap.add_argument("--course", type=int, default=12, help="cycles in a titration course")
    ap.add_argument("--cycles", type=int, default=11, help="scored cycles")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13],
                    help="evaluation seeds; 11 is the paper's")
    ap.add_argument("--study-seed", type=int, default=2024)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    sys.path.insert(0, a.repo_src)

    from twincert.generator import folfox as F
    from twincert.experiment import mp_bandit as B
    from twincert.experiment.protocol import (build_study, deploy_model, eval_cohort,
                                              eval_leadin, study_cohort)
    from twincert.experiment.runin import RunInSpec, run_runin

    spec = F.FolfoxSpec()
    K, N = spec.n_arms, a.cycles
    rs = RunInSpec(n_cycles=1)
    d = spec.dose_levels
    target = np.sqrt(spec.auc_lo * spec.auc_hi)

    # ------------------------------------------------------------- input 1: Q, from the model
    Q_model = B.model_shape_matrix(spec)

    # ------------------------------------------------------------- the titration rule
    def next_arm(arm, auc_reading):
        """Kaldate Table 5: the arm for the next cycle from the AUC the laboratory reported.
        The adjusted dose is snapped to the nearest arm (arms are 286 mg/m2 apart; the
        table's smallest step is 291 mg/m2)."""
        r = np.rint(auc_reading)
        delta = np.zeros_like(r)
        done = np.zeros(r.shape, bool)
        for lo, change in KALDATE_TABLE_5:
            hit = (~done) & (r >= lo)
            delta[hit], done[hit] = change, True
        return np.abs(d[None, :] - (d[arm] + delta)[:, None]).argmin(axis=1)

    def settled_dose(sc, rng):
        """Titrate the study cohort and return the arm each course settles at: the arm most
        often given in the last six cycles (ties go to the final dose). Cycle 1 is the
        lead-in itself, so the course starts from the same blood draw the model reads."""
        n = len(sc.coh)
        idx = np.arange(n)
        arm = np.full(n, spec.bsa_arm)
        arms = np.zeros((n, a.course), int)
        for t in range(a.course):
            if t == 0:
                noise = sc.renv["cycle_noise"][:, 0] + sc.renv["assay_noise"][:, 0]
            else:
                noise = rng.normal(0, spec.sigma_cycle, n) + rng.normal(0, spec.sigma_assay, n)
            arms[:, t] = arm
            arm = next_arm(arm, sc.auc[idx, arm] * np.exp(noise))
        tail = arms[:, -min(6, a.course - 1):]        # never count the lead-in cycle
        counts = np.stack([(tail == k).sum(axis=1) for k in range(K)], axis=1).astype(float)
        counts[idx, arms[:, -1]] += 0.5
        return counts.argmax(axis=1)

    # ------------------------------------------------------------- input 2: the prior
    def calibrate(bsa, settled, cl_hat):
        """(offset, width) of the model's error in log clearance, from a TDM cohort."""
        e = np.log(d[settled] * bsa / target) - np.log(cl_hat)
        return float(e.mean()), float(e.std())

    def gaussian_prior(bsa, cl_hat, offset, width, nodes: int = 81, floor: float = 1e-3):
        """N(log cl_hat + offset, width^2) on log clearance -> a prior over arms per patient.

        Each quadrature node is a clearance the patient might have; the model says which arm
        is best for it; the node's Gaussian weight goes to that arm. The floor keeps every
        arm possible, as Laplace smoothing does for the paper's table."""
        z = np.linspace(-4.0, 4.0, nodes)
        w = np.exp(-0.5 * z ** 2)
        w /= w.sum()
        prior = np.zeros((len(bsa), K))
        idx = np.arange(len(bsa))
        for zi, wi in zip(z, w):
            arm = B.model_recommendation(spec, bsa, cl_hat * np.exp(offset + width * zi))
            np.add.at(prior, (idx, arm), wi)
        prior += floor
        return prior / prior.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------- the study side
    # The paper's 2000-patient study cohort defines the calibration slices and feeds the
    # submitted oracle construction (the comparison rows). It plays no part in the new prior.
    sc, _ = build_study(spec, a.n_study, a.study_seed, rs=rs)

    # The TDM cohorts: independent draws of `n_cal` new patients each, with their own lead-in
    # draw and their own titration course. Only body surface area, the lead-in measurement
    # and the settled dose are taken from them.
    tdm = []
    for j in range(a.cal_draws):
        c = study_cohort(spec, a.n_cal, a.study_seed + 5000 + j, rs)
        tdm.append((c, settled_dose(c, np.random.default_rng(a.study_seed + 31337 + j))))
    all_settled = np.concatenate([lab for _, lab in tdm])
    all_truth = np.concatenate([c.pi_star for c, _ in tdm])
    label_exact = float(np.mean(all_settled == all_truth))
    label_within1 = float(np.mean(np.abs(all_settled - all_truth) <= 1))

    slices = [("low", 1.0)] + [("low", f) for f in (0.2, 0.4, 0.6, 0.8)] + \
             [("high", f) for f in (0.8, 0.6, 0.4, 0.2)]
    rows, cal_log = [], []
    for seed in a.seeds:
        ec = eval_cohort(spec, a.n, seed, N, rs)
        idx = np.arange(len(ec.coh))

        for side, frac in slices:
            dep = deploy_model(spec, sc, a.study_seed, frac, side, rs, q_source="oracle")
            st = eval_leadin(spec, ec, dep.gamma, rs)
            name = ("all" if frac == 1.0 else
                    f"{'slowest' if side == 'low' else 'fastest'} {int(frac * 100)}%")

            def run(policy, study=None, runin=st):
                r, _ = B.run_policy(policy, spec, ec.coh, ec.auc, ec.conc, ec.p, ec.pi_star,
                                    ec.env, N, dep.cl_pop_hat, dep.gamma, study=study,
                                    runin=runin)
                return float(r.mean())

            base = {"bsa": run("bsa"), "uninformed_ts": run("uninformed"),
                    "ode_once": run("mech_once"),
                    "hybrid_oracle_H6": run("hybrid", dep.study),
                    # the paper's oracle table, with Q swapped for the model's
                    "oracle_table_model_Q": run("hybrid", {"Q": Q_model,
                                                           "table": dep.study["table"]})}
            # The hybrid reads `table[pi_hat]`; a per-patient table with pi_hat = 0..n-1
            # changes the prior and nothing else in the algorithm.
            per_patient = replace(st, pi_hat=idx)
            for j, (c, lab) in enumerate(tdm):
                # what the deployed model (this slice's calibration) estimates for the TDM
                # patients from their lead-in draw
                m = run_runin(spec, c.coh, c.auc, c.conc, c.p, c.pi_star, c.renv, dep.gamma, rs)
                off, width = calibrate(c.coh.bsa, lab, m.cl_hat)
                prior = gaussian_prior(ec.coh.bsa, st.cl_hat, off, width)
                reg = run("hybrid", {"Q": Q_model, "table": prior}, per_patient)
                # the same prior with the paper's oracle Q (true plant, counterfactuals)
                reg_oq = run("hybrid", {"Q": dep.study["Q"], "table": prior}, per_patient)
                rows.append({"eval_seed": seed, "slice": name, "gamma": dep.gamma,
                             "cal_draw": j, "offset": off, "width": width,
                             "hybrid_calibrated_gaussian": reg,
                             "calibrated_gaussian_oracle_Q": reg_oq, **base})
                if seed == a.seeds[0]:
                    cal_log.append((name, dep.gamma, j, off, width))

    # ------------------------------------------------------------- outputs
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "calibrated_gaussian.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    names = list(dict.fromkeys(r["slice"] for r in rows))
    cols = ("hybrid_oracle_H6", "hybrid_calibrated_gaussian", "ode_once",
            "uninformed_ts", "bsa")

    def stat(name, col):
        v = np.array([r[col] for r in rows if r["slice"] == name])
        return v.mean(), v.min(), v.max()

    lines = [f"Calibrated Gaussian prior on a TDM cohort of {a.n_cal} patients "
             f"({a.cal_draws} independent cohorts), model-implied Q.",
             f"Evaluation: {len(a.seeds)} seeds x {a.n} patients, {N} scored cycles. "
             f"Cell = mean [min, max] over seeds and draws.",
             f"Settled dose vs true optimal arm (TDM cohorts, pooled): exact {label_exact:.3f}, "
             f"within one arm {label_within1:.3f}.", ""]
    lines.append(f"{'slice':<13s}{'gamma':>7s}" + "".join(f"{c:>31s}" for c in cols))
    for nm in names:
        g = next(r["gamma"] for r in rows if r["slice"] == nm)
        cells = "".join(f"{m:>15.3f}  [{lo:.3f}, {hi:.3f}]"
                        for m, lo, hi in (stat(nm, c) for c in cols))
        lines.append(f"{nm:<13s}{g:>7.3f}{cells}")
    lines.append("")
    lines.append("Calibration parameters by slice (first evaluation seed; mean over draws):")
    for nm in names:
        sel = [c for c in cal_log if c[0] == nm]
        lines.append(f"  {nm:<13s} gamma {sel[0][1]:.3f}   offset {np.mean([c[3] for c in sel]):+.3f}"
                     f"   width {np.mean([c[4] for c in sel]):.3f}"
                     f"   (-log gamma = {-np.log(sel[0][1]):+.3f})")

    cal_all = [stat(nm, "hybrid_calibrated_gaussian") for nm in names]
    h6_all = [stat(nm, "hybrid_oracle_H6") for nm in names]
    ode_all = [stat(nm, "ode_once") for nm in names]
    c0, h0, o0 = cal_all[0], h6_all[0], ode_all[0]
    un, bs = stat("all", "uninformed_ts"), stat("all", "bsa")

    def rng_txt(stats):
        lo, hi = min(s[0] for s in stats), max(s[0] for s in stats)
        return f"{lo:.2f}" if f"{lo:.2f}" == f"{hi:.2f}" else f"{lo:.2f} to {hi:.2f}"

    md = ["| prior and likelihood | needs | regret, calibrated model | regret, nine calibration slices |",
          "|---|---|---|---|",
          f"| H.6 table and Q (submitted) | true optimal arms, counterfactuals | {h0[0]:.2f} | {rng_txt(h6_all)} |",
          f"| Gaussian prior calibrated on {a.n_cal} titrated patients; model Q | a TDM cohort | "
          f"{c0[0]:.2f} | {rng_txt(cal_all)} |",
          f"| one ODE solve, no learning | | {o0[0]:.2f} | {rng_txt(ode_all)} |",
          "",
          f"References: uninformed TS {un[0]:.2f}, BSA {bs[0]:.2f}.",
          f"Spread of the calibrated-Gaussian row over cohort draws and seeds, calibrated model: "
          f"[{c0[1]:.3f}, {c0[2]:.3f}].",
          f"Settled dose: exact for {100 * label_exact:.0f}% of TDM patients, within one arm for "
          f"{100 * label_within1:.0f}%."]
    # The 2 x 2: which prior, which likelihood. "Oracle" inputs use the true optimal arms
    # (and, for Q, counterfactual outcomes on the true two-compartment plant).
    def cell(col):
        st_ = [stat(nm, col) for nm in names]
        return f"{st_[0][0]:.2f} ({rng_txt(st_)})"
    md += ["",
           "Prior by likelihood: regret at the calibrated model (range over the nine slices)",
           "",
           "| | oracle Q (true plant, counterfactuals) | model Q (one-compartment, no data) |",
           "|---|---|---|",
           f"| oracle prior table (true optimal arms) | {cell('hybrid_oracle_H6')} | "
           f"{cell('oracle_table_model_Q')} |",
           f"| Gaussian prior calibrated on {a.n_cal} titrated patients | "
           f"{cell('calibrated_gaussian_oracle_Q')} | {cell('hybrid_calibrated_gaussian')} |"]
    (out / "calibrated_gaussian_report.txt").write_text("\n".join(lines) + "\n")
    (out / "calibrated_gaussian_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(lines))
    print("\n" + "\n".join(md))
    print(f"\nwrote {out}/calibrated_gaussian.csv, calibrated_gaussian_report.txt, "
          f"calibrated_gaussian_table.md")


if __name__ == "__main__":
    main()
