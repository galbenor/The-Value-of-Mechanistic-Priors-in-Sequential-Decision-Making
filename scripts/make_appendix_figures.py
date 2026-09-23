"""Sensitivity analysis of the certificate: every constant measured, every sweep run.

Every constant is measured on the study cohort, the same discipline `run_example.py` follows:
the certificate has to be checkable before any evaluation patient is treated.

Sweeps, all executed on the plant:
    slice   which fraction/tail of the calibration study the model's clearance was fitted on
            (the miscalibration axis) -> B_mu, C(B_mu), certificate verdicts
    K       the discretisation choice -> H(mu), L_h, B_mu, B_crit
    rel     the finite-difference perturbation used for the L_h slope -> stability of L_h
    seed    the study-cohort draw -> sampling stability of every constant
    s       the working-point constant of Theorem 3 -> B_crit(s) curve
    noise   the form and magnitude of the reward's within-patient noise

Outputs: outputs/appendix_sensitivity.json (all numbers) and outputs/fig_sensitivity_new.pdf
(the two-panel figure); both directories are created if missing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twincert.generator import folfox as F
from twincert.experiment import mp_bandit as MB
from twincert.experiment.protocol import build_study

from _plotstyle import CERT, GRID, INK, MUTED, SURFACE_WHITE as SURFACE


#: The working point is one cycle out of the full 12-cycle FOLFOX course (1 lead-in +
#: 11 scored), i.e. Theorem 3 at s = 12/11.
N_COURSE = 12


def measure(spec, n_study=2000, study_seed=2024, frac=1.0, side="low", rel=0.02):
    """One full measurement of the certificate constants at a given configuration.

    Measured entirely on the study cohort, so the certificate is a pre-trial object rather
    than a statistic of the patients whose regret is later scored. No evaluation cohort is
    drawn here at all.
    """
    K = spec.n_arms
    # `with_table=False`: nothing here runs the hybrid, so the dose-ranging study's outcome
    # draws are skipped.
    sc, dep = build_study(spec, n_study, study_seed, frac=frac, side=side,
                          with_table=False)
    coh, conc, p, pi = sc.coh, sc.conc, sc.p, sc.pi_star
    gamma, st = dep.gamma, dep.state
    w = MB.occupancy_weights(pi, K)
    mom = MB.measurement_moments(spec, coh, conc, p, rel=rel)
    info = MB.information_quantities(pi, st.pi_hat, K)
    mv = MB.model_arm_values(spec, coh.bsa, st.cl_hat)
    # The Eq. (3) bias is centred; it is the primary constant everywhere.
    B = MB.occupancy_weighted_bias_centred(p, mv, w)
    B_unc = MB.occupancy_weighted_bias_per_patient(p, mv, w)
    H, sh, L = info["H_mu"], mom["sigma_h"], mom["L_h_mu"]
    return {"gamma": gamma, "H_mu": H, "R_mech": info["R_mech"], "B_mu": B,
            "B_mu_uncentred": B_unc, "sigma_nu": mom["sigma_nu"],
            "C_nu": MB.capacity_nu(mom["sigma_h"], mom["sigma_nu"], 1, H),
            "sigma_h": sh, "L_h_mu": L, "L_h_sup": mom["L_h"], "g_mu": mom["g_mu"],
            "C": MB.capacity_bound(B, sh, L, 1),
            "B_crit_cycle": MB.critical_bias(H, sh, L, N_COURSE, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--figdir", default="outputs",
                    help="directory for the figure; created if missing")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    figdir = Path(a.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    base_spec = F.FolfoxSpec()
    base = measure(base_spec)
    res["base"] = base
    print("base:", {k: round(v, 4) for k, v in base.items()})

    print("\nslice sweep (the miscalibration axis):")
    res["slice"] = []
    for frac in (1.0, 0.8, 0.6, 0.4, 0.2):
        for side in (("low",) if frac == 1.0 else ("low", "high")):
            r = measure(base_spec, frac=frac, side=side)
            r.update(frac=frac, side=side)
            res["slice"].append(r)
            print(f"  {side:4s} {frac:.0%}: gamma={r['gamma']:.3f}  B_mu={r['B_mu']:.3f}  "
                  f"C={r['C']:.3f}  vs one-cycle B_crit {r['B_crit_cycle']:.3f}")

    print("\nK sweep (discretization):")
    res["K"] = []
    for K in (4, 6, 8, 10, 12, 16):
        r = measure(F.FolfoxSpec(n_arms=K))
        r.update(K=K)
        res["K"].append(r)
        print(f"  K={K:2d}: H={r['H_mu']:.3f}  L_h={r['L_h_mu']:.3f}  B_mu={r['B_mu']:.3f}  "
              f"C={r['C']:.3f}  B_crit(cycle)={r['B_crit_cycle']:.3f}")

    print("\nrel sweep (slope perturbation size):")
    res["rel"] = []
    for rel in (0.005, 0.01, 0.02, 0.04):
        r = measure(base_spec, rel=rel)
        r.update(rel=rel)
        res["rel"].append(r)
        print(f"  rel={rel:.3f}: L_h={r['L_h_mu']:.4f}  C={r['C']:.3f}")

    # The certificate is a study-cohort quantity, so the sampling-stability sweep resamples
    # the study cohort; the evaluation seed would leave every constant untouched.
    print("\nseed sweep (study-cohort draw):")
    res["seed"] = []
    for seed in (2024, 2025, 2047, 2101):
        r = measure(base_spec, study_seed=seed)
        r.update(seed=seed)
        res["seed"].append(r)
        print(f"  seed={seed:3d}: sigma_h={r['sigma_h']:.4f}  L_h={r['L_h_mu']:.4f}  "
              f"B_mu={r['B_mu']:.4f}  R={r['R_mech']:.3f}  C={r['C']:.3f}")

    print("\nnoise-form sweep (the reward's within-patient noise: form and magnitude):")
    # The reward integrates the biological channel through a log-normal factor e^eps
    # (inter-occasion variability entering multiplicatively, Tan's own IIV/IOV convention).
    # The literature's proportional residual-error form a*(1+eps) applies to the observed
    # concentration, not to the exposure; this sweep measures what the reward-side choice is
    # worth.
    from scipy.special import ndtr
    coh_nf = F.sample_patients(base_spec, 10000, np.random.default_rng(11))
    auc_nf, _ = F.arm_exposures(base_spec, coh_nf)
    a_nf = np.maximum(auc_nf, 1e-9)
    pi_base = MB.arm_reward_probs(base_spec, auc_nf).argmax(axis=1)
    res["noise_form"] = []
    for sig in (0.12, 0.20):
        for form in ("lognormal", "proportional"):
            if form == "lognormal":
                p = (ndtr(np.log(base_spec.auc_hi / a_nf) / sig)
                     - ndtr(np.log(base_spec.auc_lo / a_nf) / sig))
            else:
                p = (ndtr((base_spec.auc_hi / a_nf - 1.0) / sig)
                     - ndtr((base_spec.auc_lo / a_nf - 1.0) / sig))
            pi = p.argmax(axis=1)
            occ = np.bincount(pi, minlength=base_spec.n_arms) / pi.size
            H_nf = float(-(occ[occ > 0] * np.log(occ[occ > 0])).sum())
            row = {"sigma": sig, "form": form,
                   "p_opt": float(p.max(axis=1).mean()),
                   "p_bsa": float(p[:, base_spec.bsa_arm].mean()),
                   "H_mu": H_nf,
                   "argmax_agrees_with_shipped": float((pi == pi_base).mean())}
            res["noise_form"].append(row)
            print(f"  sigma={sig:.2f} {form:12s}: p_opt={row['p_opt']:.3f}  "
                  f"p_bsa={row['p_bsa']:.3f}  H={H_nf:.4f}  "
                  f"agree={row['argmax_agrees_with_shipped']:.1%}")

    (out / "appendix_sensitivity.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {out}/appendix_sensitivity.json")

    # ---- the two-panel figure ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), facecolor=SURFACE)

    ax = axes[0]
    H, sh, L, B = base["H_mu"], base["sigma_h"], base["L_h_mu"], base["B_mu"]
    s_grid = np.linspace(1.02, 3.2, 300)
    # Theorem 3 at each working point s: B_crit(s) is critical_bias_at((1 - 1/s) H, ...).
    bc = np.array([MB.critical_bias_at((1.0 - 1.0 / sv) * H, sh, L, 1) for sv in s_grid])
    ax.plot(s_grid, bc, color=INK, lw=1.8)
    ax.axhline(B, color=MUTED, lw=1.3, ls=(0, (4, 3)))
    ax.annotate(f"measured $B_\\mu$ = {B:.3f}", (2.6, B), xytext=(0, 5),
                textcoords="offset points", color=MUTED, fontsize=9, ha="center")
    # The one-cycle working point: one cycle out of N (Thm. 3 at s = N/(N-1)).
    s_pt = N_COURSE / (N_COURSE - 1)
    b_pt = MB.critical_bias_at((1.0 - 1.0 / s_pt) * H, sh, L, 1)
    ax.plot([s_pt], [b_pt], "o", color=CERT, ms=6, zorder=3)
    ax.annotate(f"one cycle\n$s=N/(N-1)$\n$B^{{crit}}$={b_pt:.2f}", (s_pt, b_pt),
                xytext=(8, 6), textcoords="offset points", color=CERT, fontsize=8.5)
    ax.set_xlabel("working constant $s$ (Thm. 3)", fontsize=10)
    ax.set_ylabel("$B_\\mu^{crit}(s)$", fontsize=10)
    ax.set_title("(a) the certificate across working points", loc="left", fontsize=11)
    ax.grid(color=GRID, lw=0.5)
    ax.set_ylim(0, 1.6)

    ax = axes[1]
    sl = sorted(res["slice"], key=lambda r: r["gamma"])
    g = [r["gamma"] for r in sl]
    Bs = [r["B_mu"] for r in sl]
    ax.plot(g, Bs, "-o", color=INK, lw=1.8, ms=4)
    ax.axhline(base["B_crit_cycle"], color=CERT, lw=1.5, ls=(0, (5, 3)))
    # Labelled below the line: at this y the line sits at the top of the axes.
    ax.annotate(f"one-cycle $B^{{crit}}$ = {base['B_crit_cycle']:.2f}",
                (g[0], base["B_crit_cycle"]), xytext=(4, -6), textcoords="offset points",
                color=CERT, fontsize=9, va="top")
    for r in sl:
        ax.annotate(f"{r['frac']:.0%}", (r["gamma"], r["B_mu"]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xlabel("$\\gamma$ = calibrated / true clearance (labels: calibration slice)",
                  fontsize=10)
    ax.set_ylabel("measured $B_\\mu$", fontsize=10)
    ax.set_title("(b) measured bias under miscalibration", loc="left", fontsize=11)
    ax.grid(color=GRID, lw=0.5)

    fig.tight_layout()
    figpath = figdir / "fig_sensitivity_new.pdf"
    fig.savefig(figpath, bbox_inches="tight")
    print(f"wrote {figpath}")


if __name__ == "__main__":
    main()
