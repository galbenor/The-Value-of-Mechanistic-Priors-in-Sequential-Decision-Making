"""The two by-bias figures: model distance and regret against measured bias.

  fig_model_distance.pdf   How far the deployed one-compartment model is from the
                           two-compartment world, per calibration slice: the model-world AUC
                           prediction error across the dose range, one line per slice, each
                           labelled with its gamma and its measured centred B_mu. Recomputed
                           live from the plant (median-clearance patient, noise-free draw);
                           the B_mu labels come from outputs/appendix_sensitivity.json.

  fig_hybrid_bias.pdf      What that distance costs: cumulative regret of the hybrid
                           (Theorem 2's algorithm), of the ODE solved once on the lead-in and
                           played for the course, and of the every-cycle mechanistic
                           controller, as a function of the measured B_mu, with the
                           model-free references (uninformed TS and BSA) as flat lines. Mean
                           curves with shaded 95% bands over the dense sweep of
                           scripts/run_bias_sweep.py, the two miscalibration directions drawn
                           separately. Which population fitted the hybrid's prior table is
                           the sweep's --fit-table-on-slice flag, recorded in the CSV's
                           `table_fitted_on` column.

Usage:
    python3 scripts/run_bias_sweep.py --fit-table-on-slice   # -> outputs/bias_sweep.csv
    python3 scripts/make_appendix_figures.py                 # -> outputs/appendix_sensitivity.json
    python3 scripts/plot_bias_figures.py                     # reads both, draws the two figures
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twincert.generator import folfox as F

from _plotstyle import COLOUR, INK, MUTED, SLICES, SURFACE_WHITE as SURFACE, _style


def model_distance_figure(out: Path, b_by_gamma: dict) -> None:
    spec = F.FolfoxSpec()
    k_bsa = spec.bsa_arm
    dose_bsa = spec.dose_levels[k_bsa]

    coh = F.sample_patients(spec, 8000, np.random.default_rng(11))
    _, conc = F.arm_exposures(spec, coh)
    cl_app = F.fit_clearance(spec, dose_bsa, coh.bsa, conc[:, k_bsa])
    i = int(np.argsort(cl_app)[4000])
    bsa, vmax = float(coh.bsa[i]), float(coh.vmax[i])

    s_coh = F.sample_patients(spec, 6000, np.random.default_rng(2024))
    _, s_conc = F.arm_exposures(spec, s_coh)

    doses = np.linspace(spec.dose_levels[0] - 100, spec.dose_levels[-1] + 100, 90)
    auc_true = F.integrate(spec, doses, bsa, vmax)[0]

    fig, ax = plt.subplots(figsize=(6.6, 4.3), facecolor=SURFACE)
    ax.axhline(0.0, color=INK, lw=1.3, zorder=3)
    cmap = plt.get_cmap("coolwarm")
    gs = []
    for side, frac in SLICES:
        _, g = F.calibrate(spec, s_coh, s_conc[:, k_bsa], frac=frac, side=side)
        gs.append((float(g), side, frac))
    for g, side, frac in sorted(gs):
        cl_dep = F.fit_clearance(spec, dose_bsa, bsa, conc[i, k_bsa], g)
        err = F.auc_model(spec, doses, bsa, cl_dep) - auc_true
        colour = INK if frac == 1.0 else cmap(0.5 + 0.5 * np.tanh(2.2 * (g - 1.0)))
        b = b_by_gamma.get(round(g, 3))
        lbl = f"$\\gamma$ {g:.2f}" + (f", $B_\\mu$ {b:.2f}" if b else "")
        ax.plot(doses, err, color=colour, lw=2.1 if frac == 1.0 else 1.4,
                zorder=4 if frac == 1.0 else 3)
        ax.annotate(lbl, (doses[-1], err[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7.6,
                    color=INK if frac == 1.0 else colour)
    ax.set_xlim(doses[0], doses[-1] + 700)
    _style(ax, "dose (mg/m$^2$)", "model $-$ world AUC error (mg$\\cdot$h/L)",
           surface=SURFACE)
    fig.tight_layout()
    fig.savefig(out / "fig_model_distance.pdf", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}/fig_model_distance.pdf")


def hybrid_bias_figure(out: Path, sweep_csv: Path) -> None:
    """Dense simulation sweep (scripts/run_bias_sweep.py): mean curves with shaded 95%
    bands, both miscalibration directions drawn separately."""
    rows = list(csv.DictReader(open(sweep_csv)))
    # Everything but the two provenance columns is numeric.
    text_cols = ("side", "table_fitted_on")
    for r in rows:
        for k in r:
            if k not in text_cols:
                r[k] = float(r[k])

    fig, ax = plt.subplots(figsize=(6.6, 4.3), facecolor=SURFACE)
    # Three model-based policies on one bias axis: the hybrid/mech_once gap is what
    # Theorem 2's reward feedback buys, and the mech_once/mechanistic gap is what re-solving a
    # biased model every cycle costs. Colours come from `_plotstyle.COLOUR`.
    curves = [("hybrid", COLOUR["hybrid"], "hybrid TS", None),
              ("mech_once", COLOUR["mech_once"], "ODE solved once, played for the course",
               None),
              ("mechanistic", COLOUR["mechanistic"],
               "mechanistic ODE, re-solved every cycle", None)]
    for pol, colour, lbl, marker in curves:
        for side, ls in (("low", "-"), ("high", (0, (4, 2)))):
            br = sorted([r for r in rows if r["side"] == side or r["frac"] == 1.0],
                        key=lambda r: r["B_mu_centred"])
            B = np.array([r["B_mu_centred"] for r in br])
            y = np.array([r[pol] for r in br])
            ci = np.array([r[f"{pol}_ci95"] for r in br])
            ax.fill_between(B, y - ci, y + ci, color=colour, alpha=0.18, lw=0, zorder=2)
            ax.plot(B, y, color=colour, lw=2.0 if side == "low" else 1.4, ls=ls,
                    marker=marker, ms=3.0, mew=0.0, zorder=4,
                    label=lbl if side == "low" else None)
    Ball = [r["B_mu_centred"] for r in rows]
    # The two flat references are labelled at the left edge, where every curve is still far
    # below them.
    for k, colour, lbl, dy, xa in (("uninformed", COLOUR["uninformed"], "uninformed TS",
                                    5, 0.015),
                                   ("bsa", COLOUR["bsa"], "BSA static", -13, 0.015),
                                   ):
        y = rows[0][k]
        ax.axhline(y, color=colour, lw=1.1, ls=(0, (1, 2.5)), zorder=2)
        ax.annotate(f"{lbl}  ({y:.2f})",
                    (min(Ball) + xa * (max(Ball) - min(Ball)), y), xytext=(2, dy),
                    textcoords="offset points", color=colour, fontsize=8.2)
    ax.plot([], [], color=MUTED, lw=1.2, ls=(0, (4, 2)),
            label="dashed: model fitted on the fastest-clearing slice")
    ax.legend(frameon=False, fontsize=8.6, labelcolor=MUTED, loc="upper left")
    _style(ax, "measured centred bias $B_\\mu$ of the deployed prior (Eq. 3)",
           "cumulative regret, 11 scored cycles", surface=SURFACE)
    fig.tight_layout()
    fig.savefig(out / "fig_hybrid_bias.pdf", facecolor=SURFACE)
    fig.savefig(out / "fig_hybrid_bias.png", facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print(f"wrote {out}/fig_hybrid_bias.pdf (+ .png)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep-csv", default="outputs/bias_sweep.csv",
                    help="the dense sweep written by scripts/run_bias_sweep.py")
    ap.add_argument("--out", default="outputs",
                    help="directory for the figures; created if missing")
    ap.add_argument("--sensitivity", default="outputs/appendix_sensitivity.json")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    b_by_gamma = {}
    sens = Path(a.sensitivity)
    if sens.exists():
        for r in json.loads(sens.read_text())["slice"]:
            b_by_gamma[round(r["gamma"], 3)] = r["B_mu"]

    model_distance_figure(out, b_by_gamma)
    hybrid_bias_figure(out, Path(a.sweep_csv))


if __name__ == "__main__":
    main()
