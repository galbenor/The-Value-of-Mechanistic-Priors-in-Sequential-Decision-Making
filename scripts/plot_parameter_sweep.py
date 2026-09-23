"""The figures of the parameter sweep: three, at two-column width.

Reads what `scripts/run_parameter_sweep.py` wrote into `outputs/param_sweep/` and draws, in
the house style of `scripts/_plotstyle.py`:

    ofat_policies       2 x 2 panels, one per design axis (K, N, n_study, lead-in): the
                        policies' cumulative regret, 95% intervals both as a light band and
                        as capped error bars at each swept point, the operating point as a
                        dashed rule. At the biased deployment (the `make param-sweep` target)
                        every policy is drawn; at gamma = 1 the per-cycle ODE is dropped
                        (see `GAMMA1_DROP`).
    ofat_certificate    the same 2 x 2 panels: R_mech and C(B_mu) in nats on the left axis,
                        the normalised bias B_mu / B_mu^crit on the right against a dashed
                        screen line at 1. C_nu and the per-draw ceiling C(B_mu)/d_Z stay in
                        the CSV and out of the figure.
    channel             regret against mechanistic information at the calibrated model
                        (gamma = 1), over the channel scenarios `run_parameter_sweep.py`
                        builds by moving the pre-dose measurement channel: (a) the policies'
                        regret against the measured R_mech (BSA, the ODE solved once and
                        played, and the hybrid; the per-cycle ODE is in the CSV and not in
                        the panel) and (b) the certificate along the same rows. Reads
                        `channel.csv` and `channel_draws.csv`.

No figure or panel titles (the caption's job); bold panel letters instead; one compact legend
per figure, outside the panels; 8-9 pt type at a 7-inch full-width figure. Nothing is
smoothed, binned or interpolated: every marker is one Monte-Carlo configuration and every
interval is that configuration's own.

Writes PDF (vector) and PNG at 300 dpi into `outputs/param_sweep/figures/`.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from _plotstyle import CERT, COLOUR, GRID, INK, MUTED, SURFACE, _style

#: The operating point of `run_example.py`, per axis: where the dashed rule goes. The key
#: order is also the panel order of the two consolidated figures.
OP_VALUE = {"n_arms": 8, "cycles": 11, "n_study": 2000, "leadin": 1}

#: The x column of each axis, where it differs from the axis name.
X_COL = {"cycles": "n_scored", "leadin": "n_leadin"}

#: Short axis labels; the caption carries the description.
AXIS_LABEL = {"n_arms": "K  (dose arms)",
              "cycles": "N  (scored cycles)",
              "n_study": "calibration-study patients",
              "leadin": "lead-in cycles"}
AXIS_LOG = {"cycles": True, "n_study": True}

#: Short legend entries.
SHORT = {"bsa": "BSA", "uninformed": "uninformed TS", "mechanistic": "mechanistic ODE",
         "mech_once": "ODE, one solve", "hybrid": "hybrid TS"}

#: One compact legend per figure, on an opaque card in the page colour.
LEGEND = {"frameon": True, "framealpha": 0.94, "facecolor": SURFACE, "edgecolor": GRID,
          "labelcolor": MUTED}

POLICIES = ("bsa", "uninformed", "mechanistic", "mech_once", "hybrid")

#: Where the per-cycle ODE is drawn, and where it is only in the CSV. `mechanistic`
#: re-solves the model every cycle on a fresh blood draw, a per-cycle information budget the
#: theory does not assume; at the calibrated model (gamma = 1) it simply wins, so it is
#: dropped from the gamma = 1 panels, where `mech_once` is the ODE comparison that has the
#: hybrid's information. Every number stays in the CSV either way.
GAMMA1_DROP = ("mechanistic",)

#: `_plotstyle.FONT` scaled to 8-9 pt for a two-column layout.
PANEL_SCALE = 0.84
LETTER_FS = 8.5          # the bold panel letter
LEGEND_FS = 8.0

#: Line weights, thin enough for print at full width.
LW, MS, ELW = 1.2, 3.0, 0.8


def read_csv(path: Path) -> list[dict]:
    """Rows as dicts with floats where they parse, so a column name is the only key needed."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            if v in ("True", "False"):
                d[k] = v == "True"
            else:
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    d[k] = v
        out.append(d)
    return out


def collapse(rows: list[dict], xcol: str, cols: list[str]):
    """Group rows by their x value and return (mean, min, max) per column.

    Only the `n_study` axis has more than one row per x (three study seeds); everywhere else
    this is the identity and min = max = mean. The spread drawn for `n_study` is therefore
    the min-max over the three seeds, not a confidence interval.
    """
    xs = sorted({r[xcol] for r in rows})
    out = {c: ([], [], []) for c in cols}
    for x in xs:
        grp = [r for r in rows if r[xcol] == x]
        for c in cols:
            v = np.array([g[c] for g in grp], float)
            out[c][0].append(v.mean())
            out[c][1].append(v.min())
            out[c][2].append(v.max())
    return np.array(xs, float), {c: tuple(np.array(a, float) for a in v)
                                 for c, v in out.items()}


def _any_false(rows: list[dict], xcol: str, xs, flag: str):
    """Per-x mask: True where any row at that x has `flag` False.

    Where an x carries several rows (the three study seeds of `n_study`), the x is marked if
    any single run failed there, which is not the same as the seed-average failing.
    """
    return np.array([any(not r[flag] for r in rows if r[xcol] == x) for x in xs], bool)


def panel_grid(*, figsize=(7.0, 5.0)):
    """The 2 x 2 panels both consolidated figures share: one per design axis, each one
    column wide."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes.ravel()


def _letter(ax, i: int) -> None:
    """The bold panel letter, top-left, inside the axes."""
    ax.text(0.02, 0.97, f"({'abcdefgh'[i]})", transform=ax.transAxes, color=INK,
            fontsize=LETTER_FS, fontweight="bold", va="top", ha="left", zorder=8)


def _tick_the_swept_values(ax, axis: str, x) -> None:
    """Integer-valued and decade-spanning axes get the swept values as their ticks."""
    if AXIS_LOG.get(axis):
        ax.set_xscale("log")
    if AXIS_LOG.get(axis) or axis in ("n_arms", "leadin"):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
        ax.minorticks_off()


# --------------------------------------------------------------------------- #
# (i) the policies
# --------------------------------------------------------------------------- #
def _drop_per_cycle_ode(rows: list[dict]) -> bool:
    """True when every row of this set is the calibrated deployment (gamma = 1)."""
    return bool(rows) and all(float(r["calib_frac"]) == 1.0 for r in rows)


def figure_ofat_policies(sets: dict[str, list[dict]], figdir: Path) -> None:
    """Theorem 2's ordering across every design axis, one panel each, one legend.

    At the calibrated deployment the per-cycle ODE is dropped from the panels and kept in
    the CSV (see `GAMMA1_DROP`); at the biased operating point every policy is drawn.
    """
    drawn = tuple(p for p in POLICIES
                  if p not in GAMMA1_DROP
                  or not all(_drop_per_cycle_ode(r) for r in sets.values()))
    fig, flat = panel_grid()
    for i, (ax, axis) in enumerate(zip(flat, OP_VALUE)):
        rows = sets.get(axis)
        if not rows:
            ax.axis("off")
            continue
        xcol = X_COL.get(axis, axis)
        multi = axis == "n_study"
        x, agg = collapse(rows, xcol, list(drawn) + [f"{p}_ci95" for p in drawn])
        for pol in drawn:
            m, ci = agg[pol][0], agg[f"{pol}_ci95"][0]
            # Where an x carries several runs the band is the min-max over the study seeds
            # and the caps are the mean of those runs' own 95% intervals; elsewhere they
            # coincide.
            lo, hi = (agg[pol][1], agg[pol][2]) if multi else (m - ci, m + ci)
            ax.fill_between(x, lo, hi, color=COLOUR[pol], alpha=0.14, lw=0, zorder=2)
            ax.errorbar(x, m, yerr=ci, fmt="none", ecolor=COLOUR[pol], elinewidth=ELW,
                        capsize=1.8, capthick=ELW, zorder=3)
            ax.plot(x, m, color=COLOUR[pol], lw=LW, marker="o" if pol == "hybrid" else "s",
                    ms=MS, zorder=4)
        ax.axvline(OP_VALUE[axis], color=CERT, lw=0.9, ls="--", alpha=0.6, zorder=1)
        _style(ax, xlabel=AXIS_LABEL[axis],
               ylabel="cumulative regret" if axis in ("n_arms", "n_study") else "",
               scale=PANEL_SCALE)
        _tick_the_swept_values(ax, axis, x)
        _letter(ax, i)

    handles = [Line2D([], [], color=COLOUR[pol], lw=LW,
                      marker="o" if pol == "hybrid" else "s", ms=MS, label=SHORT[pol])
               for pol in ("hybrid", "mech_once", "mechanistic", "uninformed", "bsa")
               if pol in drawn]
    handles.append(Line2D([], [], color=CERT, lw=0.9, ls="--", label="operating point"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=LEGEND_FS,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.4, handlelength=1.8, **LEGEND)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save(fig, figdir / "ofat_policies")


# --------------------------------------------------------------------------- #
# (ii) the certificate
# --------------------------------------------------------------------------- #
def figure_ofat_certificate(sets: dict[str, list[dict]], figdir: Path) -> None:
    """The certificate: R_mech against C(B_mu) in nats, and B_mu / B_mu^crit against 1.

    C_nu is not drawn (it is the estimator-free robustness line, not the certificate) and
    neither is the per-draw ceiling C(B_mu)/d_Z; both stay in the CSV.
    """
    fig, flat = panel_grid()
    for i, (ax, axis) in enumerate(zip(flat, OP_VALUE)):
        rows = sets.get(axis)
        if not rows:
            ax.axis("off")
            continue
        xcol = X_COL.get(axis, axis)
        x, agg = collapse(rows, xcol, ["R_mech", "C_B_mu", "B_mu_centred", "B_crit"])
        ratio = agg["B_mu_centred"][0] / np.maximum(agg["B_crit"][0], 1e-12)

        ax.plot(x, agg["C_B_mu"][0], color="#1baf7a", lw=LW, ls="--", marker="^", ms=MS,
                zorder=3)
        ax.plot(x, agg["R_mech"][0], color=INK, lw=LW, marker="o", ms=MS, zorder=4)
        # A ring wherever a statement fails, read from the rows' own flags rather than from
        # the plotted seed-average.
        bad = _any_false(rows, xcol, x, "chain_holds")
        if bad.any():
            ax.scatter(x[bad], agg["R_mech"][0][bad], s=44, facecolor="none",
                       edgecolor=CERT, lw=1.2, zorder=6)
        # Headroom above the ceiling and below the measurement.
        ylo = float(min(agg["R_mech"][0].min(), agg["C_B_mu"][0].min()))
        yhi = float(max(agg["R_mech"][0].max(), agg["C_B_mu"][0].max()))
        ax.set_ylim(ylo - 0.08 * (yhi - ylo), yhi + 0.20 * (yhi - ylo))
        _style(ax, xlabel=AXIS_LABEL[axis],
               ylabel="nats" if axis in ("n_arms", "n_study") else "", scale=PANEL_SCALE)
        ax.axvline(OP_VALUE[axis], color=CERT, lw=0.9, ls="--", alpha=0.6, zorder=1)
        _tick_the_swept_values(ax, axis, x)
        _letter(ax, i)

        axb = ax.twinx()
        axb.plot(x, ratio, color=CERT, lw=LW, ls="-.", marker="d", ms=MS, zorder=5)
        axb.axhline(1.0, color=CERT, lw=0.7, ls=":", alpha=0.9, zorder=1)
        fail = _any_false(rows, xcol, x, "certified")
        if fail.any():
            axb.scatter(x[fail], ratio[fail], s=44, facecolor="none", edgecolor=CERT,
                        lw=1.2, zorder=6)
        axb.set_ylim(0.0, max(1.15, float(ratio.max()) * 1.15))
        axb.tick_params(colors=CERT, labelsize=9.5 * PANEL_SCALE, length=3)
        axb.spines["top"].set_visible(False)
        axb.spines["right"].set_color(GRID)
        axb.grid(False)
        if axis in ("cycles", "leadin"):        # the right column of the 2 x 2 grid
            axb.set_ylabel(r"$B_\mu / B_\mu^{\mathrm{crit}}$", color=CERT,
                           fontsize=10.5 * PANEL_SCALE)

    handles = [
        Line2D([], [], color=INK, lw=LW, marker="o", ms=MS,
               label=r"$R_{\mathrm{mech}}$ (measured)"),
        Line2D([], [], color="#1baf7a", lw=LW, ls="--", marker="^", ms=MS,
               label=r"$C(B_\mu)$ (Prop. 1)"),
        Line2D([], [], color=CERT, lw=LW, ls="-.", marker="d", ms=MS,
               label=r"$B_\mu/B_\mu^{\mathrm{crit}}$ (right axis)"),
        Line2D([], [], color=CERT, lw=0.7, ls=":", label="Thm. 3 screen at 1"),
        Line2D([], [], color=CERT, lw=0.9, ls="--", label="operating point"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=LEGEND_FS,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handlelength=1.8, **LEGEND)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save(fig, figdir / "ofat_certificate")


# --------------------------------------------------------------------------- #
# (iii) the channel scenarios: regret against mechanistic information
# --------------------------------------------------------------------------- #
#: The pooled-draw scenarios get their own marker everywhere they appear, because they reach
#: their R_mech with a different knob (more draws of the same assay, not a better one).
DRAW_MARKER = "*"

#: The policies panel (a) draws: a gamma = 1 figure, so the per-cycle ODE is dropped
#: (`GAMMA1_DROP`) and `mech_once` is the model-based comparison instead.
CHANNEL_POLICIES = tuple(p for p in ("bsa", "mechanistic", "mech_once", "hybrid")
                         if p not in GAMMA1_DROP)


def _channel_rows(chan: list[dict], draws: list[dict]):
    """(the sigma_assay scenarios at one draw, the pooled-draw scenarios at d_Z > 1).

    The d_Z = 1 row of `channel_draws.csv` is the quoted-sigma row of `channel.csv`, the same
    configuration run twice, so it is dropped here rather than drawn on top of itself.
    """
    one = sorted([r for r in chan if r["d_Z"] == 1], key=lambda r: r["R_mech"])
    pooled = sorted([r for r in draws if r["d_Z"] > 1], key=lambda r: r["R_mech"])
    return one, pooled


def _col(rows, name):
    return np.array([r[name] for r in rows], float)


def _merged_scenario_ticks(x, labels, span: float, min_gap: float = 0.03):
    """Tick positions and labels with near-coincident scenarios merged into one label.

    The channel saturates: below a small sigma_assay the assay is negligible beside the
    patient's own between-cycle variability, so those scenarios land within a hair of each
    other in R_mech and are merged into a single tick ("0.02/0.05") at their midpoint.
    """
    groups: list[list[int]] = [[0]]
    for i in range(1, len(x)):
        if x[i] - x[groups[-1][-1]] < min_gap * span:
            groups[-1].append(i)
        else:
            groups.append([i])
    pos = [float(np.mean([x[i] for i in g])) for g in groups]
    txt = ["/".join(labels[i] for i in g) for g in groups]
    return pos, txt


def figure_channel(chan: list[dict], draws: list[dict], figdir: Path) -> None:
    """Regret and the certificate against the measured mechanistic information R_mech.

    The x axis is R_mech in both panels: the theory is stated in mechanistic information,
    while a trial can only choose a measurement channel. Each marker is one
    channel scenario run end to end at the calibrated model (gamma = 1); its R_mech was
    measured on the study cohort, not set. The top axis of panel (a) carries the sigma_assay
    that produced each point.

    R_mech = 0 is the uninformed learner, not a further scenario: uninformed TS ignores the
    model, so its regret is the same number in every row, drawn as a horizontal rule with its
    value also placed as a marker at R_mech = 0. BSA is flat for the same reason.

    Panel (b) plots C(B_mu)/d_Z, the per-draw ceiling, rather than C(B_mu): `capacity_bound`
    carries a d_Z/2 prefactor while sigma_h stays single-draw, so the pooled-draw points
    would otherwise be plotted on a different convention from the sigma ones. The diagonal
    is R_mech against itself: the ceiling holds wherever the C(B_mu) curve lies above it.
    """
    one, pooled = _channel_rows(chan, draws)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.1))
    fig.patch.set_facecolor(SURFACE)

    # --- (a) the policies against R_mech -----------------------------------------
    x, xp = _col(one, "R_mech"), _col(pooled, "R_mech")
    u_mean = float(np.mean(_col(one, "uninformed")))
    u_ci = float(np.mean(_col(one, "uninformed_ci95")))
    axa.axhspan(u_mean - u_ci, u_mean + u_ci, color=COLOUR["uninformed"], alpha=0.14,
                lw=0, zorder=1)
    axa.axhline(u_mean, color=COLOUR["uninformed"], lw=LW, ls="--", zorder=2)
    # The reference as a point as well: the uninformed learner is the R_mech = 0 member of
    # this family.
    axa.errorbar([0.0], [u_mean], yerr=[u_ci], fmt="o", ms=MS + 0.6,
                 color=COLOUR["uninformed"], ecolor=COLOUR["uninformed"], elinewidth=ELW,
                 capsize=1.8, capthick=ELW, zorder=5)

    for pol in CHANNEL_POLICIES:
        m, ci = _col(one, pol), _col(one, f"{pol}_ci95")
        axa.fill_between(x, m - ci, m + ci, color=COLOUR[pol], alpha=0.14, lw=0, zorder=2)
        axa.errorbar(x, m, yerr=ci, fmt="none", ecolor=COLOUR[pol], elinewidth=ELW,
                     capsize=1.8, capthick=ELW, zorder=3)
        axa.plot(x, m, color=COLOUR[pol], lw=LW, marker="o", ms=MS, zorder=4)
        if pooled:
            mp, cp = _col(pooled, pol), _col(pooled, f"{pol}_ci95")
            axa.errorbar(xp, mp, yerr=cp, fmt=DRAW_MARKER, ms=MS + 2.6,
                         color=COLOUR[pol], ecolor=COLOUR[pol], elinewidth=ELW,
                         capsize=1.8, capthick=ELW, zorder=5)
    _style(axa, xlabel=r"$R_{\mathrm{mech}}$  (nats, measured)",
           ylabel="cumulative regret", scale=PANEL_SCALE)
    axa.set_xlim(-0.03, max(float(x.max()), float(xp.max()) if len(xp) else 0.0) + 0.05)
    # headroom above the uninformed rule, so the panel letter does not sit on it
    ylo, yhi = axa.get_ylim()
    axa.set_ylim(ylo, yhi + 0.14 * (yhi - ylo))
    _letter(axa, 0)
    # the scenario that produced each point, as axis furniture: the top spine ticks at the
    # measured R_mech and is labelled by sigma_assay
    axt = axa.twiny()
    axt.set_xlim(axa.get_xlim())
    pos, txt = _merged_scenario_ticks(x, [f"{r['sigma_assay']:g}" for r in one],
                                      abs(axa.get_xlim()[1] - axa.get_xlim()[0]))
    axt.set_xticks(pos)
    axt.set_xticklabels(txt, rotation=90)
    axt.tick_params(colors=MUTED, labelsize=6.2, length=2, pad=1)
    axt.set_xlabel(r"$\sigma_{\mathrm{assay}}$ of that scenario", color=MUTED,
                   fontsize=10.5 * PANEL_SCALE, labelpad=4)
    for s in ("left", "right", "bottom"):
        axt.spines[s].set_visible(False)
    axt.spines["top"].set_color(GRID)
    axt.grid(False)

    # --- (b) the certificate along the same rows ---------------------------------------
    # One line per knob: the sigma scenarios are joined, the pooled-draw scenarios are left
    # as isolated markers, since a single line through both would zig-zag between two
    # different experiments.
    xmax = max(float(x.max()), float(xp.max()) if len(xp) else 0.0)
    diag = np.array([0.0, xmax * 1.05])
    axb.plot(diag, diag, color=INK, lw=LW, zorder=4)
    axb.plot(x, _col(one, "C_Bmu_per_draw"), color="#1baf7a", lw=LW, ls="--", marker="^",
             ms=MS, zorder=3)
    if len(xp):
        axb.plot(xp, _col(pooled, "C_Bmu_per_draw"), color="#1baf7a", lw=0,
                 marker=DRAW_MARKER, ms=MS + 2.6, zorder=5)
    _style(axb, xlabel=r"$R_{\mathrm{mech}}$  (nats, measured)", ylabel="nats",
           scale=PANEL_SCALE)
    axb.set_xlim(axa.get_xlim())
    _letter(axb, 1)

    r_one = _col(one, "B_mu_centred") / _col(one, "B_crit")
    r_pool = (_col(pooled, "B_mu_centred") / _col(pooled, "B_crit")) if len(xp) \
        else np.array([])
    ratio = np.concatenate([r_one, r_pool]) if len(xp) else r_one
    ax2 = axb.twinx()
    ax2.plot(x, r_one, color=CERT, lw=LW, ls="-.", marker="d", ms=MS, zorder=3)
    if len(xp):
        ax2.plot(xp, r_pool, color=CERT, lw=0, marker=DRAW_MARKER, ms=MS + 2.6, zorder=5)
    ax2.axhline(1.0, color=CERT, lw=0.7, ls=":", alpha=0.9, zorder=1)
    ax2.set_ylim(0.0, max(1.15, float(ratio.max()) * 1.15))
    ax2.tick_params(colors=CERT, labelsize=9.5 * PANEL_SCALE, length=3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    ax2.grid(False)
    ax2.set_ylabel(r"$B_\mu / B_\mu^{\mathrm{crit}}$", color=CERT,
                   fontsize=10.5 * PANEL_SCALE)

    handles = [
        Line2D([], [], color=COLOUR["hybrid"], lw=LW, marker="o", ms=MS, label="hybrid TS"),
        *[Line2D([], [], color=COLOUR[p], lw=LW, marker="o", ms=MS, label=SHORT[p])
          for p in CHANNEL_POLICIES if p not in ("hybrid", "bsa")],
        Line2D([], [], color=COLOUR["bsa"], lw=LW, marker="o", ms=MS, label="BSA"),
        Line2D([], [], color=COLOUR["uninformed"], lw=LW, ls="--", marker="o", ms=MS,
               label=r"uninformed TS ($R_{\mathrm{mech}} = 0$)"),
        Line2D([], [], color=MUTED, lw=0, marker=DRAW_MARKER, ms=MS + 2.6,
               label=r"pooled draws, $d_Z = 2, 3$"),
        Line2D([], [], color=INK, lw=LW, label=r"$R_{\mathrm{mech}}$ (diagonal)"),
        Line2D([], [], color="#1baf7a", lw=LW, ls="--", marker="^", ms=MS,
               label=r"$C(B_\mu)/d_Z$ (Prop. 1)"),
        Line2D([], [], color=CERT, lw=LW, ls="-.", marker="d", ms=MS,
               label=r"$B_\mu/B_\mu^{\mathrm{crit}}$ (right axis)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=LEGEND_FS,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handlelength=1.8, **LEGEND)
    fig.tight_layout(rect=(0, 0.155, 1, 1))
    save(fig, figdir / "channel")


def save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        # PDF (vector) and PNG at 300 dpi.
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png / .pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv-dir", default="outputs/param_sweep")
    ap.add_argument("--fig-dir", default="outputs/param_sweep/figures")
    a = ap.parse_args()
    src, figdir = Path(a.csv_dir), Path(a.fig_dir)
    figdir.mkdir(parents=True, exist_ok=True)

    sets: dict[str, list[dict]] = {}
    for axis in OP_VALUE:
        path = src / f"{axis}.csv"
        if not path.exists():
            print(f"  (skipping {axis}: no {path})")
            continue
        sets[axis] = read_csv(path)
    if sets:
        figure_ofat_policies(sets, figdir)
        figure_ofat_certificate(sets, figdir)

    # The channel figure is a gamma = 1 figure and lives in its own CSV pair, so it is drawn
    # only where both files are present (`outputs/param_sweep_gamma1/`).
    chan_p, draw_p = src / "channel.csv", src / "channel_draws.csv"
    if chan_p.exists() and draw_p.exists():
        figure_channel(read_csv(chan_p), read_csv(draw_p), figdir)
    else:
        print("  (skipping channel: needs both channel.csv and channel_draws.csv)")


if __name__ == "__main__":
    main()
