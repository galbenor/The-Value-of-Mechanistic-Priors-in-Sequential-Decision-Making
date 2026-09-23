"""The house style for every figure this repository draws: palette, policy colours, the
calibration slices, and the shared axis treatment.

Not a package module: `scripts/` is on sys.path when a script in it is run directly, so
`from _plotstyle import ...` is the import.
"""
from __future__ import annotations

# --- palette -------------------------------------------------------------------------
#: The off-white page the figures sit on; `SURFACE_WHITE` is the pure-white variant.
SURFACE = "#fcfcfb"
SURFACE_WHITE = "#ffffff"
INK = "#0b0b0b"          # text and the one line that must be read first
MUTED = "#52514e"        # axis furniture, secondary annotation
GRID = "#dedcd5"         # gridlines and the two spines that stay
CERT = "#e34948"         # certificate quantities: B_crit and its working point

#: One colour per policy, used in every figure so a colour means the same thing throughout.
#: `mech_once` is the ODE solved once and played for the course; it gets its own hue because
#: at the biased operating point the two ODE policies separate.
COLOUR = {"bsa": "#2a78d6", "uninformed": "#eb6834", "mechanistic": "#1baf7a",
          "mech_once": "#c2258f", "hybrid": "#4a3aa7"}

#: The calibration slices the by-bias figures draw: (tail of the clearance distribution the
#: model was fitted on, fraction of the study cohort). 1.0 is the correctly calibrated model.
SLICES = [("low", 0.2), ("low", 0.4), ("low", 0.6), ("low", 0.8), ("low", 1.0),
          ("high", 0.8), ("high", 0.6), ("high", 0.4), ("high", 0.2)]

#: Base font sizes (tick labels, axis labels, panel title); `_style`'s `scale` multiplies
#: all three together.
FONT = (9.5, 10.5, 11.0)


def _style(ax, xlabel: str = "", ylabel: str = "", title: str = "", *,
           scale: float = 1.0, surface: str = SURFACE, pad: float = 7) -> None:
    """The shared axis treatment: light grid behind the data, two spines, muted furniture.

    `scale` multiplies every font size at once, for panels that are physically larger.
    """
    tick_fs, label_fs, title_fs = (f * scale for f in FONT)
    ax.set_facecolor(surface)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=tick_fs, length=3)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=label_fs)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=label_fs)
    if title:
        ax.set_title(title, color=INK, fontsize=title_fs, loc="left", pad=pad)
