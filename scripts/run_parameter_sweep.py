"""One-factor-at-a-time sensitivity of the certificate and of every policy.

`run_example.py` reports one operating point. This script varies each design choice around
it alone, with everything else pinned, and each configuration re-runs the whole three-cohort
protocol through `twincert.experiment.protocol`.

Axes:

    n_arms  (K)     the number of dose arms on the fixed 1600-3600 mg/m^2 range.
    cycles  (N)     scored cycles; the course is N + lead-in.
    n_study         calibration-study size, at three study seeds, because a small study is
                    the one axis whose answer is a distribution rather than a number.
    leadin          how many un-scored BSA cycles with a blood draw precede the scored ones.
    channel         sigma_assay at one lead-in draw: scenario construction over the pre-dose
                    measurement channel, i.e. the trial's own design choice of how precisely
                    the sample is assayed. The x axis of the resulting figure is the measured
                    R_mech; sigma_assay is only the label of the point.
    channel_draws   the number of pooled lead-in draws d_Z at the quoted sigma_assay.

The two channel axes run at gamma = 1 and nowhere else: the bias sweep already varies the
model, and crossing a channel scenario with a biased model would confound the two. `main`
refuses to run them at `--calib-frac != 1.0`. The plant's population constants
(sigma_cycle, omega2_vmax) are never swept: they are the plant's declared magnitudes, not
design choices.

Held fixed in every row: the PK structure and its constants, the dose range and the standard
BSA dose, the therapeutic window, the covariate distribution, the noise constants, n_eval,
the seed scheme of `protocol.py`, and the bootstrap resample (`BOOT_SEED`). The BSA arm is
the grid point nearest the standard dose, recomputed per K and recorded in `dose_bsa_arm`.
Nothing is smoothed: each CSV row is one independent Monte-Carlo run.

The operating point is a biased deployment: the mechanistic model is calibrated on the
slowest-clearing 40% of the study cohort, `run_bias_sweep.py`'s (low, 0.40) slice, the regime
in which a population model must be trusted despite drift. `--calib-frac 1.0` puts it back at
the calibrated model. Every certificate quantity (B_mu, R_mech, C(B_mu), B_crit, C_nu) is the
deployed model's, measured on the study cohort. The hybrid's prior table is fitted on the same
slice under `--fit-table-on-slice` and is otherwise the gamma = 1 table fitted once by
`protocol.deploy_model`; the `table_fitted_on` column records which.

The certificate is C(B_mu) together with B_crit: the model passes the screen iff
B_mu < B_mu^crit (`certified`), which is necessary and not sufficient, and what it can then
carry is bounded by R_mech <= C(B_mu) (`chain_holds`). C_nu, the estimator-free form, is
recorded as the robustness line (`C_nu`, `c_nu_holds`). Every certificate quantity scales
with d_Z while sigma_h and sigma_nu keep their single-draw values, so on the lead-in axis the
comparable number is the per-draw ceiling `C_Bmu_per_draw` = C(B_mu)/d_Z.

Outputs: `outputs/param_sweep/<axis>.csv`, one per axis, drawn by
`scripts/plot_parameter_sweep.py`. `--out` is honoured, so the calibrated runs
(`make param-sweep-gamma1`, `make channel-sweep`) land in `outputs/param_sweep_gamma1/`.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import time
from pathlib import Path

import numpy as np

from twincert.experiment import mp_bandit as MB
from twincert.experiment.protocol import build_eval, build_study, deploy_model
from twincert.experiment.runin import RunInSpec
from twincert.experiment.stats import boot_indices, boot_ratio_from_indices, summ
from twincert.generator import folfox as F

#: The operating point: `run_example.py`'s defaults, and `FolfoxSpec`'s defaults for the
#: plant constants, with the deployed model calibrated on the slowest-clearing 40%.
DEFAULT = F.FolfoxSpec()
OP = {"n": 10000, "n_study": 2000, "cycles": 11, "seed": 11, "study_seed": 2024,
      "leadin": 1, "calib_frac": 0.4, "calib_side": "low"}

#: The bootstrap resample's seed, the same constant `run_example.py` uses, so the ratio
#: intervals of the operating-point row reproduce that script's `example_policies.csv`.
BOOT_SEED = 123

#: The three study seeds of the `n_study` axis. 2024 is the operating point; the other two
#: only have to differ from the evaluation seed 11 and from each other.
STUDY_SEEDS = (2024, 1729, 8675)

#: The grids. The operating-point entry of an axis is `FolfoxSpec`'s own default rather
#: than a copy of it, so that the row reproduces `run_example.py` bit for bit.
N_ARMS = (4, 6, DEFAULT.n_arms, 12, 16)
CYCLES = (3, 6, 11, 20, 50)
N_STUDY = (200, 500, 1000, 2000, 5000)
LEADIN = (1, 2, 3)

#: The channel scenarios. The operating-point entry is `FolfoxSpec`'s own sigma_assay. The
#: smallest value is the limit where the assay noise is negligible beside the patient's own
#: between-cycle variability, the most informative channel this plant admits at one draw.
CHANNEL_SIGMA = (0.02, 0.05, 0.10, 0.15, DEFAULT.sigma_assay, 0.30, 0.40, 0.50)
#: Pooled lead-in draws at the quoted sigma_assay; the same grid as `leadin`, run at a
#: different deployment (gamma = 1 here, biased there).
CHANNEL_DRAWS = (1, 2, 3)

#: The design axes, at the biased operating point: what `--axes` defaults to.
DESIGN_AXES = ("n_arms", "cycles", "n_study", "leadin")

#: The axes that are scenario construction over the measurement channel, meaningful only at
#: the calibrated model. `main` refuses to run them at any other `--calib-frac`.
CHANNEL_AXES = ("channel", "channel_draws")

AXES = DESIGN_AXES + CHANNEL_AXES


# --------------------------------------------------------------------------- #
# one plant configuration
# --------------------------------------------------------------------------- #
def run_point(spec: F.FolfoxSpec, *, n: int, n_study: int, cycles: int, seed: int,
              study_seed: int, leadin: int, calib_frac: float = 1.0,
              calib_side: str = "low", fit_table_on_slice: bool = False) -> dict:
    """The whole protocol at one configuration -> one CSV row.

    `run_example.py`'s body with the operating point turned into arguments and
    `run_bias_sweep.py`'s deployment in place of the calibrated one. Every step goes through
    `twincert.experiment.protocol`, so the cohorts, the calibration, the lead-in, the prior
    table and the evaluation are drawn exactly as they are everywhere else.

    Two deployments. `dep` is the gamma = 1 deployment: its dose-ranging study is the
    hybrid's prior table unless `fit_table_on_slice` is set. `dep_b` is what is actually
    deployed, calibrated on `calib_frac` of the cohort, and everything else reads it: the
    lead-in on both cohorts, every certificate quantity, and the model both model-based
    policies use. At `calib_frac = 1.0` the two deployments are the same object.
    `fit_table_on_slice` switches the hybrid to `dep_b`'s own table, fitted on the slice the
    deployed model was calibrated on; it moves the `hybrid` column and nothing else.

    The certificate is a study-cohort quantity throughout; only the regret columns read the
    evaluation cohort.
    """
    rs = RunInSpec(n_cycles=leadin)
    K, N = spec.n_arms, cycles

    sc, dep = build_study(spec, n_study, study_seed, rs=rs)
    dep_b = (dep if calib_frac == 1.0 else
             deploy_model(spec, sc, study_seed, frac=calib_frac, side=calib_side, rs=rs,
                          with_table=fit_table_on_slice))
    table = dep_b.study if fit_table_on_slice else dep.study
    ec, state = build_eval(spec, n, seed, N, dep_b.gamma, rs)

    # --- the certificate, measured on the study cohort under the deployed model ---------
    pre = MB.model_recommendation(
        spec, sc.coh.bsa, MB.covariate_prior(spec, dep_b.cl_pop_hat, sc.coh.bsa))
    i_pre = MB.information_quantities(sc.pi_star, pre, K)
    i_post = MB.information_quantities(sc.pi_star, dep_b.state.pi_hat, K)
    w = MB.occupancy_weights(sc.pi_star, K)
    mom = MB.measurement_moments(spec, sc.coh, sc.conc, sc.p)
    d_Z = rs.n_cycles                       # one draw per lead-in cycle
    L_h = mom["L_h_mu"]                     # the weighted norm, the primary constant
    mv = MB.model_arm_values(spec, sc.coh.bsa, dep_b.state.cl_hat)
    B_cen = MB.occupancy_weighted_bias_centred(sc.p, mv, w)
    C_B = MB.capacity_bound(B_cen, mom["sigma_h"], L_h, d_Z)
    C_nu = MB.capacity_nu(mom["sigma_h"], mom["sigma_nu"], d_Z, i_post["H_mu"])
    n_course = N + rs.n_cycles
    B_crit = MB.critical_bias(i_post["H_mu"], mom["sigma_h"], L_h, n_course, d_Z)
    # Proposition 1's bias branch bounds what the model can carry; Theorem 3's threshold is
    # the pre-trial pass/fail (a rule-out: passing is necessary, not sufficient). C_nu is the
    # estimator-free robustness line.
    chain_holds = bool(i_post["R_mech"] <= C_B + 1e-9)
    certified = bool(B_cen < B_crit)
    c_nu_holds = bool(i_post["R_mech"] <= C_nu + 1e-9)

    # --- the policies over the scored cycles ------------------------------------------
    res, raw = {}, {}
    for pol in MB.POLICIES:
        regret, _ = MB.run_policy(
            pol, spec, ec.coh, ec.auc, ec.conc, ec.p, ec.pi_star, ec.env, N,
            dep_b.cl_pop_hat, dep_b.gamma,
            study=table if pol == "hybrid" else None, runin=state)
        res[pol], raw[pol] = summ(regret), regret

    # One resample shared by the comparators, drawn from `run_example.py`'s generator, so
    # the ratios are paired with each other as well as with the hybrid and the
    # operating-point row reproduces `example_policies.csv`'s interval columns.
    bidx = boot_indices(n, np.random.default_rng(BOOT_SEED))
    ratios = {}
    for pol in ("bsa", "uninformed", "mechanistic", "mech_once"):
        lo, hi = boot_ratio_from_indices(raw[pol], raw["hybrid"], bidx)
        ratios[pol] = (res[pol].mean / res["hybrid"].mean, lo, hi)

    H_mech = max(i_post["H_mu"] - i_post["R_mech"], 1e-12)
    floor = float(np.sqrt(i_post["H_mu"] / H_mech))
    idx = np.arange(n)

    return {
        # --- what this configuration is -----------------------------------------------
        "sigma_assay": spec.sigma_assay, "sigma_cycle": spec.sigma_cycle,
        "omega2_vmax": spec.omega2_vmax, "n_arms": K,
        "dose_bsa_arm": float(spec.dose_levels[spec.bsa_arm]), "bsa_arm": spec.bsa_arm,
        "n_scored": N, "n_leadin": rs.n_cycles, "n_course": n_course,
        "n_study": n_study, "study_seed": study_seed, "n_eval": n, "seed": seed,
        "calib_frac": calib_frac, "calib_side": calib_side, "gamma": dep_b.gamma,
        # --- the certificate (study cohort) --------------------------------------------
        "H_mu": i_post["H_mu"], "H_mech": i_post["H_mu"] - i_post["R_mech"],
        "R_mech_pre": i_pre["R_mech"], "R_mech": i_post["R_mech"],
        "R_mech_null": i_post["R_mech_null"],
        "accuracy": float(np.mean(dep_b.state.pi_hat == sc.pi_star)),
        "sigma_h": mom["sigma_h"], "sigma_nu": mom["sigma_nu"],
        "L_h_mu": L_h, "L_h_sup": mom["L_h"], "d_Z": d_Z,
        "B_mu_centred": B_cen, "C_B_mu": C_B,
        # the ceiling at the d_Z = 1 convention: the comparable number across rows of the
        # lead-in axis (identical to C_B_mu everywhere else, where d_Z = 1)
        "C_Bmu_per_draw": C_B / d_Z,
        "B_crit": B_crit, "margin": B_crit / max(B_cen, 1e-12),
        "chain_holds": chain_holds, "certified": certified,
        # the robustness line: estimator-free, and not the certificate
        "C_nu": C_nu, "c_nu_holds": c_nu_holds,
        # --- the plant's attainment (evaluation cohort) --------------------------------
        "p_bsa": float(ec.p[:, spec.bsa_arm].mean()),
        "p_opt": float(ec.p[idx, ec.pi_star].mean()),
        "leadin_regret": float(state.regret.mean()),
        # --- the policies (evaluation cohort) ------------------------------------------
        "bsa": res["bsa"].mean, "bsa_ci95": res["bsa"].ci,
        "uninformed": res["uninformed"].mean, "uninformed_ci95": res["uninformed"].ci,
        "mechanistic": res["mechanistic"].mean,
        "mechanistic_ci95": res["mechanistic"].ci,
        "hybrid": res["hybrid"].mean, "hybrid_ci95": res["hybrid"].ci,
        # --- the ratios, paired bootstrap ----------------------------------------------
        "ratio_bsa_hyb": ratios["bsa"][0], "ratio_bsa_hyb_lo": ratios["bsa"][1],
        "ratio_bsa_hyb_hi": ratios["bsa"][2],
        "ratio_uninf_hyb": ratios["uninformed"][0],
        "ratio_uninf_hyb_lo": ratios["uninformed"][1],
        "ratio_uninf_hyb_hi": ratios["uninformed"][2],
        "ratio_mech_hyb": ratios["mechanistic"][0],
        "ratio_mech_hyb_lo": ratios["mechanistic"][1],
        "ratio_mech_hyb_hi": ratios["mechanistic"][2],
        # --- Theorem 1's predicted floor ------------------------------------------------
        "floor_thm1": floor,
        # --- the ODE solved once and played for the course ------------------------------
        # `mech_once` carries exactly the hybrid's information (same lead-in draw, same
        # single solve, same pi_hat) and no learning, so `ratio_mech_once_hyb` is the price
        # of Theorem 2's reward feedback alone.
        "mech_once": res["mech_once"].mean, "mech_once_ci95": res["mech_once"].ci,
        "ratio_mech_once_hyb": ratios["mech_once"][0],
        "ratio_mech_once_hyb_lo": ratios["mech_once"][1],
        "ratio_mech_once_hyb_hi": ratios["mech_once"][2],
        # Which population fitted the hybrid's prior table: the two runs differ in the
        # hybrid column and nothing else.
        "table_fitted_on": "slice" if fit_table_on_slice else "full",
    }


# --------------------------------------------------------------------------- #
# the axes
# --------------------------------------------------------------------------- #
def axis_settings(axis: str, base: dict) -> list[tuple[dict, dict]]:
    """(spec overrides, protocol overrides) for every point of one axis.

    Everything the caller does not list here stays at `base`, which is the operating point:
    that is what makes this one-factor-at-a-time rather than a joint sweep.
    """
    if axis == "n_arms":
        return [({"n_arms": v}, {}) for v in N_ARMS]
    if axis == "cycles":
        return [({}, {"cycles": v}) for v in CYCLES]
    if axis == "n_study":
        return [({}, {"n_study": v, "study_seed": s})
                for v in N_STUDY for s in STUDY_SEEDS]
    if axis == "leadin":
        return [({}, {"leadin": v}) for v in LEADIN]
    # The two channel scenarios. Each names its other knob explicitly rather than leaning on
    # `base`: the sigma scenarios are all at one draw, the draw scenarios all at the quoted
    # sigma.
    if axis == "channel":
        return [({"sigma_assay": v}, {"leadin": 1}) for v in CHANNEL_SIGMA]
    if axis == "channel_draws":
        return [({"sigma_assay": DEFAULT.sigma_assay}, {"leadin": v})
                for v in CHANNEL_DRAWS]
    raise ValueError(f"unknown axis {axis!r}")


#: Which column each axis is plotted against, and how a row of it is labelled.
AXIS_X = {"n_arms": "n_arms", "cycles": "n_scored", "n_study": "n_study",
          "leadin": "n_leadin", "channel": "sigma_assay", "channel_draws": "n_leadin"}


def run_axis(axis: str, base: dict, out: Path) -> list[dict]:
    """Run one axis, write `<axis>.csv`, return the rows."""
    settings = axis_settings(axis, base)
    rows, t0 = [], time.time()
    for spec_kw, proto_kw in settings:
        cfg = dict(base)
        cfg.update(proto_kw)
        spec = dataclasses.replace(DEFAULT, **spec_kw)
        t = time.time()
        row = run_point(spec, **cfg)
        row = {"axis": axis, "axis_value": row[AXIS_X[axis]], **row}
        rows.append(row)
        print(f"  [{axis}] {AXIS_X[axis]}={row['axis_value']:<8g} "
              f"gamma={row['gamma']:.3f} R_mech={row['R_mech']:.3f} "
              f"C(B_mu)={row['C_B_mu']:.3f} "
              f"B_mu={row['B_mu_centred']:.3f} B_crit={row['B_crit']:.3f} "
              f"hyb={row['hybrid']:.3f} once={row['mech_once']:.3f} "
              f"uninf={row['uninformed']:.3f} bsa={row['bsa']:.3f} "
              f"({time.time() - t:.1f}s)", flush=True)
    write_csv(out / f"{axis}.csv", rows)
    print(f"  -> {out / (axis + '.csv')}  ({len(rows)} rows, {time.time() - t0:.1f}s)",
          flush=True)
    return rows


# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: list[dict]) -> None:
    """Six significant digits for every float, so a 2-3 dp display downstream is a single
    rounding of the stored value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(rows[0].keys())
        for r in rows:
            wr.writerow(f"{v:.6g}" if isinstance(v, float) else v for v in r.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--axes", default="",
                    help=f"comma-separated subset of {AXES}; default is the four design axes "
                         f"{DESIGN_AXES}. The channel axes {CHANNEL_AXES} are a different "
                         f"experiment -- R_mech scenarios built by moving the measurement "
                         f"channel -- and must be named, and named with --calib-frac 1.0")
    ap.add_argument("--calib-frac", type=float, default=OP["calib_frac"],
                    help="fraction of the study cohort the deployed model is calibrated on "
                         "(default 0.4: run_bias_sweep.py's slowest-40%% slice). 1.0 is the "
                         "calibrated model and reproduces run_example.py")
    ap.add_argument("--calib-side", choices=("low", "high"), default=OP["calib_side"],
                    help="which tail of the clearance distribution that slice is (default low,"
                         " the slowest-clearing patients)")
    ap.add_argument("--fit-table-on-slice", action="store_true",
                    help="fit the hybrid's prior table on the calibration slice of the "
                         "deployed model. Default: the table fitted on the full study cohort "
                         "(gamma = 1). Only the hybrid column changes; no effect at "
                         "--calib-frac 1.0")
    ap.add_argument("--out", default="outputs/param_sweep")
    a = ap.parse_args()

    out = Path(a.out)
    base = dict(OP)
    base.update(calib_frac=a.calib_frac, calib_side=a.calib_side,
                fit_table_on_slice=a.fit_table_on_slice)
    want = [s.strip() for s in a.axes.split(",") if s.strip()]

    axes = [x for x in AXES if x in want] if want else list(DESIGN_AXES)
    unknown = [x for x in want if x not in AXES]
    if unknown:
        raise SystemExit(f"unknown axes {unknown}; known: {AXES}")
    if not axes:
        raise SystemExit("no axes selected")
    # The channel axes are gamma = 1 axes: deploying a biased model at the same time would
    # confound the two. Refuse rather than silently produce a confounded CSV.
    bad_channel = [x for x in axes if x in CHANNEL_AXES] if base["calib_frac"] != 1.0 else []
    if bad_channel:
        raise SystemExit(
            f"the channel axes {bad_channel} are scenario construction over the MEASUREMENT "
            f"CHANNEL and are only meaningful at the calibrated model: re-run with "
            f"--calib-frac 1.0 (got {base['calib_frac']}). Model miscalibration is "
            f"run_bias_sweep.py's axis, and crossing the two confounds them.")
    out.mkdir(parents=True, exist_ok=True)

    dep_txt = ("the CALIBRATED model (gamma = 1)" if base["calib_frac"] == 1.0 else
               f"the model calibrated on the {base['calib_side']} "
               f"{base['calib_frac']:.0%} slice")
    print(f"parameter sweep -> {out}   deploying {dep_txt}\n  "
          f"(evaluation n={base['n']}, study n={base['n_study']}, seeds "
          f"{base['seed']}/{base['study_seed']})")

    t0 = time.time()
    for axis in axes:
        print(f"\n{axis}", flush=True)
        run_axis(axis, base, out)
    print(f"\ntotal {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
