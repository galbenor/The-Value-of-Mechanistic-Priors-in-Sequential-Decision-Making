"""The three-cohort protocol shared by every driver script.

Each driver sets up the same experiment: draw a study cohort, integrate its exposures, draw
the lead-in noise, calibrate the deployed model on it, run the lead-in, fit the hybrid's
prior table on the result, then draw a disjoint evaluation cohort and run the identical
lead-in on it. This module holds that block once.

The seed derivations are part of the result. Each stream gets its own generator, derived
from the caller's two seeds by a fixed offset:

    study_seed          the study cohort's patients
    study_seed + 4242   the study lead-in's biological and assay noise
    study_seed + 777    the dose-ranging study's binary outcomes (the Q rows)
    seed                the evaluation cohort's patients
    seed + 101          the evaluation horizon's noise (`draw_environment`)
    seed + 999          the evaluation lead-in's noise

Separate generators make the pairing checkable: shifting one stream moves exactly one thing.
Nothing here may be re-derived, re-ordered or merged into a shared generator without changing
every published number.

`build_study` / `build_eval` do the whole thing. The pieces they compose are public because
`run_bias_sweep.py` draws its cohorts once and re-calibrates and re-runs the lead-in for every
bias slice, and `make_appendix_figures.py` never draws an evaluation cohort at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twincert.generator import folfox as F
from twincert.experiment.mp_bandit import arm_reward_probs, draw_environment, run_study
from twincert.experiment.runin import RunInSpec, RunInState, draw_runin_environment, run_runin


@dataclass(frozen=True)
class StudyCohort:
    """The calibration study's patients and the truth about them, before any model."""

    coh: F.Cohort
    auc: np.ndarray             # (n_study, K) true 46-h exposure per arm
    conc: np.ndarray            # (n_study, K) true steady-state concentration per arm
    p: np.ndarray               # (n_study, K) P(in window), the reward probabilities
    pi_star: np.ndarray         # (n_study,)   each patient's own best arm
    renv: dict                  # the lead-in's noise streams


@dataclass(frozen=True)
class Deployment:
    """The deployed model and everything the hybrid knows before it treats anyone."""

    cl_pop_hat: float           # population clearance fitted on the calibration slice
    gamma: float                # the resulting scale bias; 1.0 when fitted on everyone
    state: RunInState           # the study cohort's lead-in
    study: dict | None          # run_study's {"Q", "table", "R_mech"}; None if not fitted


@dataclass(frozen=True)
class EvalCohort:
    """The disjoint evaluation patients, their truth, and their pre-drawn noise."""

    coh: F.Cohort
    auc: np.ndarray
    conc: np.ndarray
    p: np.ndarray
    pi_star: np.ndarray
    env: dict                   # the scored horizon's noise streams
    renv: dict                  # the lead-in's noise streams


def study_cohort(spec: F.FolfoxSpec, n_study: int, study_seed: int,
                 rs: RunInSpec | None = None) -> StudyCohort:
    """Draw the calibration study's patients, their exposures and their lead-in noise."""
    rs = RunInSpec(n_cycles=1) if rs is None else rs
    coh = F.sample_patients(spec, n_study, np.random.default_rng(study_seed))
    auc, conc = F.arm_exposures(spec, coh)
    renv = draw_runin_environment(np.random.default_rng(study_seed + 4242), n_study,
                                  rs.n_cycles, spec)
    p = arm_reward_probs(spec, auc)
    return StudyCohort(coh=coh, auc=auc, conc=conc, p=p, pi_star=p.argmax(axis=1),
                       renv=renv)


def deploy_model(spec: F.FolfoxSpec, sc: StudyCohort, study_seed: int, frac: float = 1.0,
                 side: str = "low", rs: RunInSpec | None = None,
                 with_table: bool = True) -> Deployment:
    """Calibrate the model on a slice of `sc`, run the study lead-in, fit the prior table.

    `frac` / `side` are the miscalibration axis: the deployed model's population clearance is
    fitted on the slowest- or fastest-clearing `frac` of the study cohort, so `gamma` is 1.0
    only at frac = 1.0. Set `with_table=False` when the caller only needs the measured
    constants and never runs the hybrid; it skips the dose-ranging study's outcome draws.

    The prior table is fitted on the lead-in of the model this call deploys, so at frac < 1.0
    it conditions on a biased model's recommendation. The sweeps expose both disciplines
    through `--fit-table-on-slice`: with the flag the study and the deployment share one
    population model; without it the table is fitted once, at gamma = 1, and carried into
    every slice. Only the table differs between the two:
    `run_study`'s Q matrix is drawn from its own generator and does not depend on `runin`.
    """
    rs = RunInSpec(n_cycles=1) if rs is None else rs
    cl_pop_hat, gamma = F.calibrate(spec, sc.coh, sc.conc[:, spec.bsa_arm],
                                    frac=frac, side=side)
    state = run_runin(spec, sc.coh, sc.auc, sc.conc, sc.p, sc.pi_star, sc.renv, gamma, rs)
    study = (run_study(spec, sc.auc, np.random.default_rng(study_seed + 777), runin=state)
             if with_table else None)
    return Deployment(cl_pop_hat=cl_pop_hat, gamma=gamma, state=state, study=study)


def build_study(spec: F.FolfoxSpec, n_study: int, study_seed: int, frac: float = 1.0,
                side: str = "low", rs: RunInSpec | None = None,
                with_table: bool = True) -> tuple[StudyCohort, Deployment]:
    """`study_cohort` followed by `deploy_model`: the whole pre-trial side."""
    rs = RunInSpec(n_cycles=1) if rs is None else rs
    sc = study_cohort(spec, n_study, study_seed, rs)
    return sc, deploy_model(spec, sc, study_seed, frac, side, rs, with_table)


def eval_cohort(spec: F.FolfoxSpec, n: int, seed: int, n_cycles: int,
                rs: RunInSpec | None = None) -> EvalCohort:
    """Draw the evaluation patients, their truth, and both of their noise environments."""
    rs = RunInSpec(n_cycles=1) if rs is None else rs
    coh = F.sample_patients(spec, n, np.random.default_rng(seed))
    auc, conc = F.arm_exposures(spec, coh)
    p = arm_reward_probs(spec, auc)
    env = draw_environment(np.random.default_rng(seed + 101), n, n_cycles, spec.n_arms,
                           spec)
    renv = draw_runin_environment(np.random.default_rng(seed + 999), n, rs.n_cycles, spec)
    return EvalCohort(coh=coh, auc=auc, conc=conc, p=p, pi_star=p.argmax(axis=1),
                      env=env, renv=renv)


def eval_leadin(spec: F.FolfoxSpec, ec: EvalCohort, gamma: float,
                rs: RunInSpec | None = None) -> RunInState:
    """The shared lead-in on the evaluation cohort: same arm, same noise, same cost for all.

    Split from `eval_cohort` because the bias sweep re-runs exactly this, and nothing else,
    for every calibration slice.
    """
    rs = RunInSpec(n_cycles=1) if rs is None else rs
    return run_runin(spec, ec.coh, ec.auc, ec.conc, ec.p, ec.pi_star, ec.renv, gamma, rs)


def build_eval(spec: F.FolfoxSpec, n: int, seed: int, n_cycles: int, gamma: float,
               rs: RunInSpec | None = None) -> tuple[EvalCohort, RunInState]:
    """`eval_cohort` followed by `eval_leadin`: the whole evaluation side."""
    rs = RunInSpec(n_cycles=1) if rs is None else rs
    ec = eval_cohort(spec, n, seed, n_cycles, rs)
    return ec, eval_leadin(spec, ec, gamma, rs)
