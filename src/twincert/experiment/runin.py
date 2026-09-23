"""Cycle 1: the body-surface-area dose plus one concentration test, before the scored cycles.

Before any measurement the mechanistic model has nothing patient-specific to say: the dose is
per square metre while apparent clearance scales as BSA^b with b close to one, so the
covariate-only recommendation is the same dose for every patient and R_mech = I(pi*; pi_hat)
is exactly zero. One blood test during a standard first cycle changes that: the model is
inverted on the measured concentration, and only then does it start choosing doses.

This module is not a policy. It runs a fixed protocol, identical for every policy in arm,
noise draws and regret, and hands back the model's resulting belief as a `RunInState`.
`mp_bandit` does not import this module; it takes a `runin=` object structurally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twincert.generator import folfox as F
from twincert.experiment.mp_bandit import log_cl_estimate, model_recommendation


@dataclass(frozen=True)
class RunInSpec:
    """The protocol for the un-scored cycles that precede the scored ones.

    `n_cycles = 1` is the clinical case: one cycle of standard dosing with one
    therapeutic-drug-monitoring draw. More cycles average down the measurement noise in the
    inverted clearance, at the cost of more cycles at the standard dose.
    """

    n_cycles: int = 1               # [LIT] one lead-in cycle is the standard TDM protocol


@dataclass(frozen=True)
class RunInState:
    """The handoff from cycle 1 into the scored cycles.

    Everything a downstream policy may use is in here, which is what makes "the ODE is used as
    a prior only" a checkable property. `log_cl_sum` and `n_obs` are the sufficient statistics
    of `log_cl_estimate`, so a policy that keeps refitting gets a correct warm start.
    """

    log_cl_sum: np.ndarray          # (n,)  sufficient statistic for the log-CL estimate
    n_obs: float                    #       number of tests behind it
    cl_hat: np.ndarray              # (n,)  the model's clearance estimate entering cycle 2
    pi_hat: np.ndarray              # (n,)  the dose it therefore recommends
    regret: np.ndarray              # (n,)  regret paid in the lead-in; equal for every policy
    reward: np.ndarray              # (n, n_cycles) in-window indicator, per lead-in cycle

    def __len__(self) -> int:
        return int(self.cl_hat.size)


def draw_runin_environment(rng: np.random.Generator, n: int, n_runin: int,
                           spec: F.FolfoxSpec) -> dict:
    """Pre-draw the lead-in's noise on a stream separate from the scored horizon's.

    Widening `draw_environment`'s arrays instead would shift which numbers land in the scored
    cycles and break every paired comparison. `cycle_noise` is biological and moves the
    exposure the patient receives; `assay_noise` is measurement only and moves the reading.
    """
    return {"cycle_noise": rng.normal(0.0, spec.sigma_cycle, (n, n_runin)),
            "assay_noise": rng.normal(0.0, spec.sigma_assay, (n, n_runin))}


def run_runin(spec: F.FolfoxSpec, coh: F.Cohort, auc_true: np.ndarray, conc_true: np.ndarray,
              p_true: np.ndarray, pi_star: np.ndarray, renv: dict,
              gamma: float = 1.0, rs: RunInSpec | None = None) -> RunInState:
    """Run the lead-in and return what the model believes at the end of it.

    The same function runs on the study and on the evaluation cohort: `hybrid_prior_table`
    estimates P(pi* | pi_hat), so the recommendation it conditions on must be built exactly as
    the deployed policy builds it. The lead-in never consults the model: the dose is fixed by
    protocol at the body-surface-area arm, and `gamma` enters only through `fit_clearance`,
    i.e. only through what the model believes afterwards.
    """
    rs = RunInSpec() if rs is None else rs
    arm = spec.bsa_arm
    n = len(coh.bsa)
    idx = np.arange(n)

    log_cl_sum = np.zeros(n)
    regret = np.zeros(n)
    reward = np.zeros((n, rs.n_cycles))
    best = p_true[idx, pi_star]

    for t in range(rs.n_cycles):
        regret += best - p_true[:, arm]
        # One biological draw drives both the exposure received and the reading taken from it;
        # the assay error is layered on the reading alone.
        cycle = np.exp(renv["cycle_noise"][:, t])
        auc_real = auc_true[:, arm] * cycle
        reward[:, t] = ((auc_real >= spec.auc_lo) & (auc_real <= spec.auc_hi)).astype(float)
        c_obs = conc_true[:, arm] * cycle * np.exp(renv["assay_noise"][:, t])
        log_cl_sum += np.log(np.maximum(
            F.fit_clearance(spec, spec.dose_levels[arm], coh.bsa, c_obs, gamma), 1e-9))

    n_obs = float(rs.n_cycles)
    cl_hat = np.exp(log_cl_estimate(log_cl_sum, n_obs))
    return RunInState(log_cl_sum=log_cl_sum, n_obs=n_obs, cl_hat=cl_hat,
                      pi_hat=model_recommendation(spec, coh.bsa, cl_hat),
                      regret=regret, reward=reward)
