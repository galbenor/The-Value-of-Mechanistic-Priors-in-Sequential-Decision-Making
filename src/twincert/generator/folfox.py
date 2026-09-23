"""The 5-FU environment: Tan's saturable two-compartment ODE, and the model that approximates it.

The truth is an ODE; the deployed model is a different, simpler ODE. Every quantity the theory
defines is therefore measured on that pair rather than stipulated.

The truth, Tan's two-compartment model with purely saturable elimination:

    dA1/dt = R(t) - Vmax*C1/(Km + C1) - Q*(C1 - C2)
    dA2/dt =                            Q*(C1 - C2)
    C1 = A1/V1,  C2 = A2/V2,  R(t) = dose*BSA/T over the 46-hour infusion

Elimination is Michaelis-Menten because DPD saturates, which is why exposure is not
proportional to dose. The model, one compartment with linear clearance, keeps the plasma
compartment and drops both the peripheral one and the saturation:

    AUC_hat(d) = d*BSA/CL

Under a 46-hour constant infusion the compartment count is nearly free (mass balance gives
AUC = (Dose/CL)*(1 - Vss/(CL*T)), so topology enters only through the total volume), and the
linearisation of the elimination is the actual misspecification.

Three sources of variation, which are not interchangeable:

    people          BSA and Vmax, drawn per patient. Patients differ, so the optimal arm
                    differs, and the spread of that arm across the cohort is the entropy a
                    learner must resolve. The population magnitudes are illustrative, rounded
                    choices around the published covariate model of Tan et al. (see
                    `FolfoxSpec`): the plant is an example for the theory, not a calibrated
                    digital twin.
    cycle noise     sigma_cycle, biological, within patient between cycles. The same patient
                    on the same dose does not get the same exposure twice, so the reward is
                    random: arm k pays off with probability P(AUC_k * e^eps in the window).
    assay noise     sigma_assay, measurement only. It moves the observed concentration but
                    not the exposure the patient received, so it corrupts the model's
                    estimate without touching the reward.

Calibration is the only source of model bias here. `calibrate` fits the model's population
clearance on a slice of the cohort: fitted on everyone it is right, fitted on the fastest- or
slowest-clearing X% it is systematically wrong by an amount that grows as X shrinks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Provenance tags: [LIT] published, [DER] derived from something published, [SYN] a
# modelling choice.


@dataclass(frozen=True)
class FolfoxSpec:
    """Population constants. Per-patient random effects live in :class:`Cohort`.

    Every field is defined at its declaration, with units and a provenance tag.
    """

    # --- protocol (hours) ---
    infusion_h: float = 46.0        # [LIT] infusion duration (h): FOLFOX6 continuous infusion
    obs_h: float = 40.0             # [SYN] time of the one TDM blood draw (h after infusion
    #                                       start); late enough to sit on the plateau
    dt_h: float = 0.04              # [DER] integrator step (h); halving it does not move the
    #                                       reported exposures

    # --- Tan's two-compartment saturable disposition ---
    V1: float = 19.7                # [LIT] central (plasma) compartment volume (L)
    V2: float = 15.0                # [LIT] peripheral compartment volume (L)
    Q: float = 52.4                 # [LIT] intercompartmental clearance (L/h)
    Km: float = 8.04                # [LIT] Michaelis constant (mg/L): the plasma concentration
    #                                       at which elimination runs at half of Vmax
    # Illustrative, rounded typical value, well inside the published uncertainty (Tan et al.
    # report TVVmax = 1900 mg/h with bootstrap CI 1398-2842). Not an estimate of any dataset:
    # the plant is an example for the theory. The value is chosen so that the covariate-only
    # recommendation is the same arm for every patient, which is the premise `runin.py`
    # states (R_mech is exactly zero before the lead-in test).
    vmax_pop: float = 1670.0        # [SYN] population geometric-mean maximal elimination rate
    #                                       (mg/h) at the reference body surface area `bsa_ref`
    # Illustrative total between-patient variance of log Vmax, set somewhat above the
    # published parameter-only value (Tan Table 2) to stand in for the covariate residuals a
    # total spread also carries. An assumed magnitude, not a measurement.
    omega2_vmax: float = 0.14       # [SYN] between-patient variance of log Vmax
    #                                       (dimensionless, log scale)
    bsa_exponent: float = 1.14      # [LIT] allometric exponent a in Vmax ~ BSA^a
    #                                       (dimensionless); Tan Table S4
    bsa_ref: float = 1.94           # [LIT] reference body surface area (m^2) at which
    #                                       `vmax_pop` is stated

    # --- the cohort ---
    bsa_mean: float = 1.90          # [LIT] cohort mean body surface area (m^2); Tan Table 1
    bsa_sd: float = 0.25            # [LIT] cohort BSA standard deviation (m^2); Tan Table 1
    bsa_lo: float = 1.40            # [LIT] lower truncation of the BSA draw (m^2)
    bsa_hi: float = 2.60            # [LIT] upper truncation of the BSA draw (m^2)

    # --- noise: biological, then measurement ---
    # The biological magnitude is quoted (the ~20% intra-patient cycle-to-cycle variability of
    # Li et al. 2023; Tan et al. fit a 12% inter-occasion variability on Vmax, Morawska et al.
    # 2018 measure a 17% intra-individual CV of AUC/dose). The assay magnitude, and with it
    # the split, is an assumed rounded order of magnitude, of plausible size next to the
    # proportional residual errors the 5-FU literature reports (~25-34%). The split matters:
    # sigma_cycle moves the exposure the patient received and so caps how well
    # any policy can do, while sigma_assay corrupts only the number the laboratory reports.
    sigma_cycle: float = 0.20       # [LIT] sd of log exposure, within-patient between-cycle
    #                                       (dimensionless, log scale); moves the reward
    sigma_assay: float = 0.22       # [SYN] sd of log concentration, measurement error only;
    #                                       moves only the observation

    # --- the therapeutic window ---
    auc_lo: float = 20.0            # [LIT] lower edge of the target 46-h AUC window (mg.h/L)
    auc_hi: float = 30.0            # [LIT] upper edge of the target 46-h AUC window (mg.h/L)

    # --- the arms ---
    dose_min: float = 1600.0        # [LIT] lowest arm (mg/m^2); Kaldate adjustment range
    dose_max: float = 3600.0        # [LIT] highest arm (mg/m^2); Kaldate adjustment range
    n_arms: int = 8                 # [SYN] number of arms K
    dose_bsa: float = 2400.0        # [LIT] the standard body-surface-area dose (mg/m^2)

    @property
    def dose_levels(self) -> np.ndarray:
        """The K arms, evenly spaced on [dose_min, dose_max]."""
        return np.linspace(self.dose_min, self.dose_max, self.n_arms)

    @property
    def bsa_arm(self) -> int:
        """Index of the standard-of-care arm: the grid point nearest the BSA dose."""
        return int(np.argmin(np.abs(self.dose_levels - self.dose_bsa)))


@dataclass(frozen=True)
class Cohort:
    """The patients. Two random effects, and nothing else differs between them."""

    bsa: np.ndarray                 # (n,) body surface area per patient (m^2)
    vmax: np.ndarray                # (n,) maximal elimination rate per patient (mg/h)

    def __len__(self) -> int:
        return int(self.bsa.size)


def sample_patients(spec: FolfoxSpec, n: int, rng: np.random.Generator) -> Cohort:
    """Draw `n` patients: a truncated-normal BSA and a log-normal Vmax around it.

    Parameters:
        spec  population constants (means, spreads, truncation, allometric exponent)
        n     number of patients to draw
        rng   the random generator; the only source of randomness here
    """
    bsa = np.clip(rng.normal(spec.bsa_mean, spec.bsa_sd, n), spec.bsa_lo, spec.bsa_hi)
    eta = rng.normal(0.0, np.sqrt(spec.omega2_vmax), n)
    vmax = spec.vmax_pop * (bsa / spec.bsa_ref) ** spec.bsa_exponent * np.exp(eta)
    return Cohort(bsa, vmax)


# ---------------------------------------------------------------------------
# the truth
# ---------------------------------------------------------------------------
def integrate(spec: FolfoxSpec, dose, bsa, vmax, *, n_compartments: int = 2,
              dt: float | None = None, full: bool = False):
    """Integrate Tan's model over the infusion. Returns (AUC, concentration at `obs_h`).

    Linearly implicit backward Euler, with the Michaelis-Menten coefficient frozen at the
    previous step (one Picard iteration), so each step is a 2x2 linear solve with an analytic
    inverse. The plant is stiff (V1/CL is minutes against a 46-hour horizon), so an explicit
    scheme would need a far smaller step than the answer needs.

    Parameters:
        spec            protocol and disposition constants (infusion length, volumes, Q, Km)
        dose            dose intensity (mg/m^2); broadcasts against `bsa` and `vmax`, so the
                        whole (patients x arms) grid is one call
        bsa             body surface area (m^2); the delivered dose is dose * bsa (mg)
        vmax            the patient's maximal elimination rate (mg/h)
        n_compartments  1 or 2: switches the peripheral compartment off, to measure the
                        topology effect
        dt              integrator step (h); None -> `spec.dt_h`
        full            if True, additionally return the mass-balance terms (infused,
                        eliminated, in_body; all mg)

    Returns (AUC in mg.h/L, concentration at `spec.obs_h` in mg/L), each shaped like the
    broadcast inputs.
    """
    dt = spec.dt_h if dt is None else float(dt)
    dose, bsa, vmax = np.broadcast_arrays(np.asarray(dose, float), np.asarray(bsa, float),
                                          np.asarray(vmax, float))
    rate = dose * bsa / spec.infusion_h                  # mg/h, constant over the infusion
    q = spec.Q if n_compartments >= 2 else 0.0

    a1 = np.zeros(rate.shape)
    a2 = np.zeros(rate.shape)
    auc = np.zeros(rate.shape)
    elim = np.zeros(rate.shape)
    c_obs = np.zeros(rate.shape)

    n_steps = int(round(spec.infusion_h / dt))
    obs_step = int(round(spec.obs_h / dt))
    for step in range(n_steps):
        # Elimination clearance in L/h, frozen at the previous step: (mg/h)/(mg/L).
        cl_eff = vmax / (spec.Km + a1 / spec.V1)
        # Backward Euler on dA/dt = r + J A  =>  (I - dt J) A_new = A_old + dt r, solved by hand:
        #     [alpha beta] [a1]   [rhs1]
        #     [delta eps ] [a2] = [rhs2]
        alpha = 1.0 + dt * (q + cl_eff) / spec.V1
        beta = -dt * q / spec.V2
        delta = -dt * q / spec.V1
        eps = 1.0 + dt * q / spec.V2
        rhs1, rhs2 = a1 + dt * rate, a2
        a1 = (rhs1 - beta * rhs2 / eps) / (alpha - beta * delta / eps)
        a2 = (rhs2 - delta * a1) / eps
        c1 = a1 / spec.V1
        auc += dt * c1                                   # quadrature consistent with the scheme
        elim += dt * cl_eff * c1
        if step + 1 == obs_step:
            c_obs = c1

    if not full:
        return auc, c_obs
    return auc, c_obs, {"infused": rate * spec.infusion_h, "eliminated": elim,
                        "in_body": a1 + a2}


def arm_exposures(spec: FolfoxSpec, coh: Cohort, **kw):
    """True (AUC, observed concentration) for every patient on every arm -> two (n, K) arrays.

    Parameters:
        spec  protocol and disposition constants; supplies the K-arm dose grid
        coh   the patients; one row of the output per patient
        kw    forwarded to `integrate` (n_compartments, dt, full)
    """
    return integrate(spec, spec.dose_levels[None, :], coh.bsa[:, None], coh.vmax[:, None], **kw)


# ---------------------------------------------------------------------------
# the model: one compartment, linear clearance
# ---------------------------------------------------------------------------
def auc_model(spec: FolfoxSpec, dose, bsa, cl_hat):
    """Exposure the deployed model predicts: AUC = dose * BSA / CL.

    The clinical dose-over-clearance rule: no saturation, no peripheral compartment, and no
    finite-infusion correction.

    Parameters:
        spec    unused; kept so every model function takes the plant spec
        dose    dose intensity (mg/m^2); broadcasts against `bsa` and `cl_hat`
        bsa     body surface area (m^2)
        cl_hat  the model's clearance estimate for the patient (L/h)

    Returns the predicted 46-h AUC (mg.h/L), shaped like the broadcast inputs.
    """
    dose, bsa, cl_hat = np.broadcast_arrays(np.asarray(dose, float), np.asarray(bsa, float),
                                            np.asarray(cl_hat, float))
    return dose * bsa / cl_hat


def fit_clearance(spec: FolfoxSpec, dose, bsa, c_obs, gamma: float = 1.0):
    """Invert the model on one blood draw: at steady state Css = R/CL, so CL = gamma * R/Css.

    This is the model's only point of contact with data, and where the one-compartment
    assumption bites: the draw is read as a plateau, and the clearance it implies is treated
    as constant across every arm even though the truth's saturates.

    `gamma` applies here rather than only to the prior because a population model reports
    clearance on the scale its calibration study established, so a calibration fitted on the
    wrong cohort puts every estimate on the wrong scale, the first and the last alike. It is a
    level error: it shifts predicted log exposure by a constant at every arm, including the
    one just measured, so no amount of data at a single dose reveals it. A scalar `gamma` is a
    population-level scale error, which preserves the model's ranking of patients; a
    per-patient vector destroys the ranking itself.

    Parameters:
        spec   supplies the infusion length (h)
        dose   the dose that produced the draw (mg/m^2); broadcasts against `bsa` and `c_obs`
        bsa    body surface area (m^2)
        c_obs  the measured plasma concentration (mg/L), read as a plateau value
        gamma  the calibration scale (dimensionless): calibrated / true population clearance;
               1.0 = correctly calibrated. Scalar or per-patient vector

    Returns the apparent clearance CL = gamma * R / c_obs (L/h), shaped like the inputs.
    """
    rate = np.asarray(dose, float) * np.asarray(bsa, float) / spec.infusion_h
    return np.asarray(gamma, float) * rate / np.maximum(np.asarray(c_obs, float), 1e-9)


def covariate_exponent(spec: FolfoxSpec) -> float:
    """How apparent clearance scales with body size in this plant.

    At the plateau the apparent clearance the model infers is CL = R/Css = (Vmax - R)/Km, and
    both terms scale with body size: Vmax as BSA^a, and the infusion rate R linearly, since the
    dose is itself per square metre. Differentiating at the typical patient,

        dlogCL/dlogBSA = (a*Vmax - R) / (Vmax - R)

    which is not the plant's Vmax exponent `a`. Computed rather than pinned so that it follows
    `bsa_exponent`, `vmax_pop` and the dose grid automatically.

    Parameters:
        spec  supplies the Vmax allometry (a, vmax_pop, bsa_ref) and the BSA-arm dose that
              sets the typical infusion rate

    Returns the dimensionless exponent b = dlogCL/dlogBSA at the typical patient.
    """
    rate = spec.dose_levels[spec.bsa_arm] * spec.bsa_ref / spec.infusion_h
    return float((spec.bsa_exponent * spec.vmax_pop - rate) / (spec.vmax_pop - rate))


def calibrate(spec: FolfoxSpec, coh: Cohort, c_obs_bsa: np.ndarray,
              frac: float = 1.0, side: str = "low") -> tuple[float, float]:
    """Fit the model's clearance scale on a slice of the cohort -> (cl_pop_hat, gamma).

    This is the only source of bias in the experiment. A population pharmacokinetic model's
    typical value is estimated from whichever patients the calibration study enrolled. Fitted
    on everyone (`frac = 1`) it is right and `gamma = 1`, leaving only the structural error
    of the dropped compartment and the linearised saturation. Fitted on the fastest-clearing
    (`side = "high"`) or slowest-clearing (`side = "low"`) `frac` of the cohort it is wrong in
    a known direction, by an amount that grows as `frac` shrinks.

    `gamma` is the ratio to the correctly calibrated value, and it multiplies the model's
    whole clearance scale: the prior mean and every subsequent inversion alike. Both sides are
    swept because a one-sided sweep confounds robustness with a lucky direction.

    Calibration reads noise-free plateau concentrations, since a real calibration study
    averages over its cohort: what is varied is who was enrolled, not how well they were
    measured.

    Parameters:
        spec       protocol constants; supplies the BSA-arm dose the study dosed at
        coh        the calibration-study patients
        c_obs_bsa  (n,) noise-free plateau concentration of each study patient at the BSA
                   arm (mg/L): `arm_exposures(...)[1][:, spec.bsa_arm]`
        frac       the enrolled fraction of the study cohort, in (0, 1]; 1.0 = everyone
        side       which tail of the apparent-clearance distribution was enrolled:
                   "low" = slowest-clearing, "high" = fastest-clearing

    Returns (cl_pop_hat, gamma): the fitted typical clearance at `bsa_ref` (L/h), and the
    dimensionless scale bias relative to the full-cohort fit (1.0 exactly at frac = 1).
    """
    cl_apparent = fit_clearance(spec, spec.dose_levels[spec.bsa_arm], coh.bsa, c_obs_bsa)
    # The typical value is reported at `bsa_ref`, so the covariate is regressed out first: a
    # bare geometric mean would be the typical value at the cohort's mean body size, which
    # `covariate_prior` would then rescale as though it were the value at `bsa_ref`.
    adj = np.log(cl_apparent) - covariate_exponent(spec) * np.log(coh.bsa / spec.bsa_ref)
    order = np.argsort(cl_apparent)
    m = max(int(round(float(frac) * order.size)), 2)
    take = order[:m] if side == "low" else order[-m:]
    cl_pop_hat = float(np.exp(np.mean(adj[take])))
    return cl_pop_hat, cl_pop_hat / float(np.exp(np.mean(adj)))
