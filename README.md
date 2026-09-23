# An ODE used as a prior: the 5-FU example

Code for the in-silico experiments of *The Value of Mechanistic Priors in Sequential
Decision Making*. The truth is a pharmacokinetic ODE and the deployed model is a different,
simpler ODE, so every quantity the paper defines (the mechanistic information `R_mech`, the
occupancy-weighted bias `B_µ`, the capacity bound `C(B_µ)`, the critical bias `B_µ^crit`) is
measured on that pair rather than assumed, and every table and figure of the paper's
calibration appendix is regenerated from this repository.

Everything here is in silico. No clinical claim is made.

## What is simulated

**The plant (the truth).** Tan et al.'s two-compartment 5-FU model with saturable
(Michaelis–Menten) elimination, integrated once per (patient, arm) pair. Patients differ in
body surface area and in `Vmax`. The observable is the 46-hour exposure AUC; the reward is
`r = 1{AUC ∈ [20, 30] mg·h/L}`, random through cycle-to-cycle biological variability.
`K = 8` dose arms span 1,600–3,600 mg/m².

**The model (the prior's engine).** One compartment, linear clearance: `AUC = dose·BSA/CL`,
inverted on a single blood draw, `CL = R/Css`. Its population clearance is calibrated on a
disjoint study cohort; calibrating it on a slice of that cohort is the only source of model
bias.

**The protocol.** Cycle 1 is the standard BSA dose plus one concentration test, identical for
every policy and not scored; the model is inverted once on that test; cycles 2–12 are scored.
Before the draw the model recommends the same arm to every patient and `R_mech = 0` exactly;
the single test is what makes the ODE a prior.

**The five policies**, sharing the identical cycle 1:

| policy | scored cycles | lead-in draw | per-cycle readout | ODE solves |
|---|---|---|---|---|
| BSA static | the same dose, never changed | not read | none | 0 |
| uninformed TS | Thompson sampling on Beta counts | not read | binary | 0 |
| mechanistic ODE | the model's recommendation, re-solved every cycle | read | continuous | 12 |
| ODE, one solve | the cycle-1 recommendation, played every cycle | read | none | 1 |
| hybrid | Theorem 2: Thompson sampling over the latent best arm with prior `P(π* \| π̂)`; the ODE solved once, the binary reward the only later input | read | binary | 1 |

The hybrid's prior is the conditional table `P(π* | π̂)` estimated on the study cohort, so
`H(π* | π̂) = H(µ) − R_mech` holds by construction. Every certificate quantity is measured on
the study cohort alone; only the regret columns read the disjoint evaluation cohort.

## Installation

```bash
pip install -e .                      # numpy, scipy, matplotlib; Python >= 3.10
pip install -r requirements-lock.txt  # the exact versions the reported numbers used
```

## Reproducing the paper

Every target writes into `outputs/`. Seeds and cohort sizes are the scripts' defaults
(evaluation `n = 10,000`, study `n = 2,000`, 11 scored cycles, seeds 11 / 2024), which are
the paper's.

| command | what it produces | in the paper |
|---|---|---|
| `make example` | the certificate and the five policies at the calibrated model; `outputs/example_summary.csv`, `outputs/example_policies.csv` | the policy table of the main text and its appendix version; the measured constants of the calibration appendix; the literature tier |
| `make ablation` | the scalar end-to-end channel; `outputs/channel_ablation.csv` | the scalar channel ablation table |
| `make bias-sweep` | the 33-slice calibration-bias sweep; `outputs/bias_sweep.csv` | the by-bias tables (three slices in the main text, nine in the appendix) |
| `make figures` | `fig_model_distance.pdf`, `fig_hybrid_bias.pdf`, `fig_sensitivity_new.pdf`, `outputs/appendix_sensitivity.json` | the model-distance, hybrid-vs-bias and sensitivity figures; the slice, discretisation and estimator-stability tables |
| `make channel-sweep` | `outputs/param_sweep_gamma1/channel.csv`, `channel_draws.csv`, `figures/channel.pdf` | the channel table (main text and appendix) and the channel figure |
| `make param-sweep-gamma1` | `outputs/param_sweep_gamma1/cycles.csv` | the horizon-scaling table |
| `make param-sweep` | `outputs/param_sweep/*.csv`, `figures/ofat_policies.pdf`, `figures/ofat_certificate.pdf` | the design-axis table and its two figures |
| `make all` | everything, in dependency order | |

The bias sweep and the design sweep pass `--fit-table-on-slice`: the hybrid's prior table is
fitted on the same slice the deployed model is calibrated on, so the study and the deployment
share one population model. That is the run the paper reports. Without the flag the scripts
carry the `γ = 1` table, fitted once, into every slice; the `table_fitted_on` column of the
CSVs records which run a file is. At `γ = 1` the two coincide.

## Where each paper quantity lives

| paper | code |
|---|---|
| `R_mech = I(π*; π̂)` (Def. 2) | `mp_bandit.information_quantities` (plug-in MI, with a shuffled null) |
| `B_µ` (Eq. 3, centred) | `mp_bandit.occupancy_weighted_bias_centred`; the uncentred form `occupancy_weighted_bias_per_patient` is reported alongside |
| `Z = h(M*) + ν`, `σ_h`, `σ_ν`, `L_h`, `d_Z` (Section 2.1) | `mp_bandit.measurement_moments`; the lead-in blood test is the pre-trial measurement |
| `C_ν` (Prop. 1, estimator-free form) | `mp_bandit.capacity_nu` |
| `C(B_µ)` (Prop. 1, bias form) | `mp_bandit.capacity_bound` |
| `B_µ^crit(s)` (Thm. 3) | `mp_bandit.critical_bias` (the one-cycle point `s = N/(N−1)`); `critical_bias_at` for a general target |
| Bayesian regret (Def. 1) | `mp_bandit.run_policy`; expected regret against each patient's own best arm |
| the hybrid prior of Thm. 2 | `mp_bandit.hybrid_prior_table` + `mp_bandit.run_study` |
| the lead-in protocol | `runin.run_runin` |
| the literature-tier constants | `twincert.literature` |

## Layout

```
src/twincert/literature.py            the Tan-sourced published constants, declared once
src/twincert/generator/folfox.py      the plant, the model, cohort sampling, calibration
src/twincert/experiment/mp_bandit.py  policies, information quantities, the certificate
src/twincert/experiment/runin.py      the lead-in cycle (BSA dose + one blood test)
src/twincert/experiment/protocol.py   the three-cohort protocol and its seed derivations
src/twincert/experiment/stats.py      the mean/CI summary and the paired bootstrap
scripts/run_example.py                the experiment driver (five policies + certificate)
scripts/run_bias_sweep.py             the calibration-bias sweep
scripts/run_parameter_sweep.py        the design-axis and channel sweeps
scripts/run_channel_ablation.py       the scalar end-to-end channel ablation
scripts/make_appendix_figures.py      the sensitivity sweeps and their figure
scripts/plot_bias_figures.py          the model-distance and hybrid-vs-bias figures
scripts/plot_parameter_sweep.py       the design-axis and channel figures
scripts/_plotstyle.py                 the shared figure style
```

## License

MIT (see `LICENSE`).
