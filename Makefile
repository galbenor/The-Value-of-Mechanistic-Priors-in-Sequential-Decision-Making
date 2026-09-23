# Reproduction driver for the in-silico experiments of the paper.
#
#   make install             editable install (numpy, scipy, matplotlib)
#   make example             the experiment at the calibrated model: lead-in protocol, five
#                            policies, the certificate; the policy table of the main text
#   make ablation            the scalar end-to-end channel ablation of the calibration appendix
#   make bias-sweep          the calibration-bias sweep: the by-bias tables and figure
#   make figures             fig_model_distance, fig_hybrid_bias, fig_sensitivity_new and the
#                            sensitivity tables (runs `example` and `bias-sweep` first)
#   make channel-sweep       regret against mechanistic information at the calibrated model:
#                            the channel table and figure
#   make param-sweep-gamma1  the scored-horizon axis at the calibrated model: the horizon table
#   make param-sweep         the one-factor-at-a-time design sweep at the biased operating
#                            point: the design-axis table and its two figures
#   make all                 everything, in dependency order
#
# Every target writes into outputs/ (gitignored). Seeds and cohort sizes are the scripts'
# defaults, which are the paper's.

PY := python3

.PHONY: all install example ablation figures bias-sweep param-sweep param-sweep-gamma1 \
        channel-sweep

# `figures` reads what `example` and `bias-sweep` write; `param-sweep` is the long one and
# produces no input for the appendix figures, so it comes last.
all: example ablation bias-sweep figures channel-sweep param-sweep-gamma1 param-sweep

install:
	$(PY) -m pip install -e .

# Cycle 1 is the BSA dose plus one blood test (unscored, identical for every policy), the ODE
# is inverted once on it, then 11 scored cycles. Prints the mechanistic information, the
# certificate and every policy's regret; writes outputs/example_summary.csv and
# outputs/example_policies.csv. One operating point (the model calibrated on the whole study
# cohort); the miscalibration axis is `bias-sweep`.
example:
	$(PY) scripts/run_example.py

# The scalar channel of the calibration appendix: realised R_mech against its ceiling, plus
# the three-policy regret comparison on the scalar plant.
ablation:
	$(PY) scripts/run_channel_ablation.py

# The calibration-bias sweep (33 slices): for every slice, the hybrid, the ODE solved once and
# the per-cycle controller on the slice's deployed model. The hybrid's prior table is fitted
# on the same slice as the deployed model (--fit-table-on-slice), which is the run the paper
# reports. Writes outputs/bias_sweep.csv, which scripts/plot_bias_figures.py reads.
bias-sweep:
	$(PY) scripts/run_bias_sweep.py --fit-table-on-slice --out outputs

# The one-factor-at-a-time design sweep: four axes (K, N, n_study at three study seeds, lead-in
# length), each varied alone around the operating point. The deployed model is biased,
# calibrated on the slowest-clearing 40% of the study cohort, and the hybrid's prior table is
# fitted on that same slice (--fit-table-on-slice), as in the paper. The plant's population
# constants are not swept. Writes outputs/param_sweep/*.csv and the figures beside them.
param-sweep:
	$(PY) scripts/run_parameter_sweep.py --fit-table-on-slice --out outputs/param_sweep
	$(PY) scripts/plot_parameter_sweep.py --csv-dir outputs/param_sweep \
	      --fig-dir outputs/param_sweep/figures

# The scored-horizon axis at the calibrated model (gamma = 1), the deployment `example`
# publishes, into its own directory so the biased sweep above is not overwritten.
param-sweep-gamma1:
	$(PY) scripts/run_parameter_sweep.py --calib-frac 1.0 --axes cycles \
	      --out outputs/param_sweep_gamma1

# Regret against mechanistic information at the calibrated model. R_mech is varied by the two
# knobs of the pre-dose measurement channel: the assay precision at one lead-in draw (axis
# `channel`) and the number of pooled lead-in draws (axis `channel_draws`). Writes channel.csv
# and channel_draws.csv beside the horizon axis above, then draws figures/channel.{pdf,png}.
channel-sweep:
	$(PY) scripts/run_parameter_sweep.py --calib-frac 1.0 --axes channel,channel_draws \
	      --out outputs/param_sweep_gamma1
	$(PY) scripts/plot_parameter_sweep.py --csv-dir outputs/param_sweep_gamma1 \
	      --fig-dir outputs/param_sweep_gamma1/figures

# The appendix figures, in the order their inputs appear:
#   fig_sensitivity_new.pdf  the sensitivity sweeps        <- measured live
#   fig_model_distance.pdf   model-world distance          <- outputs/appendix_sensitivity.json
#   fig_hybrid_bias.*        regret against measured bias  <- outputs/bias_sweep.csv
figures: example bias-sweep
	$(PY) scripts/make_appendix_figures.py
	$(PY) scripts/plot_bias_figures.py --sweep-csv outputs/bias_sweep.csv
