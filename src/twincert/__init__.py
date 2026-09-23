"""The value of a mechanistic prior for 5-fluorouracil dosing, measured on an in-silico plant.

Given a K-armed choice over FOLFOX dose levels, how many nats about the optimal arm does a
mechanistic model reveal before a trial begins, and what does that buy in cycles a patient
does not spend mis-dosed? The truth is a pharmacokinetic ODE and the model is a simpler ODE,
so every quantity the theory defines is measured rather than assumed.

  * ``generator.folfox``       Tan's two-compartment ODE with saturable (Michaelis-Menten)
                               elimination as the truth, and the one-compartment linear model
                               that approximates it. The observable is the 46-h plasma
                               exposure AUC; the reward is whether it lands in the
                               20-30 mg.h/L therapeutic window.
  * ``experiment.mp_bandit``   the bandit layer: the mechanistic information
                               R_mech = I(pi*; pi_hat), the occupancy-weighted model bias
                               B_mu (Eq. 3), the capacity bound C(B_mu) (Eq. 4), the critical
                               bias (Eq. 8), and the cumulative regret of every policy.
  * ``experiment.runin``       the un-scored first cycle: the body-surface-area dose plus one
                               concentration test, which is what makes the model's
                               recommendation differ between patients at all.
  * ``experiment.protocol``    the three-cohort protocol and its seed derivations, shared by
                               every driver script.

Everything here is in silico. No clinical claim is made.
"""
