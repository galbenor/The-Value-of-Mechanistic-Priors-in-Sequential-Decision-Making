"""The published constants behind the literature-tier certificate, and what is derived from them.

Three numbers are quoted from Tan et al.; the two derived constants are computed here and
imported by every script that needs them, so the derivation exists in one place.

    OMEGA2_VMAX  Tan et al., Table 2: the inter-individual variance of V_max on the log scale.
                 Under the log-normal eta convention, sigma_h = sqrt(OMEGA2_VMAX).
    LSS_MAPE     Tan et al., Sec. 3.4: mean absolute prediction error of the best three-sample
                 limited-sampling scheme under continuous infusion (T = 0.8, 2, 5 h). Their
                 "MAPE" is in absolute units (mg*h/L), not a percentage.
    MEDIAN_AUC   Tan et al., Sec. 3.5 / Fig. 5: the median AUC(0-inf) of the 2,400 mg/m^2
                 continuous-infusion regimen (90% PI [12, 36]).

Derived chain:

    sigma_h  = sqrt(OMEGA2_VMAX)
    sigma_nu = (LSS_MAPE / MEDIAN_AUC) * sqrt(pi / 2)

The sqrt(pi/2) factor converts a mean absolute error to a standard deviation under a
zero-mean Gaussian error, E|X| = sigma sqrt(2/pi). That step and the choice of reference
exposure are both assumptions; `run_example.literature_tier` reports the ceiling under each
alternative.
"""
from __future__ import annotations

import numpy as np

#: Tan et al., Table 2: inter-individual variance of V_max on the log scale.
OMEGA2_VMAX = 0.10

#: Tan et al., Sec. 3.4: best 3-sample LSS mean absolute prediction error (mg*h/L).
LSS_MAPE = 2.76

#: Tan et al., Sec. 3.5 / Fig. 5: median AUC(0-inf) of the 2,400 mg/m^2 continuous-infusion
#: regimen (mg*h/L).
MEDIAN_AUC = 21.0

#: The population spread of the eliminating parameter, on the log scale.
SIGMA_H_LIT = float(np.sqrt(OMEGA2_VMAX))

#: The precision of the pre-dose limited-sampling estimate, relative scale: derived, not
#: quoted, as (mean absolute error / reference exposure) * sqrt(pi/2).
SIGMA_NU_LIT = float((LSS_MAPE / MEDIAN_AUC) * np.sqrt(np.pi / 2.0))
