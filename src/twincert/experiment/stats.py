"""Summary statistics shared by the driver scripts.

`summ` is a mean with a normal-approximation 95% interval. `boot_indices` draws a paired
bootstrap resample from a caller-supplied generator and `boot_ratio_from_indices` evaluates a
ratio of means under it, so several ratios against a common denominator can share one
resample. Which numbers come out is fixed by the caller's seed and the order of the calls.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

#: Mean and half-width of the 95% interval; `s.mean` / `s.ci` and `s[0]` / `s[1]` both work.
Summary = namedtuple("Summary", "mean ci")


def summ(x: np.ndarray) -> Summary:
    """Sample mean and the half-width of its normal-approximation 95% interval."""
    return Summary(float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(x.size)))


def boot_indices(m: int, rng: np.random.Generator, n_boot: int = 2000) -> np.ndarray:
    """`n_boot` resamples of `m` units -> (n_boot, m) index matrix.

    Several ratios against a common denominator share one resample: drawing a fresh matrix per
    comparator would un-pair them and change the generator's draw order.
    """
    return rng.integers(0, m, (n_boot, m))


def boot_ratio_from_indices(num: np.ndarray, den: np.ndarray,
                            idx: np.ndarray) -> tuple[float, float]:
    """Percentile 95% interval for mean(num)/mean(den) under a given paired resample."""
    r = num[idx].mean(axis=1) / np.maximum(den[idx].mean(axis=1), 1e-12)
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))
