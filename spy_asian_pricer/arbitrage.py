"""No-arbitrage diagnostics for an SVI/JWSVI implied vol surface.

Three independent checks:
    1. Butterfly arbitrage  -> Gatheral PDF positivity per slice.
    2. Calendar arbitrage   -> total variance non-decreasing in T per strike.
    3. Spread arbitrage     -> call prices monotone decreasing in K with
                              slope bounded below by -exp(-rT).

Plus :func:`filter_butterfly_arbitrage`, a "trader-style" post-calibration
helper that drops slices whose SVI fit produced a catastrophically
negative density.  In production this kind of slice is excluded by hand
after eyeballing the smile; the helper just automates a threshold cut.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm

from .surface import JWSVIVolSurface
from .svi import SVIParam

TOLERANCE = 1e-8


def check_butterfly_arbitrage(
    svi: SVIParam,
    dcf: float,
    y_range: Tuple[float, float] = (-0.5, 0.5),
    n: int = 200,
) -> Tuple[bool, int, float]:
    """Check PDF positivity for a single SVI slice.

    Returns
    -------
    ok : bool
        True iff no violations beyond ``TOLERANCE``.
    violations : int
        Number of grid points violating positivity.
    g_min : float
        Minimum value of Gatheral's density discriminant ``g(y)``.
    """
    y = np.linspace(y_range[0], y_range[1], n)
    w = svi.total_variance(y)
    w1 = svi.dw_dy(y)
    w2 = svi.d2w_dy2(y)

    g = (1.0 - y * w1 / (2 * w)) ** 2 - w1 ** 2 / 4 * (1.0 / w + 0.25) + w2 / 2

    violations = int(np.sum(g < -TOLERANCE))
    return violations == 0, violations, float(g.min())


def check_calendar_arbitrage(
    vol_surface: JWSVIVolSurface,
    K: np.ndarray,
) -> Tuple[bool, int, List[dict]]:
    """Total variance must be non-decreasing in T at every strike."""
    violations = 0
    details: List[dict] = []
    dcfs = vol_surface.dcfs

    K = np.asarray(K, dtype=float)
    for j in range(len(K)):
        tvars = np.array([
            vol_surface.total_variance(np.array([K[j]]), dcf)[0] for dcf in dcfs
        ])
        diffs = np.diff(tvars)
        for t_idx, d in enumerate(diffs):
            if d < -TOLERANCE:
                violations += 1
                details.append(
                    {
                        "strike": float(K[j]),
                        "dcf_from": float(dcfs[t_idx]),
                        "dcf_to": float(dcfs[t_idx + 1]),
                        "tvar_decrease": float(d),
                    }
                )

    return violations == 0, violations, details


def check_spread_arbitrage(
    vol_surface: JWSVIVolSurface,
    dcf: float,
    K: np.ndarray,
    r: float,
    q: float = None,
) -> Tuple[bool, int, List[dict]]:
    """Call price monotonicity and slope bound (>= -exp(-rT)).

    Forward uses ``r - q``.  If ``q`` is None it is taken from
    ``vol_surface.q``.
    """
    if q is None:
        q = getattr(vol_surface, "q", 0.0)
    K = np.asarray(K, dtype=float)
    fwd = vol_surface.spot * np.exp((r - q) * dcf)
    df = np.exp(-r * dcf)
    iv_arr = vol_surface.implied_vol(K, dcf)

    var_sqrt = iv_arr * np.sqrt(dcf)
    d1 = np.log(fwd / K) / var_sqrt + 0.5 * var_sqrt
    d2 = d1 - var_sqrt
    call_prices = df * (fwd * norm.cdf(d1) - K * norm.cdf(d2))

    violations = 0
    details: List[dict] = []

    for i in range(len(K) - 1):
        dK = K[i + 1] - K[i]
        dC = call_prices[i + 1] - call_prices[i]
        slope = dC / dK
        if dC > TOLERANCE:
            violations += 1
            details.append(
                {
                    "type": "monotonicity",
                    "K_range": (float(K[i]), float(K[i + 1])),
                    "slope": float(slope),
                }
            )
        if slope < -(df + TOLERANCE):
            violations += 1
            details.append(
                {
                    "type": "slope_bound",
                    "K_range": (float(K[i]), float(K[i + 1])),
                    "slope": float(slope),
                    "bound": float(-df),
                }
            )

    return violations == 0, violations, details


def filter_butterfly_arbitrage(
    svi_slices: Dict[str, Tuple[SVIParam, float]],
    threshold: float = -0.05,
    y_range: Tuple[float, float] = (-0.5, 0.5),
    n: int = 200,
    verbose: bool = True,
) -> Tuple[Dict[str, Tuple[SVIParam, float]], List[dict]]:
    """Drop SVI slices whose Gatheral density discriminant ``g(y)`` falls
    below ``threshold`` (catastrophically negative density).

    This is the "trader-style" cleanup that real desks do manually
    after eyeballing each calibrated smile.  In production the trader
    pulls the bad slice out of the calibration set, optionally re-fits
    with tighter bounds or different weights, or marks the surface
    around it by hand.  Here we just drop the offending slice; the
    JWSVI time interpolator then bridges that tenor smoothly using the
    surviving neighbouring knots.

    Default ``threshold = -0.05`` is conservative -- only catches
    catastrophic fits (g_min more negative than -0.05).  Tighten to
    ``threshold = -0.01`` for a stricter cut, loosen to ``-0.5`` to
    only catch the absolutely worst.

    Parameters
    ----------
    svi_slices : dict
        Mapping ``{label: (SVIParam, dcf)}`` -- the per-slice fits.
    threshold : float, default -0.05
        Slice is dropped iff ``g_min < threshold``.
    y_range, n : passed through to :func:`check_butterfly_arbitrage`.
    verbose : bool, default True
        Print one line per dropped slice.

    Returns
    -------
    kept : dict
        Same shape as input, with offending slices removed.
    dropped : list of dict
        Per-dropped-slice info ``{label, dcf, g_min, n_violations}``.
    """
    kept: Dict[str, Tuple[SVIParam, float]] = {}
    dropped: List[dict] = []
    for label, (svi, dcf) in svi_slices.items():
        ok, n_viol, g_min = check_butterfly_arbitrage(svi, dcf, y_range=y_range, n=n)
        if g_min < threshold:
            info = {
                "label": label,
                "dcf": float(dcf),
                "g_min": float(g_min),
                "n_violations": int(n_viol),
            }
            dropped.append(info)
            if verbose:
                print(
                    f"  DROP {label} (dcf={dcf:.4f}): "
                    f"g_min={g_min:.3e}, n_viol={n_viol}"
                )
        else:
            kept[label] = (svi, dcf)

    if verbose:
        n_in = len(svi_slices)
        n_out = len(kept)
        print(
            f"filter_butterfly_arbitrage(threshold={threshold}): "
            f"kept {n_out}/{n_in} slices, dropped {len(dropped)}."
        )
    return kept, dropped
