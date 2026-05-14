"""JWSVI volatility surface with time interpolation.

Re-derives ``nu_tilda`` from the wing slopes:

    - Interpolate (nu * t), phi, p, c linearly (cubic if more than 3 slices).
    - Re-derive nu_tilda from wing slopes:  nu_tilda = 4 * nu * p * c / (p + c)^2.
    - Clip nu_tilda into [NUTILDA_FLOOR, 0.99 * nu] so that
      ``to_svi`` always sees a non-negative ``nu - nu_tilda`` and a strictly
      positive minimum variance.

Note: because nu_tilda is re-derived (not preserved from the per-slice
calibration), the surface is NOT lossless at calibrated knots — see the
"surface stability check" in the notebook for the expected error scale.
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import interp1d

from .svi import JWSVIParam, SVIParam

NUTILDA_FLOOR = 1e-6
_DCF_FLOOR = 1e-4


class JWSVIVolSurface:
    """Per-slice JWSVI parameters joined by time interpolation.

    Parameters
    ----------
    jwsvi_slices : dict
        Mapping ``{label: (JWSVIParam, dcf)}`` where ``dcf`` is the time to
        expiry in years.
    spot : float
        Underlying spot price.
    r : float
        Continuously compounded risk-free rate.
    q : float, default 0.0
        Continuous dividend yield. Forward used by ``implied_vol`` and
        ``total_variance`` is ``spot * exp((r - q) * dcf)``.
    """

    def __init__(
        self,
        jwsvi_slices: Dict[str, Tuple[JWSVIParam, float]],
        spot: float,
        r: float,
        q: float = 0.0,
    ) -> None:
        if len(jwsvi_slices) < 2:
            raise ValueError(
                f"JWSVIVolSurface requires at least 2 expiry slices for time "
                f"interpolation; got {len(jwsvi_slices)}. Common cause: "
                "build_vol_grid filtered out too many expiries. Try lowering "
                "min_open_interest (e.g. to 0 or 1) or min_strikes_per_slice "
                "in data.build_vol_grid, or pass `expiries=` manually."
            )
        self.spot = float(spot)
        self.r = float(r)
        self.q = float(q)

        sorted_items = sorted(jwsvi_slices.items(), key=lambda x: x[1][1])
        self.expiry_strs = [item[0] for item in sorted_items]
        self.dcfs = np.array([item[1][1] for item in sorted_items])
        self.jwsvi_list = [item[1][0] for item in sorted_items]

        nu_arr = np.array([jw.nu for jw in self.jwsvi_list])
        phi_arr = np.array([jw.phi for jw in self.jwsvi_list])
        p_arr = np.array([jw.p for jw in self.jwsvi_list])
        c_arr = np.array([jw.c for jw in self.jwsvi_list])

        kind = "linear" if len(self.dcfs) <= 3 else "cubic"
        fill = "extrapolate"

        # nu*t (total ATM variance) interpolated linearly to keep monotonicity.
        self._nu_t_interp = interp1d(self.dcfs, nu_arr * self.dcfs, kind=kind, fill_value=fill)
        self._phi_interp = interp1d(self.dcfs, phi_arr, kind=kind, fill_value=fill)
        self._p_interp = interp1d(self.dcfs, p_arr, kind=kind, fill_value=fill)
        self._c_interp = interp1d(self.dcfs, c_arr, kind=kind, fill_value=fill)

    @staticmethod
    def _compute_nu_tilda(nu: float, p: float, c: float) -> float:
        """Re-derive nu_tilda from wing slopes.

        Formula: ``nu_tilda = 4 * nu * p * c / (p + c) ** 2``.

        Clip into [NUTILDA_FLOOR, 0.99 * nu]:
          - upper cap at 0.99 * nu so ``to_svi`` keeps ``nu - nu_tilda > 0``;
          - floor at NUTILDA_FLOOR so the SVI radius stays well-defined when
            both wings are very small.
        """
        if p + c < 1e-12:
            nt = nu
        else:
            nt = 4.0 * nu * p * c / ((p + c) ** 2)
        return max(NUTILDA_FLOOR, min(nt, 0.99 * nu))

    def get_jwsvi_at(self, dcf: float) -> JWSVIParam:
        """Interpolate JWSVI parameters at arbitrary tenor ``dcf``.

        Tenors below ``_DCF_FLOOR`` (~9 seconds) are clipped with a warning
        — extrapolation that close to expiry is meaningless.
        """
        dcf_in = float(dcf)
        if dcf_in < _DCF_FLOOR:
            warnings.warn(
                f"JWSVIVolSurface: dcf={dcf_in:g} below floor {_DCF_FLOOR:g}; clipping.",
                RuntimeWarning,
                stacklevel=2,
            )
        dcf = max(dcf_in, _DCF_FLOOR)
        nu = float(self._nu_t_interp(dcf)) / dcf
        phi = float(self._phi_interp(dcf))
        p = max(float(self._p_interp(dcf)), 0.0)
        c = max(float(self._c_interp(dcf)), 0.0)
        nu_tilda = self._compute_nu_tilda(nu, p, c)
        nu = max(nu, nu_tilda)
        if p + c > 1e-12:
            conv_diag = 4.0 * p * c / ((p + c) ** 2)
        else:
            conv_diag = 0.0
        return JWSVIParam(nu=nu, phi=phi, p=p, c=c, nu_tilda=nu_tilda, conv=conv_diag)

    def get_svi_at(self, dcf: float) -> SVIParam:
        """Equivalent raw SVI slice at tenor ``dcf``."""
        return self.get_jwsvi_at(dcf).to_svi(dcf)

    def forward(self, dcf: float) -> float:
        """Forward = spot * exp((r - q) * dcf)."""
        return float(self.spot * np.exp((self.r - self.q) * dcf))

    def implied_vol(self, K: np.ndarray, dcf: float) -> np.ndarray:
        fwd = self.forward(dcf)
        y = np.log(np.asarray(K, dtype=float) / fwd)
        return self.get_svi_at(dcf).implied_vol(y, dcf)

    def total_variance(self, K: np.ndarray, dcf: float) -> np.ndarray:
        fwd = self.forward(dcf)
        y = np.log(np.asarray(K, dtype=float) / fwd)
        return self.get_svi_at(dcf).total_variance(y)

    def implied_vol_grid(self, K_arr: np.ndarray, dcf_arr: np.ndarray) -> np.ndarray:
        K_arr = np.asarray(K_arr, dtype=float)
        out = np.zeros((len(dcf_arr), len(K_arr)))
        for i, d in enumerate(dcf_arr):
            out[i, :] = self.implied_vol(K_arr, d)
        return out
