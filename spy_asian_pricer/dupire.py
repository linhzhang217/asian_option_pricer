"""Dupire local volatility surface from a JWSVI implied vol surface.

Per-strike cubic-spline interpolation of total variance in T provides an
analytic dw/dT (no numerical differencing). Gatheral's butterfly denominator
gives the strike-side normalization. Numerical safety clamps are tracked in
``DupireLocalVol.clamp_stats`` for diagnostics.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline

from .surface import JWSVIVolSurface


class DupireLocalVol:
    """Local volatility surface derived via Dupire's formula."""

    DENOM1_FLOOR = 0.2
    # Bound on (w''/2)/denom1.  Old default 0.75 silently capped legitimate
    # convex smiles; 5.0 still prevents pathological blow-ups but no longer
    # systematically biases short-dated OTM local vol low.  Cap-hit rate is
    # reported in ``clamp_stats``.
    DENS_RATIO_CAP = 5.0
    DWDT_FLOOR = 1e-4
    LV_OVER_ATM_CAP = 10.0

    def __init__(self, vol_surface: JWSVIVolSurface) -> None:
        if len(vol_surface.dcfs) < 2:
            raise ValueError(
                f"DupireLocalVol needs at least 2 calibrated tenors for the "
                f"per-strike cubic spline in T; the surface has only "
                f"{len(vol_surface.dcfs)}. Add more expiries in build_vol_grid."
            )
        self.vol_surface = vol_surface
        self.spot = vol_surface.spot
        self.r = vol_surface.r
        self.q = getattr(vol_surface, "q", 0.0)
        self.clamp_stats: dict = {}
        self._build_grid()

    def _build_grid(self) -> None:
        # Wider strike grid (+/-50%) so the bivariate spline isn't extrapolating
        # at the edges of typical Asian payoffs / stress scenarios.
        self.K_grid = np.linspace(self.spot * 0.5, self.spot * 1.5, 160)
        # Don't extrapolate below the shortest calibrated tenor.
        self.dcf_grid = np.linspace(
            self.vol_surface.dcfs[0],
            self.vol_surface.dcfs[-1],
            80,
        )
        bm_dcfs = self.vol_surface.dcfs

        # Step 1: total variance at benchmarks
        w_matrix = np.zeros((len(self.K_grid), len(bm_dcfs)))
        for k, dcf in enumerate(bm_dcfs):
            fwd = self.spot * np.exp((self.r - self.q) * dcf)
            y = np.log(self.K_grid / fwd)
            svi = self.vol_surface.get_svi_at(dcf)
            w_matrix[:, k] = svi.total_variance(y)

        # Step 2: per-strike cubic spline in T
        dw_dt_splines = []
        w_splines = []
        for j in range(len(self.K_grid)):
            cs = CubicSpline(bm_dcfs, w_matrix[j, :], bc_type="natural")
            w_splines.append(cs)
            dw_dt_splines.append(cs.derivative())

        # Step 3: evaluate local vol on the (T, K) grid
        n_dwdt_floor = 0
        n_denom1_floor = 0
        n_dens_cap = 0
        n_lv_cap = 0
        n_total = 0

        lv_grid = np.zeros((len(self.dcf_grid), len(self.K_grid)))
        for i, dcf in enumerate(self.dcf_grid):
            fwd = self.spot * np.exp((self.r - self.q) * dcf)
            svi = self.vol_surface.get_svi_at(dcf)
            atm_iv = svi.implied_vol(np.array([0.0]), dcf)[0]

            for j, K in enumerate(self.K_grid):
                n_total += 1
                y = np.log(K / fwd)

                dw_dt_raw = float(dw_dt_splines[j](dcf))
                dw_dt = max(dw_dt_raw, self.DWDT_FLOOR)
                if dw_dt_raw < self.DWDT_FLOOR:
                    n_dwdt_floor += 1

                w = max(float(w_splines[j](dcf)), 1e-8)
                dw_dy = svi.dw_dy(np.array([y]))[0]
                d2w_dy2 = svi.d2w_dy2(np.array([y]))[0]

                denom1_raw = 1.0 - dw_dy * (
                    y / w - 0.25 * (-0.25 - 1.0 / w + y ** 2 / (w ** 2)) * dw_dy
                )
                denom1 = max(denom1_raw, self.DENOM1_FLOOR)
                if denom1_raw < self.DENOM1_FLOOR:
                    n_denom1_floor += 1

                density_ratio_raw = 0.5 * d2w_dy2 / denom1
                density_ratio = max(0.0, min(self.DENS_RATIO_CAP, density_ratio_raw))
                if density_ratio_raw > self.DENS_RATIO_CAP or density_ratio_raw < 0.0:
                    n_dens_cap += 1
                denom = denom1 * (1.0 + density_ratio)

                local_var = dw_dt / max(denom, 1e-8)
                lv = np.sqrt(max(local_var, 0.0))
                lv_cap = self.LV_OVER_ATM_CAP * atm_iv
                if lv > lv_cap:
                    n_lv_cap += 1
                    lv = lv_cap
                lv_grid[i, j] = lv

        self.clamp_stats = {
            "dwdt_floor_pct": n_dwdt_floor / n_total * 100,
            "denom1_floor_pct": n_denom1_floor / n_total * 100,
            "density_cap_pct": n_dens_cap / n_total * 100,
            "lv_cap_pct": n_lv_cap / n_total * 100,
            "grid_points": n_total,
        }

        self.lv_interp = RectBivariateSpline(
            self.dcf_grid, self.K_grid, lv_grid, kx=3, ky=3
        )
        self.lv_grid_data = lv_grid

    def _clip(self, S_arr, dcf, warn: bool):
        """Clip (S, dcf) into the grid; optionally warn on any out-of-range."""
        dcf_in = float(dcf)
        dcf_c = float(np.clip(dcf_in, self.dcf_grid[0], self.dcf_grid[-1]))
        S_arr = np.asarray(S_arr, dtype=float)
        S_lo, S_hi = self.K_grid[0], self.K_grid[-1]
        S_c = np.clip(S_arr, S_lo, S_hi)
        if warn:
            clipped = (dcf_c != dcf_in) or bool(
                np.any((S_arr < S_lo) | (S_arr > S_hi))
            )
            if clipped:
                warnings.warn(
                    "DupireLocalVol: input clipped to grid "
                    f"[K=({S_lo:.2f},{S_hi:.2f}), "
                    f"dcf=({self.dcf_grid[0]:.4f},{self.dcf_grid[-1]:.4f})]",
                    RuntimeWarning,
                    stacklevel=3,
                )
        return S_c, dcf_c

    def local_vol(self, S: float, dcf: float) -> float:
        """Scalar local vol at (spot, time). Warns on grid clipping."""
        S_c, dcf_c = self._clip(np.array([float(S)]), dcf, warn=True)
        lv = float(self.lv_interp(dcf_c, S_c[0]))
        return max(lv, 0.0)

    def local_vol_vec(self, S_arr: np.ndarray, dcf: float) -> np.ndarray:
        """Vectorized local vol at a single time across many spots.

        Used inside the MC hot loop so we silently clip extreme paths instead
        of warning on every Euler step.  Negative interpolated values (cubic
        spline overshoot near the grid boundary) are floored at 0.
        """
        S_c, dcf_c = self._clip(S_arr, dcf, warn=False)
        dcf_arr = np.full_like(S_c, dcf_c)
        lv = self.lv_interp(dcf_arr, S_c, grid=False)
        return np.maximum(lv, 0.0)
