"""SSVI: Gatheral surface SVI with power-law phi.

Reference: Gatheral (2014), "Arbitrage-free SVI volatility surfaces".

Surface form (vectorised in log-forward-moneyness ``k``):

    w(k, t) = (theta(t) / 2)
              * [ 1 + rho * phi(theta) * k
                  + sqrt((phi(theta) * k + rho)^2 + 1 - rho^2) ]

with the power-law:

    phi(theta) = eta * theta ** (-gamma)

Three global parameters ``(eta, rho, gamma)`` plus a discrete term
structure ``theta(t_i)``.  The surface is calendar-arbitrage-free by
construction when:

    - theta(t) is non-decreasing in t
    - 0 <= gamma <= 1/2

Two calibration modes:

    'pinned'  -- theta(t_i) := atm_iv_i^2 * t_i, taken directly from each
                 calibrated slice's market ATM IV; only (eta, rho, gamma)
                 are jointly optimised.  3 parameters, fast, but inherits
                 any noise in the ATM term structure.

    'full'    -- (eta, rho, gamma, theta_1, ..., theta_N) all jointly
                 optimised under the monotonicity constraint
                 theta_{i+1} >= theta_i.  Implemented via a cumulative-sum
                 of squares parameterisation theta_i = sum_{j<=i} d_j^2 so
                 the bound becomes unconstrained on ``d_j``.  3 + N
                 parameters; slower; smooths out noisy ATM IVs.

At each fixed tenor the SSVI parameterisation is closed-form-equivalent
to a raw SVI slice, so an SSVI surface plugs into ``DupireLocalVol``
through ``get_svi_at(dcf)`` exactly like ``JWSVIVolSurface`` does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .svi import SVIParam


# ─────────────────────────────────────────────────────────────────────────
# Surface
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class SSVISurface:
    """Gatheral SSVI with power-law phi.

    Plugs into :class:`DupireLocalVol` via duck-typing: exposes ``dcfs``,
    ``spot``, ``r``, ``q`` attrs and ``get_svi_at(dcf)`` that returns an
    equivalent raw :class:`SVIParam` slice (closed-form mapping).
    """

    dcfs: np.ndarray            # tenors at which theta is known
    thetas: np.ndarray          # ATM total variance theta(t_i)
    eta: float
    rho: float
    gamma: float
    spot: float
    r: float
    q: float = 0.0

    def __post_init__(self) -> None:
        idx = np.argsort(np.asarray(self.dcfs, dtype=float))
        self.dcfs = np.asarray(self.dcfs, dtype=float)[idx]
        self.thetas = np.asarray(self.thetas, dtype=float)[idx]

    # ── core SSVI evaluation ────────────────────────────────────────────

    def theta_at(self, dcf: float) -> float:
        """ATM total variance at tenor ``dcf`` (linear interp on theta_i)."""
        return float(np.interp(max(float(dcf), 1e-4), self.dcfs, self.thetas))

    def phi_at(self, dcf: float) -> float:
        """Power-law phi(theta) at tenor ``dcf``."""
        theta = self.theta_at(dcf)
        if theta < 1e-10:
            return 0.0
        return self.eta * theta ** (-self.gamma)

    def forward(self, dcf: float) -> float:
        return float(self.spot * np.exp((self.r - self.q) * float(dcf)))

    # ── closed-form SSVI -> raw SVI at fixed tenor ─────────────────────

    def get_svi_at(self, dcf: float) -> SVIParam:
        """SSVI total variance at fixed ``dcf`` is exactly a raw SVI slice.

        Mapping (Gatheral & Jacquier 2014, Theorem 4.1, with ``phi > 0``)::

            a_svi    = theta * (1 - rho^2) / 2
            b_svi    = theta * phi / 2
            rho_svi  = rho
            m_svi    = -rho / phi
            sigma_svi = sqrt(1 - rho^2) / phi
        """
        theta = self.theta_at(dcf)
        phi = self.phi_at(dcf)
        if abs(phi) < 1e-12:
            # Degenerate: flat smile at this tenor
            return SVIParam(a=theta, b=0.0, rho=0.0, m=0.0, sigma=1e-4)
        rho = self.rho
        return SVIParam(
            a=0.5 * theta * (1.0 - rho ** 2),
            b=0.5 * theta * phi,
            rho=rho,
            m=-rho / phi,
            sigma=max(np.sqrt(max(1.0 - rho ** 2, 0.0)) / phi, 1e-8),
        )

    # ── consumer-facing API (matches JWSVIVolSurface) ──────────────────

    def implied_vol(self, K: np.ndarray, dcf: float) -> np.ndarray:
        fwd = self.forward(dcf)
        y = np.log(np.asarray(K, dtype=float) / fwd)
        return self.get_svi_at(dcf).implied_vol(y, dcf)

    def total_variance(self, K: np.ndarray, dcf: float) -> np.ndarray:
        fwd = self.forward(dcf)
        y = np.log(np.asarray(K, dtype=float) / fwd)
        return self.get_svi_at(dcf).total_variance(y)

    def implied_vol_grid(self, K_arr: np.ndarray, dcf_arr: np.ndarray) -> np.ndarray:
        out = np.zeros((len(dcf_arr), len(K_arr)))
        for i, d in enumerate(dcf_arr):
            out[i, :] = self.implied_vol(K_arr, d)
        return out


# ─────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────

def _ssvi_w(
    k: np.ndarray, theta: np.ndarray, eta: float, rho: float, gamma: float
) -> np.ndarray:
    """Vectorised SSVI total variance ``w(k, theta; eta, rho, gamma)``."""
    phi = eta * np.maximum(theta, 1e-10) ** (-gamma)
    u = phi * k + rho
    return 0.5 * theta * (1.0 + rho * phi * k + np.sqrt(u ** 2 + 1.0 - rho ** 2))


def _stack_calibration_data(
    vol_data: Dict[str, "object"], spot: float, r: float, q: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the global LS arrays ``(k, w_market, slice_idx, dcfs, theta_init)``."""
    items = sorted(vol_data.items(), key=lambda kv: kv[1]["dcf"].iloc[0])
    dcfs = np.array([df["dcf"].iloc[0] for _, df in items], dtype=float)
    theta_init = np.zeros(len(items))
    all_k_list, all_w_list, slice_idx_list = [], [], []
    for i, (_, df) in enumerate(items):
        dcf = float(df["dcf"].iloc[0])
        fwd = spot * np.exp((r - q) * dcf)
        atm_iv = float(
            np.interp(fwd, df["strike"].values, df["impliedVolatility"].values)
        )
        theta_init[i] = atm_iv ** 2 * dcf
        k = np.log(df["strike"].values / fwd)
        w = (df["impliedVolatility"].values ** 2) * dcf
        all_k_list.append(k)
        all_w_list.append(w)
        slice_idx_list.append(np.full_like(k, i, dtype=int))
    all_k = np.concatenate(all_k_list)
    all_w = np.concatenate(all_w_list)
    slice_idx = np.concatenate(slice_idx_list)
    return all_k, all_w, slice_idx, dcfs, theta_init


def calibrate_ssvi(
    vol_data: Dict[str, "object"],
    spot: float,
    r: float,
    q: float = 0.0,
    mode: str = "pinned",
    weight_scale: float = 0.3,
    verbose: bool = True,
    maxiter_full: int = 5000,
) -> SSVISurface:
    """Joint SSVI calibration on a per-expiry vol grid.

    Parameters
    ----------
    vol_data : dict
        ``{label: DataFrame}`` as produced by :func:`build_vol_grid`.
        Each frame must have columns ``strike, impliedVolatility, dcf``.
    spot, r, q : float
        Pricing context.  Forward at tenor ``dcf`` is
        ``spot * exp((r - q) * dcf)``.
    mode : ``'pinned'`` (default) or ``'full'``
        See module docstring.
    weight_scale : float, default 0.3
        Gaussian-in-log-moneyness weight scale (smaller = stronger ATM
        emphasis).  ``0.3`` matches the per-slice SVI default.
    verbose : bool, default True
        Print fitted params + loss summary.
    maxiter_full : int, default 5000
        Iteration cap for ``mode='full'`` (LBFGS-B).

    Returns
    -------
    SSVISurface
    """
    if mode not in ("pinned", "full"):
        raise ValueError(f"mode must be 'pinned' or 'full', got {mode!r}")

    all_k, all_w, slice_idx, dcfs, theta_init = _stack_calibration_data(
        vol_data, spot, r, q
    )
    weights = np.exp(-0.5 * all_k ** 2 / weight_scale ** 2)
    n_obs = len(all_k)
    N = len(dcfs)

    if mode == "pinned":
        # theta fixed at theta_init; only (eta, rho, gamma) optimised
        all_theta = theta_init[slice_idx]

        def loss(params):
            eta, rho, gamma = params
            return float(
                np.sum(weights * (_ssvi_w(all_k, all_theta, eta, rho, gamma) - all_w) ** 2)
            )

        res = minimize(
            loss,
            x0=[1.0, -0.5, 0.4],
            method="L-BFGS-B",
            bounds=[(0.01, 50.0), (-0.999, 0.999), (0.0, 0.499)],
        )
        eta, rho, gamma = res.x
        thetas = theta_init.copy()

    else:  # mode == 'full'
        # (eta, rho, gamma, theta_1, ..., theta_N) all jointly optimised.
        # theta is parameterised as cumsum of squared deltas so the
        # monotone constraint theta_{i+1} >= theta_i is automatic.
        diffs = np.diff(theta_init, prepend=0.0)
        diffs = np.maximum(diffs, 1e-6)
        delta_init = np.sqrt(diffs)
        x0 = np.concatenate([[1.0, -0.5, 0.4], delta_init])
        bounds = [(0.01, 50.0), (-0.999, 0.999), (0.0, 0.499)] + [
            (1e-6, None)
        ] * N

        def loss_full(params):
            eta_, rho_, gamma_ = params[0], params[1], params[2]
            deltas = params[3:]
            theta_arr = np.cumsum(deltas ** 2)
            all_theta = theta_arr[slice_idx]
            return float(
                np.sum(
                    weights
                    * (_ssvi_w(all_k, all_theta, eta_, rho_, gamma_) - all_w) ** 2
                )
            )

        res = minimize(
            loss_full,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter_full, "maxfun": 50000},
        )
        eta, rho, gamma = res.x[0], res.x[1], res.x[2]
        deltas = res.x[3:]
        thetas = np.cumsum(deltas ** 2)

    if verbose:
        rmse = float(np.sqrt(res.fun / max(n_obs, 1)))
        print(f"SSVI calibration (mode='{mode}'):")
        print(f"  eta   = {eta:.4f}   (skew curvature scale)")
        print(f"  rho   = {rho:+.4f}   (spot-vol correlation, equity skew is negative)")
        print(f"  gamma = {gamma:.4f}   (term-structure decay, in [0, 0.5])")
        print(f"  fit    weighted SS = {res.fun:.4e}, n_obs = {n_obs}, rmse(w) = {rmse:.4e}")
        if mode == "full":
            print(f"  theta(t): {N} knots, monotone-by-construction")

    return SSVISurface(
        dcfs=dcfs, thetas=thetas,
        eta=float(eta), rho=float(rho), gamma=float(gamma),
        spot=float(spot), r=float(r), q=float(q),
    )
