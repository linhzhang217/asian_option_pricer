"""SVI / JWSVI parameterizations and per-slice calibrator.

Raw SVI total variance (Gatheral, 2004):
    w(y) = a + b * (rho * (y - m) + sqrt((y - m)^2 + sigma^2))

JWSVI is the Jump-Wing parameterization with financially interpretable
parameters (ATM variance, ATM skew, put/call wing slopes, minimum variance,
convexity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SVIParam:
    """Raw SVI 5-parameter set."""

    a: float       # variance level
    b: float       # vol-of-var (curvature), b >= 0
    rho: float     # correlation, |rho| < 1
    m: float       # moneyness offset
    sigma: float   # vol-of-vol, sigma > 0

    def total_variance(self, y: np.ndarray) -> np.ndarray:
        """w(y) = a + b * (rho*(y-m) + sqrt((y-m)^2 + sigma^2))."""
        dy = y - self.m
        return self.a + self.b * (self.rho * dy + np.sqrt(dy ** 2 + self.sigma ** 2))

    def implied_vol(self, y: np.ndarray, dcf: float) -> np.ndarray:
        """sigma_iv = sqrt(w / T)."""
        w = self.total_variance(y)
        return np.sqrt(np.maximum(w, 1e-10) / dcf)

    def dw_dy(self, y: np.ndarray) -> np.ndarray:
        """First derivative dw/dy."""
        dy = y - self.m
        return self.b * (self.rho + dy / np.sqrt(dy ** 2 + self.sigma ** 2))

    def d2w_dy2(self, y: np.ndarray) -> np.ndarray:
        """Second derivative d^2w/dy^2."""
        dy = y - self.m
        return self.b * self.sigma ** 2 / (dy ** 2 + self.sigma ** 2) ** 1.5

    def to_jwsvi(self, t: float) -> "JWSVIParam":
        """Convert SVI -> JWSVI for slice at maturity ``t`` (years)."""
        w_t = self.total_variance(np.array([0.0]))[0]  # ATM total variance
        nu = w_t / t
        w_t_inv = 1.0 / np.sqrt(w_t) if w_t > 1e-12 else 0.0

        # ATM skew (continuous in m via m / sqrt(m^2 + sigma^2)).
        radius0 = np.sqrt(self.m ** 2 + self.sigma ** 2)
        m_unit = self.m / radius0 if radius0 > 1e-12 else 0.0
        phi = w_t_inv * self.b * 0.5 * (-m_unit + self.rho)

        # Wing slopes
        p = w_t_inv * self.b * (1.0 - self.rho)
        c = w_t_inv * self.b * (1.0 + self.rho)

        # Minimum variance
        min_var = self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho ** 2)
        nu_tilda = min_var / t

        # Convexity
        if abs(self.b) < 1e-12:
            conv = 0.0
        else:
            beta = self.rho - 2.0 * phi * np.sqrt(w_t) / self.b
            radius = np.sqrt(self.m ** 2 + self.sigma ** 2)
            conv = (1.0 - beta ** 2) / max(radius, 1e-12) * (p + c) * 0.5

        return JWSVIParam(nu=nu, phi=phi, p=p, c=c, nu_tilda=nu_tilda, conv=conv)


@dataclass
class JWSVIParam:
    """Jump-Wing SVI 6-parameter set (Gatheral & Jacquier, 2014)."""

    nu: float        # ATM variance per unit time
    phi: float       # ATM skew
    p: float         # put wing slope (>= 0)
    c: float         # call wing slope (>= 0)
    nu_tilda: float  # minimum variance per unit time
    conv: float      # convexity

    def to_svi(self, t: float) -> SVIParam:
        """Convert JWSVI -> SVI for slice at maturity ``t`` (years)."""
        if self.p < 1e-8 and self.c < 1e-8:
            return SVIParam(a=self.nu * t, b=0.0, rho=0.0, m=0.0, sigma=0.0)

        w_t = self.nu * t
        b = np.sqrt(w_t) * 0.5 * (self.p + self.c)
        rho = 1.0 - self.p * np.sqrt(w_t) / b if abs(b) > 1e-12 else 0.0
        beta = rho - 2.0 * self.phi * np.sqrt(w_t) / b if abs(b) > 1e-12 else 0.0

        cos_theta = np.sqrt(max(1.0 - rho ** 2, 0.0))
        delta_nut_over_b = (self.nu - self.nu_tilda) * t / b if abs(b) > 1e-12 else 1e8
        denom_r = 1.0 - cos_theta * np.sqrt(max(1.0 - beta ** 2, 0.0)) - rho * beta
        radius = delta_nut_over_b / max(abs(denom_r), 1e-12)
        radius = min(abs(radius), 1e8)

        sigma = radius * np.sqrt(max(1.0 - beta ** 2, 0.0))
        m_val = radius * beta
        a = self.nu * t + b * (rho * m_val - radius)

        return SVIParam(a=a, b=b, rho=rho, m=m_val, sigma=max(sigma, 1e-8))


def calibrate_svi(
    y: np.ndarray,
    iv: np.ndarray,
    dcf: float,
    weights: Optional[np.ndarray] = None,
) -> SVIParam:
    """Fit a 5-parameter raw SVI slice to a market implied vol smile.

    Parameters
    ----------
    y : np.ndarray
        Log-forward-moneyness, ln(K/F).
    iv : np.ndarray
        Implied volatility quotes corresponding to ``y``.
    dcf : float
        Time to expiry in years (day count fraction).
    weights : np.ndarray, optional
        Fitting weights. Defaults to a vega-like Gaussian weight in y.

    Returns
    -------
    SVIParam
        Calibrated SVI parameters.
    """
    y = np.asarray(y, dtype=float)
    iv = np.asarray(iv, dtype=float)
    w_market = iv ** 2 * dcf  # market total variance

    if weights is None:
        weights = np.exp(-0.5 * y ** 2 / 0.3 ** 2)
        weights = weights / weights.sum()

    def svi_residuals(params):
        a, b, rho, m, sigma = params
        dy = y - m
        w_model = a + b * (rho * dy + np.sqrt(dy ** 2 + sigma ** 2))
        return np.sqrt(weights) * (w_model - w_market)

    atm_var = float(np.interp(0.0, y, w_market))
    a0 = atm_var * 0.5
    b0 = max(atm_var * 0.3, 1e-4)

    x0 = [a0, b0, -0.3, 0.0, 0.2]

    result = least_squares(
        svi_residuals,
        x0,
        bounds=(
            [-np.inf, 1e-8, -0.999, -2.0, 1e-4],
            [np.inf, 5.0, 0.999, 2.0, 5.0],
        ),
        method="trf",
        max_nfev=5000,
    )

    a, b, rho, m, sigma = result.x
    return SVIParam(a=a, b=b, rho=rho, m=m, sigma=sigma)
