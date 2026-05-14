"""Monte Carlo Asian-option pricer under Dupire local volatility.

Variance reduction:
    - Antithetic variates (paired ``Z`` and ``-Z`` draws).

Observation schedule (NEW: trading-day averaging by default):
    By default observation dates fall on the next ``n_obs`` weekdays after
    ``start_date`` (today by default).  Consecutive weekday pairs are 1
    calendar day apart, weekend gaps are 3 calendar days apart, and the
    Euler-Maruyama scheme integrates each interval with the matching
    ``sqrt(dt)`` so the diffusion is correct over weekends.  Holidays are
    NOT skipped (no built-in market calendar) — for an exact NYSE schedule
    pass ``obs_dcfs=`` directly with dcfs built from
    ``pandas_market_calendars`` or similar.

    Pass ``obs_schedule='calendar'`` to recover the legacy uniform-in-T
    spacing (used by the Kemna-Vorst closed-form benchmark).

Greeks (Delta, Gamma, Vega, Theta) are computed via finite-difference
repricing under common random numbers.  Vega applies an absolute parallel
shift (``vol_bump_abs``) directly to the local vol diffusion.  Theta
shifts the entire observation schedule by -1 calendar day.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, Optional, Union

import numpy as np
from scipy.stats import norm

from .dupire import DupireLocalVol
from .surface import JWSVIVolSurface

# Calendar-day basis used everywhere in the package (matches data.py:
# ``dcf = dte / 365.0``).
CALENDAR_BASIS = 365.0

DateLike = Union[date, datetime, None]


# ─────────────────────────────────────────────────────────────────────────
# Observation schedules
# ─────────────────────────────────────────────────────────────────────────

def trading_day_obs_dcfs(
    n_obs: int,
    start_date: DateLike = None,
    calendar_basis: float = CALENDAR_BASIS,
) -> np.ndarray:
    """Calendar-year dcfs for the next ``n_obs`` trading days.

    Trading days = weekdays (Mon-Fri).  Holidays are NOT skipped — pure
    weekday calendar.  For an exact NYSE schedule including holidays,
    build the dcfs externally (e.g. via ``pandas_market_calendars``) and
    pass them as ``obs_dcfs=``.

    Parameters
    ----------
    n_obs : int
        Number of trading-day observations.  Must be >= 1.
    start_date : datetime.date or datetime.datetime, optional
        Valuation date (the "T = 0" reference).  Defaults to ``date.today()``.
        The first observation is the next *strictly later* weekday.
    calendar_basis : float, default 365.0
        Calendar days per year used to convert ``(date - start_date).days``
        to a year fraction.  Matches ``data.py``.

    Returns
    -------
    np.ndarray, shape (n_obs,)
        Strictly increasing calendar-year dcfs of each observation.
    """
    if n_obs < 1:
        raise ValueError(f"n_obs must be >= 1, got {n_obs}")
    if start_date is None:
        start_date = date.today()
    elif isinstance(start_date, datetime):
        start_date = start_date.date()

    out = []
    d = start_date
    while len(out) < n_obs:
        d = d + timedelta(days=1)
        if d.weekday() < 5:  # 0=Mon ... 4=Fri
            out.append((d - start_date).days / calendar_basis)
    return np.asarray(out, dtype=float)


def _calendar_obs_dcfs(T: float, n_obs: int) -> np.ndarray:
    """Uniform calendar-time observation dcfs ``i*T/n_obs`` for i=1..n_obs.

    Reproduces the legacy behavior (``dt = T / (n_obs * n_steps_per_obs)``).
    """
    if n_obs < 1:
        raise ValueError(f"n_obs must be >= 1, got {n_obs}")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    return np.linspace(T / n_obs, T, n_obs)


def _resolve_obs_dcfs(
    T: Optional[float],
    n_obs: Optional[int],
    obs_dcfs: Optional[np.ndarray],
    obs_schedule: str,
    start_date: DateLike,
) -> np.ndarray:
    """Resolve the observation dcf vector from possibly partial inputs."""
    if obs_dcfs is not None:
        arr = np.asarray(obs_dcfs, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("obs_dcfs must be a non-empty 1-D array")
        if arr[0] <= 0:
            raise ValueError("obs_dcfs[0] must be > 0 (no observation at t=0)")
        if not np.all(np.diff(arr) > 0):
            raise ValueError("obs_dcfs must be strictly increasing")
        return arr

    if n_obs is None:
        raise ValueError("must pass either obs_dcfs or n_obs")

    if obs_schedule == "trading":
        return trading_day_obs_dcfs(int(n_obs), start_date=start_date)
    if obs_schedule == "calendar":
        if T is None:
            raise ValueError("obs_schedule='calendar' requires T")
        return _calendar_obs_dcfs(float(T), int(n_obs))
    raise ValueError(
        f"obs_schedule must be 'trading' or 'calendar', got {obs_schedule!r}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Closed-form geometric Asian (Kemna-Vorst, generalised to non-uniform t_i)
# ─────────────────────────────────────────────────────────────────────────

def geometric_asian_call_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: Optional[float] = None,
    n_obs: Optional[int] = None,
    call: bool = True,
    q: float = 0.0,
    *,
    obs_dcfs: Optional[np.ndarray] = None,
) -> float:
    """Closed-form discrete geometric Asian price under GBM (Kemna-Vorst).

    Drift uses ``r - q`` (continuous dividend yield); discounting uses ``r``.
    Provided as a public utility for flat-vol benchmarking and unit tests.
    NOT used internally by :meth:`AsianMCPricer.price_asian` (CV was removed —
    see module docstring).

    Two calling conventions:

    1. Uniform spacing (legacy): pass ``T`` and ``n_obs``.  Observations are
       at ``t_i = i * T / n_obs``, i = 1..N.  Closed form uses the
       compact :math:`(N+1)(2N+1)/(6N^2)` variance factor.

    2. Arbitrary spacing: pass ``obs_dcfs``.  Observations are at the given
       times (must be strictly increasing, ``> 0``).  Variance is computed
       as :math:`\\frac{\\sigma^2}{N^2}\\sum_{i,j}\\min(t_i, t_j)` and mean
       uses :math:`\\bar t = (1/N)\\sum t_i`.  ``T_eff = t_N`` is used for
       discounting and forward.

    The two paths agree exactly when ``obs_dcfs = i*T/n_obs`` (modulo
    floating-point).
    """
    b = r - q
    if obs_dcfs is not None:
        t = np.asarray(obs_dcfs, dtype=float)
        N = int(t.size)
        T_eff = float(t[-1])
        sum_min = float(np.sum(np.minimum.outer(t, t)))
        sigma_g2_T = sigma ** 2 * sum_min / (N ** 2)
        mu_g_T = (b - 0.5 * sigma ** 2) * float(t.mean()) + 0.5 * sigma_g2_T
    else:
        if T is None or n_obs is None:
            raise ValueError("must pass either (T, n_obs) or obs_dcfs")
        N = int(n_obs)
        T_eff = float(T)
        sigma_g2_T = sigma ** 2 * T_eff * (N + 1.0) * (2.0 * N + 1.0) / (6.0 * N ** 2)
        mu_g_T = (
            (b - 0.5 * sigma ** 2) * T_eff * (N + 1.0) / (2.0 * N) + 0.5 * sigma_g2_T
        )

    if sigma_g2_T <= 0.0:
        intrinsic = (
            max(S0 * np.exp(mu_g_T) - K, 0.0)
            if call
            else max(K - S0 * np.exp(mu_g_T), 0.0)
        )
        return float(np.exp(-r * T_eff) * intrinsic)

    sd = np.sqrt(sigma_g2_T)
    d1 = (np.log(S0 / K) + mu_g_T + 0.5 * sigma_g2_T) / sd
    d2 = d1 - sd

    if call:
        return float(
            np.exp(-r * T_eff) * (S0 * np.exp(mu_g_T) * norm.cdf(d1) - K * norm.cdf(d2))
        )
    return float(
        np.exp(-r * T_eff) * (K * norm.cdf(-d2) - S0 * np.exp(mu_g_T) * norm.cdf(-d1))
    )


# ─────────────────────────────────────────────────────────────────────────
# Pricer
# ─────────────────────────────────────────────────────────────────────────

class AsianMCPricer:
    """Monte Carlo pricer for arithmetic-average Asian options under local vol.

    By default observations fall on **trading days** (weekdays) starting
    after ``start_date`` (today by default), so consecutive observation
    spacing is 1 calendar day on weekdays and 3 over weekends.  Pass
    ``obs_schedule='calendar'`` to recover the legacy uniform-in-T spacing
    (needed when comparing to the Kemna-Vorst closed form), or ``obs_dcfs=...``
    for a fully custom schedule (e.g. holiday-aware NYSE business days).

    Parameters
    ----------
    S0, r : float
        Spot and risk-free rate.
    T : float, optional
        Calendar maturity (years).  Required when ``obs_schedule='calendar'``.
        In ``'trading'`` mode (or when ``obs_dcfs`` is given) the effective
        maturity is taken from the last observation, so passing ``T`` here
        is informational only.
    n_obs : int, optional
        Number of averaging dates.  Required unless ``obs_dcfs`` is given.
    vol_surface : JWSVIVolSurface or SSVISurface
        Implied vol surface (used for ATM IV reporting).
    local_vol_surface : DupireLocalVol
        Local vol surface used for the Euler-Maruyama diffusion.
    n_steps_per_obs : int, default 1
        Euler sub-steps within each observation interval.  Decouples the
        averaging schedule from the simulation grid; bump this for accuracy
        when intervals are large (e.g. monthly averaging).
    vol_scale : float, default 1.0
        Multiplicative bump applied to local vol.
    vol_bump_abs : float, default 0.0
        Absolute parallel shift added to local vol AFTER ``vol_scale``.
        Used by :func:`compute_greeks` to compute Vega.
    q : float, optional
        Continuous dividend yield. Defaults to ``vol_surface.q``.
    obs_schedule : {'trading', 'calendar'}, default 'trading'
        Observation date convention.  Ignored if ``obs_dcfs`` is given.
    obs_dcfs : np.ndarray, optional
        Explicit calendar-year dcfs of every observation, strictly
        increasing and > 0.  Overrides ``obs_schedule``/``T``/``n_obs``.
        ``self.T`` is set to ``obs_dcfs[-1]``.
    start_date : datetime.date, optional
        Valuation date for ``obs_schedule='trading'``. Defaults to today.
    """

    def __init__(
        self,
        S0: float,
        r: float,
        T: Optional[float] = None,
        n_obs: Optional[int] = None,
        vol_surface: Optional[JWSVIVolSurface] = None,
        local_vol_surface: Optional[DupireLocalVol] = None,
        n_steps_per_obs: int = 1,
        vol_scale: float = 1.0,
        vol_bump_abs: float = 0.0,
        q: Optional[float] = None,
        obs_schedule: str = "trading",
        obs_dcfs: Optional[np.ndarray] = None,
        start_date: DateLike = None,
    ) -> None:
        if vol_surface is None or local_vol_surface is None:
            raise TypeError("vol_surface and local_vol_surface are required")

        self.S0 = float(S0)
        self.r = float(r)
        if q is None:
            q = getattr(vol_surface, "q", 0.0)
        self.q = float(q)

        self.obs_dcfs = _resolve_obs_dcfs(
            T=T,
            n_obs=n_obs,
            obs_dcfs=obs_dcfs,
            obs_schedule=obs_schedule,
            start_date=start_date,
        )
        self.n_obs = int(self.obs_dcfs.size)
        # In trading / custom modes T is *derived* from the schedule.  Any
        # T the user passed is ignored — the docstring documents this.
        self.T = float(self.obs_dcfs[-1])
        self.obs_schedule = obs_schedule if obs_dcfs is None else "custom"
        self.start_date = start_date

        self.n_steps_per_obs = max(int(n_steps_per_obs), 1)
        self.n_steps = self.n_obs * self.n_steps_per_obs

        # Per-observation interval boundaries.  ``boundaries[0] = 0`` is the
        # valuation time; ``boundaries[i+1] = obs_dcfs[i]``.  The Euler
        # sub-step inside interval i is ``intervals[i] / n_steps_per_obs``,
        # which differs across intervals in trading mode (1d weekday vs 3d
        # weekend).  This is the whole point of the rewrite.
        self._boundaries = np.concatenate([[0.0], self.obs_dcfs])
        self._intervals = np.diff(self._boundaries)
        if np.any(self._intervals <= 0):
            raise ValueError("non-positive interval in obs schedule")

        self.vol_surface = vol_surface
        self.local_vol_surface = local_vol_surface
        self.vol_scale = float(vol_scale)
        self.vol_bump_abs = float(vol_bump_abs)
        # ATM IV diagnostic — useful for reporting; no longer drives any logic.
        self.atm_iv = float(vol_surface.implied_vol(np.array([self.S0]), self.T)[0])

    def simulate(self, n_paths: int, antithetic: bool = True) -> np.ndarray:
        """Simulate spot at every observation date.

        Each observation interval is integrated with ``n_steps_per_obs``
        Euler sub-steps of width ``intervals[i] / n_steps_per_obs``, so a
        weekend gap (3 calendar days) gets a ``sqrt(3)`` larger Brownian
        increment than a weekday gap (1 calendar day), which is the correct
        Brownian scaling.

        Returns
        -------
        np.ndarray
            Shape ``(n_paths_actual, n_obs)`` -- spot at each observation.
            ``n_paths_actual == 2 * (n_paths // 2)`` when ``antithetic=True``.
        """
        n_half = n_paths // 2 if antithetic else n_paths
        Z = np.random.randn(n_half, self.n_steps)
        if antithetic:
            Z = np.vstack([Z, -Z])

        n_actual = Z.shape[0]
        S_obs = np.zeros((n_actual, self.n_obs))
        S_prev = np.full(n_actual, self.S0)

        lv_t_lo = self.local_vol_surface.dcf_grid[0]
        lv_t_hi = self.local_vol_surface.dcf_grid[-1]

        step = 0
        for obs_idx in range(self.n_obs):
            sub_dt = self._intervals[obs_idx] / self.n_steps_per_obs
            sqrt_sub_dt = np.sqrt(sub_dt)
            t_left = self._boundaries[obs_idx]
            for sub in range(self.n_steps_per_obs):
                # Calendar-time midpoint of this sub-step
                dcf_t = t_left + (sub + 0.5) * sub_dt
                dcf_t = float(np.clip(dcf_t, lv_t_lo, lv_t_hi))
                sigma_local = (
                    self.local_vol_surface.local_vol_vec(S_prev, dcf_t) * self.vol_scale
                    + self.vol_bump_abs
                )
                sigma_local = np.clip(sigma_local, 0.01, 3.0)
                drift = (self.r - self.q - 0.5 * sigma_local ** 2) * sub_dt
                diffusion = sigma_local * sqrt_sub_dt * Z[:, step]
                S_prev = S_prev * np.exp(drift + diffusion)
                step += 1
            S_obs[:, obs_idx] = S_prev

        return S_obs

    def price_asian(
        self,
        K: float,
        n_paths: int = 100_000,
        call: bool = True,
        antithetic: bool = True,
    ) -> Dict[str, float]:
        """Price an arithmetic-average Asian option.

        Variance reduction is antithetic-only (no control variate; see
        module docstring).  Set ``antithetic=False`` to compare plain MC.

        **Standard error** is computed correctly for the chosen mode:
          - ``antithetic=True``: SE is the std of the ``n_paths/2`` pair
            averages ``(f(Z_i) + f(-Z_i))/2`` divided by ``sqrt(n_paths/2)``.
            This is the textbook antithetic estimator and *does* reflect
            the negative pair correlation that drives the variance gain.
            (Naive ``std(disc_payoff) / sqrt(n_paths)`` would treat the
            paired samples as iid and underestimate the variance reduction.)
          - ``antithetic=False``: usual ``std(disc_payoff) / sqrt(n_paths)``.

        Returns a dict with keys: ``price, std_err, ci_95, n_paths``.
        """
        S = self.simulate(n_paths, antithetic=antithetic)
        n_actual = S.shape[0]
        df = np.exp(-self.r * self.T)

        A_arith = S.mean(axis=1)
        if call:
            payoff = np.maximum(A_arith - K, 0.0)
        else:
            payoff = np.maximum(K - A_arith, 0.0)

        disc_payoff = df * payoff
        price = float(disc_payoff.mean())

        if antithetic:
            n_half = n_actual // 2
            pair_avg = 0.5 * (disc_payoff[:n_half] + disc_payoff[n_half:])
            std_err = float(pair_avg.std(ddof=1) / np.sqrt(n_half))
        else:
            std_err = float(disc_payoff.std(ddof=1) / np.sqrt(n_actual))

        return {
            "price": price,
            "std_err": std_err,
            "ci_95": (price - 1.96 * std_err, price + 1.96 * std_err),
            "n_paths": int(n_actual),
        }


# ─────────────────────────────────────────────────────────────────────────
# Greeks
# ─────────────────────────────────────────────────────────────────────────

def compute_greeks(
    S0: float,
    K: float,
    r: float,
    T: Optional[float],
    n_obs: Optional[int],
    vol_surface: JWSVIVolSurface,
    local_vol_surface: DupireLocalVol,
    n_paths: int = 150_000,
    n_steps_per_obs: int = 1,
    seed: int = 42,
    call: bool = True,
    q: Optional[float] = None,
    obs_schedule: str = "trading",
    obs_dcfs: Optional[np.ndarray] = None,
    start_date: DateLike = None,
) -> Dict[str, float]:
    """Finite-difference Greeks under common random numbers.

    Bumps:
        - Spot:  +/- 1% of S0           -> Delta, Gamma (central)
        - Vol:   +/- 1 absolute vol pt  -> Vega        (per 1 vol point)
        - Time:  -1 calendar day        -> Theta       (per calendar day)

    Vol bump is an absolute parallel shift (``vol_bump_abs``) applied to
    the local-vol diffusion only.

    The observation schedule is **resolved once** and reused (with a
    uniform -1 calendar day shift for theta) so every bumped repricing
    sees the same ``obs_dcfs`` and CRN works exactly between bumps.  In
    trading mode this also pins the same business-day calendar for all
    bumps — without that, two pricers built milliseconds apart could
    accidentally land on different ``date.today()`` values around midnight.

    ``call`` selects the option type for both the MC price and the small-T
    intrinsic fallback.  ``q`` defaults to ``vol_surface.q`` when None.
    """
    dS = S0 * 0.01
    dvol = 0.01
    dT = 1.0 / CALENDAR_BASIS

    if q is None:
        q = getattr(vol_surface, "q", 0.0)

    base_obs_dcfs = _resolve_obs_dcfs(
        T=T,
        n_obs=n_obs,
        obs_dcfs=obs_dcfs,
        obs_schedule=obs_schedule,
        start_date=start_date,
    )
    T_resolved = float(base_obs_dcfs[-1])

    atm_iv_base = vol_surface.implied_vol(np.array([S0]), T_resolved)[0]

    def _intrinsic(S: float, T_use: float) -> float:
        T_safe = max(T_use, 0.0)
        fwd = S * np.exp((r - q) * T_safe)
        df = np.exp(-r * T_safe)
        if call:
            return float(max(fwd - K, 0.0) * df)
        return float(max(K - fwd, 0.0) * df)

    def price_with(S: float, vol_bump: float = 0.0, T_adj: float = 0.0) -> float:
        if abs(T_adj) > 1e-12:
            shifted = base_obs_dcfs + T_adj
            shifted = shifted[shifted > 1e-6]
            if shifted.size == 0:
                return _intrinsic(S, T_resolved + T_adj)
            obs_dcfs_use = shifted
        else:
            obs_dcfs_use = base_obs_dcfs

        T_use = float(obs_dcfs_use[-1])
        if T_use < 0.01:
            return _intrinsic(S, T_use)

        p = AsianMCPricer(
            S0=S,
            r=r,
            n_steps_per_obs=n_steps_per_obs,
            vol_surface=vol_surface,
            local_vol_surface=local_vol_surface,
            vol_scale=1.0,
            vol_bump_abs=vol_bump,
            q=q,
            obs_dcfs=obs_dcfs_use,
        )
        np.random.seed(seed)
        return float(p.price_asian(K, n_paths, call=call)["price"])

    P0 = price_with(S0)
    P_up = price_with(S0 + dS)
    P_dn = price_with(S0 - dS)
    delta = (P_up - P_dn) / (2 * dS)
    gamma = (P_up - 2 * P0 + P_dn) / (dS ** 2)

    P_vup = price_with(S0, vol_bump=dvol)
    P_vdn = price_with(S0, vol_bump=-dvol)
    vega = (P_vup - P_vdn) / 2.0  # per 1 vol point

    P_tminus = price_with(S0, T_adj=-dT)
    theta_per_day = -(P0 - P_tminus)

    return {
        "price": P0,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta_per_day,
        "dollar_delta": delta * S0,
        "dollar_gamma": 0.5 * gamma * S0 ** 2 * 0.01,
        "atm_iv_base": float(atm_iv_base),
        "call": bool(call),
    }
