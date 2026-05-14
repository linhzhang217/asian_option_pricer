"""End-to-end smoke tests that run offline (no yfinance).

Builds a synthetic flat-IV vol grid, runs the full pipeline, and checks:
    * SVI fits a flat smile to ~0 RMSE
    * SVI <-> JWSVI round-trips
    * Geometric Asian Kemna-Vorst closed form reduces to BS for n_obs=1
    * MC arithmetic Asian (antithetic only, no CV) stays within 3*SE of the
      closed-form geometric Asian price under a flat vol (the two averages
      are highly correlated, so the gap is small, ~10c on a $25 ATM 6M)
    * Asian MC price stays close (3*SE + 5c) to the European Black-Scholes
      price for a 1-observation Asian (which IS a European).
    * Trading-day observation schedule skips weekends correctly.

Tests that require uniform-in-T spacing (Kemna-Vorst comparison, 1-obs
European equivalence) explicitly pass ``obs_schedule='calendar'`` because
the new pricer default is ``'trading'`` (weekday calendar with weekend
gaps), which would otherwise break those closed-form benchmarks.
"""

from __future__ import annotations

from datetime import date

import numpy as np
from scipy.stats import norm

from spy_asian_pricer import (
    AsianMCPricer,
    DupireLocalVol,
    JWSVIVolSurface,
    SSVISurface,
    SVIParam,
    calibrate_ssvi,
    calibrate_svi,
    geometric_asian_call_price,
    trading_day_obs_dcfs,
)
from spy_asian_pricer.ssvi import _ssvi_w


def _bs_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    sd = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / sd
    d2 = d1 - sd
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _build_flat_surface(spot=500.0, r=0.04, vol=0.20):
    """Build a synthetic JWSVIVolSurface from flat IV at multiple expiries."""
    expiries = [(30 / 365.0, "1M"), (90 / 365.0, "3M"), (182 / 365.0, "6M"), (365 / 365.0, "1Y")]
    y_grid = np.linspace(-0.2, 0.2, 11)
    iv_flat = np.full_like(y_grid, vol)

    jwsvi_slices = {}
    for dcf, label in expiries:
        svi = calibrate_svi(y_grid, iv_flat, dcf)
        jwsvi_slices[label] = (svi.to_jwsvi(dcf), dcf)
    return JWSVIVolSurface(jwsvi_slices, spot=spot, r=r)


# -------------------- SVI sanity --------------------

def test_svi_fits_flat_smile():
    y = np.linspace(-0.3, 0.3, 21)
    iv = np.full_like(y, 0.25)
    dcf = 0.5
    svi = calibrate_svi(y, iv, dcf)
    iv_fit = svi.implied_vol(y, dcf)
    rmse = float(np.sqrt(np.mean((iv_fit - iv) ** 2)))
    assert rmse < 1e-3, f"SVI should fit flat smile to high accuracy, got rmse={rmse}"


def test_svi_to_jwsvi_round_trip():
    svi = SVIParam(a=0.01, b=0.05, rho=-0.3, m=0.0, sigma=0.1)
    t = 0.5
    y = np.linspace(-0.2, 0.2, 11)
    jw = svi.to_jwsvi(t)
    svi_rt = jw.to_svi(t)
    iv_orig = svi.implied_vol(y, t)
    iv_rt = svi_rt.implied_vol(y, t)
    max_err = float(np.max(np.abs(iv_orig - iv_rt)))
    assert max_err < 5e-3, f"SVI->JWSVI round-trip max IV error {max_err}"


# -------------------- Surface --------------------

def test_surface_implied_vol_constant_for_flat_input():
    surface = _build_flat_surface(vol=0.20)
    iv = surface.implied_vol(np.array([450.0, 500.0, 550.0]), dcf=0.25)
    assert np.allclose(iv, 0.20, atol=1e-3), iv


# -------------------- Geometric Asian closed-form --------------------

def test_geometric_asian_matches_european_for_one_obs():
    S0, K, r, sigma, T = 100.0, 100.0, 0.04, 0.20, 1.0
    geo = geometric_asian_call_price(S0, K, r, sigma, T, n_obs=1)
    bs = _bs_call(S0, K, r, sigma, T)
    assert abs(geo - bs) < 1e-6, (geo, bs)


# -------------------- MC pricer --------------------

def test_mc_asian_close_to_geometric_under_flat_vol():
    """Under flat IV, the arithmetic Asian (antithetic-only MC) should
    sit slightly above the closed-form geometric Asian (Kemna-Vorst)
    because arithmetic >= geometric pointwise, with a gap of order
    (sigma^2 * T) cents per dollar of price.  We check the gap is
    positive and within a generous 3*SE + 30c tolerance."""
    spot = 500.0
    r = 0.04
    vol = 0.20
    T = 0.5
    n_obs = 26  # weekly

    surface = _build_flat_surface(spot=spot, r=r, vol=vol)
    lv = DupireLocalVol(surface)

    pricer = AsianMCPricer(
        S0=spot, r=r, T=T, n_obs=n_obs,
        vol_surface=surface, local_vol_surface=lv,
        n_steps_per_obs=1,
        obs_schedule="calendar",  # KV closed form assumes uniform spacing
    )

    np.random.seed(123)
    res = pricer.price_asian(K=spot, n_paths=80_000)

    assert res["price"] > 0
    assert res["std_err"] > 0
    assert res["n_paths"] == 80_000

    geom_kv = geometric_asian_call_price(spot, spot, r, vol, T, n_obs, q=0.0)
    gap = res["price"] - geom_kv
    tol = 3.0 * res["std_err"] + 0.30

    # Arithmetic should exceed geometric by a small but positive amount.
    assert -tol < gap < 1.0, (
        f"arith MC ${res['price']:.4f} vs geom KV ${geom_kv:.4f}: "
        f"gap={gap:.4f}, tol={tol:.4f}"
    )


def test_mc_one_obs_asian_matches_european():
    """A 1-observation arithmetic Asian IS a European. MC should match BS.

    Forces ``obs_schedule='calendar'`` so the single observation lands at
    T (not at the next trading day, which would be only ~1-3 calendar
    days away regardless of T)."""
    spot = 100.0
    r = 0.04
    vol = 0.20
    T = 0.5

    surface = _build_flat_surface(spot=spot, r=r, vol=vol)
    lv = DupireLocalVol(surface)
    pricer = AsianMCPricer(
        S0=spot, r=r, T=T, n_obs=1,
        vol_surface=surface, local_vol_surface=lv,
        n_steps_per_obs=20,
        obs_schedule="calendar",
    )
    np.random.seed(7)
    res = pricer.price_asian(K=spot, n_paths=80_000)
    bs = _bs_call(spot, spot, r, vol, T)
    err = abs(res["price"] - bs)
    # 3*SE tolerance
    tol = 3.0 * res["std_err"] + 0.05
    assert err < tol, f"MC ${res['price']:.4f} vs BS ${bs:.4f} (err={err:.4f}, tol={tol:.4f})"


def test_antithetic_se_strictly_lower_than_plain():
    """Antithetic SE should be strictly lower than plain MC SE at the same
    n_paths on an ATM Asian (where f(Z) and f(-Z) are negatively correlated).
    Verifies the pair-averaged SE formula in price_asian.

    Pinned to ``obs_schedule='calendar'`` so the result is reproducible
    independently of what day of the week the test happens to run."""
    spot, r, vol, T = 100.0, 0.04, 0.20, 0.5
    surface = _build_flat_surface(spot=spot, r=r, vol=vol)
    lv = DupireLocalVol(surface)
    pricer = AsianMCPricer(
        S0=spot, r=r, T=T, n_obs=26,
        vol_surface=surface, local_vol_surface=lv,
        n_steps_per_obs=1,
        obs_schedule="calendar",
    )
    np.random.seed(123)
    res_anti = pricer.price_asian(K=spot, n_paths=80_000, antithetic=True)
    np.random.seed(123)
    res_plain = pricer.price_asian(K=spot, n_paths=80_000, antithetic=False)

    # SE should drop noticeably for ATM (typical 25-40% on ATM Asian)
    assert res_anti["std_err"] < res_plain["std_err"], (
        f"antithetic SE ({res_anti['std_err']:.4f}) should be < "
        f"plain SE ({res_plain['std_err']:.4f}); the pair-averaged "
        f"SE formula is broken."
    )
    se_ratio = res_anti["std_err"] / res_plain["std_err"]
    assert se_ratio < 0.95, (
        f"antithetic SE ratio {se_ratio:.3f} too close to 1 — "
        f"variance reduction should be material on ATM"
    )


def test_mc_one_obs_asian_put_matches_european():
    """1-obs arithmetic Asian put = BS put.

    Same calendar-mode pin as the call version above."""
    spot = 100.0
    r = 0.04
    vol = 0.20
    T = 0.5

    surface = _build_flat_surface(spot=spot, r=r, vol=vol)
    lv = DupireLocalVol(surface)
    pricer = AsianMCPricer(
        S0=spot, r=r, T=T, n_obs=1,
        vol_surface=surface, local_vol_surface=lv,
        n_steps_per_obs=20,
        obs_schedule="calendar",
    )
    np.random.seed(7)
    res = pricer.price_asian(K=spot, n_paths=80_000, call=False)
    # BS put = BS call - (S - K e^{-rT})  with q=0
    bs_call = _bs_call(spot, spot, r, vol, T)
    bs_put = bs_call - (spot - spot * np.exp(-r * T))
    err = abs(res["price"] - bs_put)
    tol = 3.0 * res["std_err"] + 0.05
    assert err < tol, f"MC put ${res['price']:.4f} vs BS put ${bs_put:.4f} (err={err:.4f}, tol={tol:.4f})"


# -------------------- Trading-day observation schedule --------------------

def test_trading_day_obs_dcfs_skips_weekends():
    """Five trading days starting from a Wednesday should land on
    Thu (1d), Fri (2d), Mon (5d — weekend gap), Tue (6d), Wed (7d).
    Spacing = [1, 1, 3, 1, 1] calendar days / 365."""
    start = date(2024, 1, 3)  # Wednesday
    obs = trading_day_obs_dcfs(5, start_date=start)
    expected = np.array([1, 2, 5, 6, 7]) / 365.0
    assert np.allclose(obs, expected, atol=1e-12), (obs, expected)
    # And the per-interval gap pattern
    intervals = np.diff(np.concatenate([[0.0], obs])) * 365.0
    assert np.allclose(intervals, [1, 1, 3, 1, 1], atol=1e-9), intervals


def test_trading_day_starting_friday_jumps_over_weekend_first():
    """If the valuation date is a Friday, the first observation is
    Monday (3 calendar days later), not Saturday."""
    start = date(2024, 1, 5)  # Friday
    obs = trading_day_obs_dcfs(3, start_date=start)
    # Mon (3d), Tue (4d), Wed (5d)
    expected = np.array([3, 4, 5]) / 365.0
    assert np.allclose(obs, expected, atol=1e-12), (obs, expected)


def test_pricer_trading_mode_overrides_T_with_last_obs():
    """In trading mode, the pricer's T is taken from obs_dcfs[-1], not
    the user-passed T (which is informational only)."""
    spot, r, vol = 100.0, 0.04, 0.20
    surface = _build_flat_surface(spot=spot, r=r, vol=vol)
    lv = DupireLocalVol(surface)
    pricer = AsianMCPricer(
        S0=spot, r=r, T=0.5, n_obs=10,
        vol_surface=surface, local_vol_surface=lv,
        obs_schedule="trading",
        start_date=date(2024, 1, 3),
    )
    # 10 trading days starting Wed 2024-01-03 ends on Wed 2024-01-17:
    #   Thu(1) Fri(2) Mon(5) Tue(6) Wed(7) Thu(8) Fri(9) Mon(12) Tue(13) Wed(14)
    # so T = 14 calendar days / 365.
    expected_T = 14.0 / 365.0
    assert abs(pricer.T - expected_T) < 1e-10, (pricer.T, expected_T)
    assert pricer.n_obs == 10


def test_pricer_explicit_obs_dcfs_overrides_T_and_schedule():
    """Passing obs_dcfs directly should override both T and obs_schedule."""
    spot, r, vol = 100.0, 0.04, 0.20
    surface = _build_flat_surface(spot=spot, r=r, vol=vol)
    lv = DupireLocalVol(surface)
    custom = np.array([0.10, 0.20, 0.40])
    pricer = AsianMCPricer(
        S0=spot, r=r, T=999.0, n_obs=99,  # both ignored
        vol_surface=surface, local_vol_surface=lv,
        obs_dcfs=custom,
        obs_schedule="trading",  # also ignored when obs_dcfs is given
    )
    assert pricer.n_obs == 3
    assert abs(pricer.T - 0.40) < 1e-12
    assert np.allclose(pricer.obs_dcfs, custom)


def test_pricer_trading_default_runs_end_to_end():
    """Smoke test that the trading-mode default actually prices something."""
    spot, r, vol = 100.0, 0.04, 0.20
    surface = _build_flat_surface(spot=spot, r=r, vol=vol)
    lv = DupireLocalVol(surface)
    pricer = AsianMCPricer(
        S0=spot, r=r, T=0.5, n_obs=63,  # ~3M of trading days
        vol_surface=surface, local_vol_surface=lv,
    )
    assert pricer.obs_schedule == "trading"
    np.random.seed(7)
    res = pricer.price_asian(K=spot, n_paths=20_000)
    assert res["price"] > 0
    assert res["std_err"] > 0


def test_geometric_asian_with_explicit_obs_dcfs_matches_uniform():
    """The generalised closed form should equal the uniform-spacing form
    when fed uniform dcfs."""
    S0, K, r, sigma, T, n_obs = 100.0, 100.0, 0.04, 0.20, 0.5, 26
    p_uniform = geometric_asian_call_price(S0, K, r, sigma, T, n_obs)
    obs_dcfs = np.linspace(T / n_obs, T, n_obs)
    p_general = geometric_asian_call_price(S0, K, r, sigma, obs_dcfs=obs_dcfs)
    assert abs(p_uniform - p_general) < 1e-10, (p_uniform, p_general)


def test_geometric_asian_non_uniform_dcfs_runs():
    """Non-uniform dcfs (trading-day spacing) should produce a sensible
    price — between the uniform-spacing version and the European."""
    S0, K, r, sigma = 100.0, 100.0, 0.04, 0.20
    obs_dcfs = trading_day_obs_dcfs(126, start_date=date(2024, 1, 3))
    p_trading = geometric_asian_call_price(S0, K, r, sigma, obs_dcfs=obs_dcfs)
    p_uniform = geometric_asian_call_price(S0, K, r, sigma,
                                            T=float(obs_dcfs[-1]), n_obs=126)
    # Non-uniform schedule should be very close to uniform at same N and
    # same T_eff (spacing differences are second-order in the variance sum).
    assert abs(p_trading - p_uniform) < 0.05, (p_trading, p_uniform)


# -------------------- SSVI --------------------

def _make_ssvi_synthetic_grid(spot=714.0, r=0.036, q=0.011,
                              eta=1.2, rho=-0.55, gamma=0.35):
    """Generate a synthetic per-expiry IV grid sampled exactly from a known
    SSVI surface, so we can recover (eta, rho, gamma) cleanly."""
    import pandas as pd
    vol_data = {}
    for label, dcf in [("1M", 30 / 365.0), ("3M", 90 / 365.0),
                       ("6M", 0.5), ("1Y", 1.0), ("2Y", 2.0)]:
        fwd = spot * np.exp((r - q) * dcf)
        strikes = spot * np.exp(np.linspace(-0.2, 0.2, 31))
        k = np.log(strikes / fwd)
        theta = 0.04 * dcf
        w = _ssvi_w(k, np.full_like(k, theta), eta, rho, gamma)
        iv = np.sqrt(w / dcf)
        vol_data[label] = pd.DataFrame({
            "strike": strikes, "impliedVolatility": iv,
            "dcf": dcf, "fwd": fwd, "logMoneyness": k,
        })
    return vol_data, spot, r, q


def test_ssvi_pinned_recovers_synthetic_truth():
    """SSVI pinned-mode calibration should recover (eta, rho, gamma) on a
    surface generated from known truth, since theta(t) is exact."""
    vol_data, spot, r, q = _make_ssvi_synthetic_grid()
    s = calibrate_ssvi(vol_data, spot=spot, r=r, q=q, mode="pinned",
                       verbose=False)
    assert abs(s.eta - 1.2) < 5e-2,   f"eta off: {s.eta}"
    assert abs(s.rho + 0.55) < 5e-2,  f"rho off: {s.rho}"
    assert abs(s.gamma - 0.35) < 5e-2, f"gamma off: {s.gamma}"


def test_ssvi_full_recovers_synthetic_truth_and_theta():
    """Full mode should recover both globals AND theta(t) on synthetic data."""
    vol_data, spot, r, q = _make_ssvi_synthetic_grid()
    s = calibrate_ssvi(vol_data, spot=spot, r=r, q=q, mode="full",
                       verbose=False)
    assert abs(s.eta - 1.2) < 5e-2
    assert abs(s.rho + 0.55) < 5e-2
    assert abs(s.gamma - 0.35) < 5e-2
    # theta(t) = 0.04 * t should be recovered
    expected_thetas = 0.04 * s.dcfs
    assert np.max(np.abs(s.thetas - expected_thetas)) < 5e-3, (
        s.thetas, expected_thetas
    )


def test_ssvi_plugs_into_dupire_and_pricer():
    """SSVI surface should drop into DupireLocalVol and AsianMCPricer
    via duck-typed get_svi_at()."""
    vol_data, spot, r, q = _make_ssvi_synthetic_grid()
    surf = calibrate_ssvi(vol_data, spot=spot, r=r, q=q, mode="pinned",
                          verbose=False)
    lv = DupireLocalVol(surf)
    # Synthetic SSVI is calendar-arb-free by construction, so clamps
    # should fire on essentially zero of the grid.
    assert lv.clamp_stats["dwdt_floor_pct"] < 5.0
    assert lv.clamp_stats["denom1_floor_pct"] < 5.0

    p = AsianMCPricer(S0=spot, r=r, T=0.5, n_obs=26,
                      vol_surface=surf, local_vol_surface=lv)
    np.random.seed(99)
    res = p.price_asian(K=spot, n_paths=20_000)
    assert res["price"] > 0
    assert res["std_err"] > 0
