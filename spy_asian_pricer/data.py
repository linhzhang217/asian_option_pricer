"""Helpers for pulling SPY (or any underlying) option chains from Yahoo Finance.

Optional: requires the ``yfinance`` and ``pandas`` extras (install via
``pip install spy-asian-pricer[data]``).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

try:
    import pandas as pd
    import yfinance as yf
except ImportError as e:  # pragma: no cover - exercised only without extras
    raise ImportError(
        "spy_asian_pricer.data requires `pandas` and `yfinance`. "
        "Install with `pip install spy-asian-pricer[data]`."
    ) from e


# ─────────────────────────────────────────────────────────────────────────
# Black-Scholes helpers (used to re-imply IV from lastPrice, since Yahoo's
# own `impliedVolatility` field is unreliable outside US market hours).
# ─────────────────────────────────────────────────────────────────────────

def bs_european_price(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    call: bool = True,
) -> float:
    """Black-Scholes European option price with continuous dividend yield ``q``."""
    if T <= 0:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    if sigma <= 0:
        fwd = S * np.exp((r - q) * T)
        df_ = np.exp(-r * T)
        intr = max(fwd - K, 0.0) if call else max(K - fwd, 0.0)
        return df_ * intr
    sd = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sd
    d2 = d1 - sd
    if call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol_from_price(
    price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    call: bool = True,
    lo: float = 1e-4,
    hi: float = 5.0,
) -> float:
    """Reverse-imply Black-Scholes volatility from an observed option price.

    Uses Brent on ``[lo, hi]``.  Returns ``NaN`` if the price is below the
    no-arbitrage forward intrinsic, or if the solver brackets fail.

    NOTE on staleness: ``price`` here is Yahoo's ``lastPrice`` (last trade,
    not mid).  For illiquid strikes this can be hours or days old, so the
    re-implied IV is only as fresh as the most recent trade.  In practice
    SPY ATM strikes update intraday; deep-OTM wings can lag a session.
    """
    if not np.isfinite(price) or price <= 0 or T <= 0:
        return float("nan")
    fwd = S * np.exp((r - q) * T)
    df_ = np.exp(-r * T)
    intr = df_ * (max(fwd - K, 0.0) if call else max(K - fwd, 0.0))
    if price <= intr + 1e-8:
        return float("nan")
    try:
        return float(
            brentq(
                lambda v: bs_european_price(S, K, r, q, v, T, call) - price,
                lo,
                hi,
                xtol=1e-6,
                maxiter=100,
            )
        )
    except (ValueError, RuntimeError):
        return float("nan")


def fetch_spot(ticker: str = "SPY") -> float:
    """Most recent close for ``ticker``."""
    return float(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1])


def fetch_risk_free_rate(
    tenor_years: float = 0.25,
    default: float = 0.043,
) -> float:
    """Risk-free rate from Yahoo Finance Treasury-yield tickers.

    Picks the closest available point (or linearly interpolates between
    bracketing points) on the Treasury yield curve:

        ^IRX  -- 13-week T-bill   (~0.25y)
        ^FVX  -- 5-year T-note    (5y)
        ^TNX  -- 10-year T-note   (10y)
        ^TYX  -- 30-year T-bond   (30y)

    Yahoo quotes these in *percent* (e.g. 4.30 means 4.30%); the helper
    divides by 100.  For ``tenor_years < 0.25`` ^IRX is reused.  For
    ``tenor_years > 30`` ^TYX is reused.

    The returned value is treated as a continuously-compounded rate by
    the rest of the package; the difference vs the bond-equivalent yield
    Yahoo actually quotes is < 10 bp on a 4% rate at 1y maturity --
    acceptable for the SPY-equity scope of this package.

    Falls back to ``default`` if yfinance throws or returns NaN.
    """
    nodes = [(0.25, "^IRX"), (5.0, "^FVX"), (10.0, "^TNX"), (30.0, "^TYX")]

    def _pull(ticker: str) -> Optional[float]:
        try:
            v = float(yf.Ticker(ticker).history(period="5d")["Close"].iloc[-1])
            if np.isfinite(v) and v > 0:
                return v / 100.0  # percent -> fraction
        except Exception:
            pass
        return None

    t = float(tenor_years)
    if t <= nodes[0][0]:
        v = _pull(nodes[0][1])
        return v if v is not None else float(default)
    if t >= nodes[-1][0]:
        v = _pull(nodes[-1][1])
        return v if v is not None else float(default)

    # Bracket and linearly interpolate
    for (t_lo, tk_lo), (t_hi, tk_hi) in zip(nodes, nodes[1:]):
        if t_lo <= t <= t_hi:
            v_lo = _pull(tk_lo)
            v_hi = _pull(tk_hi)
            if v_lo is None or v_hi is None:
                return v_lo if v_lo is not None else (v_hi if v_hi is not None else float(default))
            w = (t - t_lo) / (t_hi - t_lo)
            return float((1 - w) * v_lo + w * v_hi)
    return float(default)


def fetch_dividend_yield(
    ticker: str = "SPY",
    default: float = 0.013,
) -> float:
    """Continuous-equivalent dividend yield for ``ticker`` from yfinance.

    Yahoo's ``Ticker.info`` returns the field as either a fraction (0.013)
    or a percent (1.3) depending on version; this helper normalizes.  Falls
    back to ``default`` if the field is missing or yfinance throws.

    Returns
    -------
    float
        Annualized continuous dividend yield (fraction, e.g. 0.013 = 1.3%).
    """
    try:
        info = yf.Ticker(ticker).info
        for key in ("dividendYield", "trailingAnnualDividendYield", "yield"):
            v = info.get(key)
            if v is None:
                continue
            v = float(v)
            # yfinance is inconsistent: <1 => already a fraction; >=1 => percent.
            return v if v < 1.0 else v / 100.0
    except Exception:
        pass
    return float(default)


def select_expiries(
    ticker: str = "SPY",
    max_expiries: Optional[int] = None,
    min_dte: int = 2,
    max_dte: int = 365 * 7,
) -> List[str]:
    """Return all expiry strings in ``[min_dte, max_dte]`` calendar days.

    Defaults follow the industry-standard "use every expiry within a 7-year
    cutoff" convention.  This keeps the helper neutral: the library does NOT
    decide for you which Yahoo tenors are "noisy" -- that judgment belongs
    in the calling code.

    For Yahoo-quality data on volatile short-dated wings you typically
    want to **tighten** this band, e.g.::

        select_expiries('SPY', min_dte=14, max_dte=365*5)   # drops weeklies + far LEAPS

    or, for weekly products only::

        select_expiries('SPY', min_dte=2,  max_dte=60)

    See :func:`spy_asian_pricer.arbitrage.filter_butterfly_arbitrage` for
    the complementary post-calibration filter that drops slices whose
    SVI fit produced negative-density (catastrophic butterfly arb) -- in
    production this kind of slice is removed manually by traders after
    eyeballing the smile; the helper automates the threshold cut.

    ``max_expiries=None`` returns every expiry in the band; pass an
    integer to evenly sub-sample.
    """
    tkr = yf.Ticker(ticker)
    today = pd.Timestamp.today().normalize()
    expiries = []
    for exp_str in tkr.options:
        dte = (pd.Timestamp(exp_str) - today).days
        if min_dte <= dte <= max_dte:
            expiries.append(exp_str)
    if max_expiries is not None and len(expiries) > max_expiries:
        idx = np.linspace(0, len(expiries) - 1, max_expiries, dtype=int)
        expiries = [expiries[i] for i in idx]
    return expiries


def build_vol_grid(
    ticker: str = "SPY",
    spot: Optional[float] = None,
    r: float = 0.043,
    q: Optional[float] = None,
    expiries: Optional[List[str]] = None,
    moneyness_band: float = 0.3,
    min_volume: int = 0,
    min_strikes_per_slice: int = 5,
    min_dte: int = 2,
    max_dte: int = 365 * 7,
    max_rel_spread: float = 0.5,
    use_mid: bool = True,
) -> Dict[str, "pd.DataFrame"]:
    """Pull option chains and build the per-expiry IV smile grid.

    Forward uses ``spot * exp((r - q) * dcf)``.  When ``q`` is None the
    dividend yield is fetched from yfinance via :func:`fetch_dividend_yield`.

    The ``impliedVolatility`` column is **re-implied locally** from the
    bid/ask **mid** (or ``lastPrice`` fallback when bid/ask are 0/NaN) via
    :func:`implied_vol_from_price`, using the ``r``, ``q`` passed into this
    function.  Yahoo's own ``impliedVolatility`` field is ignored because
    it is unreliable outside US market hours.

    Liquidity / quote-quality filter (when ``use_mid=True``):

      - ``bid > 0``                       (someone willing to buy)
      - ``ask > bid``                     (not a crossed quote)
      - ``(ask - bid) / mid < max_rel_spread``   (default 50%; tighter cuts more wing)
      - ``volume > min_volume`` OR ``openInterest > 0``  (some sign of life)

    With ``use_mid=False`` we fall back to the older ``lastPrice``-only
    pipeline with a flat ``volume > min_volume`` filter (kept for
    reproducibility; not recommended).

    Tenor band ``[min_dte, max_dte]`` is forwarded to
    :func:`select_expiries` when ``expiries`` is None.  Defaults
    ``[2, 365*7]`` match the standard 7-year window used in production.  On Yahoo data
    you typically want to TIGHTEN this -- weekly Yahoo IVs are noisy
    and produce calendar/butterfly arbitrage in the calibrated surface.
    A reasonable "clean Yahoo" preset:
    ``build_vol_grid(..., min_dte=14, max_dte=365*5)``.

    See :func:`spy_asian_pricer.arbitrage.filter_butterfly_arbitrage`
    for the complementary post-calibration filter that automates what
    a trader does manually in production -- drop SVI slices whose fit
    produced a catastrophically negative density.

    Returns ``{expiry_str: DataFrame}`` with columns:
        strike, impliedVolatility, lastPrice, volume, openInterest, optType,
        dte, dcf, fwd, logMoneyness.
    """
    tkr = yf.Ticker(ticker)
    today = pd.Timestamp.today().normalize()
    if spot is None:
        spot = fetch_spot(ticker)
    if q is None:
        q = fetch_dividend_yield(ticker)
    if expiries is None:
        expiries = select_expiries(ticker, min_dte=min_dte, max_dte=max_dte)

    vol_data: Dict[str, pd.DataFrame] = {}
    for exp_str in expiries:
        chain = tkr.option_chain(exp_str)
        cols = [
            "strike", "impliedVolatility", "lastPrice", "bid", "ask",
            "volume", "openInterest",
        ]
        # yfinance sometimes omits bid/ask on illiquid tickers; tolerate.
        avail = [c for c in cols if c in chain.calls.columns]
        calls = chain.calls[avail].copy()
        puts = chain.puts[avail].copy()
        for c in ("bid", "ask"):
            if c not in calls.columns:
                calls[c] = 0.0; puts[c] = 0.0
        calls["optType"] = "C"
        puts["optType"] = "P"

        exp_date = pd.Timestamp(exp_str)
        dte = (exp_date - today).days
        dcf = dte / 365.0
        fwd = spot * np.exp((r - q) * dcf)

        otm_puts = puts[puts["strike"] <= fwd].copy()
        otm_calls = calls[calls["strike"] > fwd].copy()

        # Compute mid (with lastPrice fallback when bid/ask missing) and
        # relative spread for both legs.
        for side_df in (otm_puts, otm_calls):
            bid = side_df["bid"].fillna(0.0).clip(lower=0.0)
            ask = side_df["ask"].fillna(0.0).clip(lower=0.0)
            mid = (bid + ask) / 2.0
            # Where bid/ask absent or crossed, fall back to lastPrice
            valid_quote = (bid > 0) & (ask > bid)
            side_df["mid"] = mid.where(valid_quote, side_df["lastPrice"])
            side_df["rel_spread"] = np.where(
                valid_quote & (mid > 0), (ask - bid) / mid, np.inf
            )
            side_df["price_for_iv"] = side_df["mid"] if use_mid else side_df["lastPrice"]

        # Re-imply IV from chosen quote price.
        otm_puts["impliedVolatility"] = otm_puts.apply(
            lambda x: implied_vol_from_price(
                x["price_for_iv"], spot, x["strike"], r, q, dcf, call=False
            ),
            axis=1,
        )
        otm_calls["impliedVolatility"] = otm_calls.apply(
            lambda x: implied_vol_from_price(
                x["price_for_iv"], spot, x["strike"], r, q, dcf, call=True
            ),
            axis=1,
        )
        df = pd.concat([otm_puts, otm_calls], ignore_index=True)
        df = df.dropna(subset=["impliedVolatility"]).copy()  # drop solver failures

        # IV-range sanity (always)
        mask = (df["impliedVolatility"] > 0.01) & (df["impliedVolatility"] < 2.0)

        if use_mid:
            # Two-tier quote-quality filter so the path is well-defined any
            # time of week:
            #
            #   tier A (preferred, market hours): bid/ask are populated,
            #     bid > 0 AND ask > bid AND (ask-bid)/mid < max_rel_spread.
            #
            #   tier B (fallback, weekends/pre-market when Yahoo bid/ask=0):
            #     accept the strike if lastPrice > 0 -- we already used it
            #     to reverse-imply IV via the mid-fallback path above.
            has_valid_quote = (df["bid"].fillna(0) > 0) & (
                df["ask"].fillna(0) > df["bid"].fillna(0)
            )
            tier_a = has_valid_quote & (
                df["rel_spread"].fillna(np.inf) < max_rel_spread
            )
            tier_b = ~has_valid_quote & (df["lastPrice"].fillna(0) > 0)
            mask &= tier_a | tier_b
            # Liquidity sanity (independent of quote tier).
            mask &= (
                (df["volume"].fillna(0) > min_volume)
                | (df["openInterest"].fillna(0) > 0)
            )
        else:
            # Legacy lastPrice path: flat volume filter only.
            mask &= (df["volume"].fillna(0) > min_volume)

        df = df[mask].copy()

        lo = (1.0 - moneyness_band) * spot
        hi = (1.0 + moneyness_band) * spot
        df = df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()

        if len(df) >= min_strikes_per_slice:
            df["dte"] = dte
            df["dcf"] = dcf
            df["fwd"] = fwd
            df["logMoneyness"] = np.log(df["strike"] / fwd)
            df = df.sort_values("strike").reset_index(drop=True)
            vol_data[exp_str] = df

    if len(vol_data) < 2:
        warnings.warn(
            f"build_vol_grid: only {len(vol_data)} expiry slice(s) survived "
            f"the filters (min_volume={min_volume}, "
            f"min_strikes_per_slice={min_strikes_per_slice}, "
            f"moneyness_band={moneyness_band}). JWSVIVolSurface and "
            "DupireLocalVol both require >=2 tenors. Try lowering "
            "min_strikes_per_slice, widening moneyness_band, or passing "
            "`expiries=` manually.  The min_volume=0 default already keeps "
            "any strike that has ever traded recently; if that is still "
            "empty, Yahoo is returning a degenerate chain (try a different "
            "ticker or wait for market hours).",
            RuntimeWarning,
            stacklevel=2,
        )
    return vol_data
