"""spy-asian-pricer: Asian options on SPY under Dupire local volatility.

Pipeline
--------
1. Fetch SPY option chains (yfinance, optional extra)
2. Per-expiry SVI calibration -> raw 5-parameter slice
3. SVI -> JWSVI conversion + time interpolation (wing-derived nu_tilda)
4. Dupire local vol via cubic-spline dw/dT + Gatheral denominator
5. Monte Carlo pricing (Euler-Maruyama, antithetic)
6. Greeks via finite-difference repricing under common random numbers
"""

from .arbitrage import (
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    check_spread_arbitrage,
    filter_butterfly_arbitrage,
)
from .dupire import DupireLocalVol
from .mc import (
    AsianMCPricer,
    compute_greeks,
    geometric_asian_call_price,
    trading_day_obs_dcfs,
)
from .ssvi import SSVISurface, calibrate_ssvi
from .surface import JWSVIVolSurface
from .svi import JWSVIParam, SVIParam, calibrate_svi

# data.py is optional (yfinance / pandas extras); expose helpers when present.
try:
    from .data import (  # noqa: F401  -- re-exported
        bs_european_price,
        build_vol_grid,
        fetch_dividend_yield,
        fetch_risk_free_rate,
        fetch_spot,
        implied_vol_from_price,
        select_expiries,
    )
    _DATA_OK = True
except ImportError:
    _DATA_OK = False

__all__ = [
    "SVIParam",
    "JWSVIParam",
    "calibrate_svi",
    "JWSVIVolSurface",
    "SSVISurface",
    "calibrate_ssvi",
    "DupireLocalVol",
    "AsianMCPricer",
    "geometric_asian_call_price",
    "compute_greeks",
    "trading_day_obs_dcfs",
    "check_butterfly_arbitrage",
    "check_calendar_arbitrage",
    "check_spread_arbitrage",
    "filter_butterfly_arbitrage",
]
if _DATA_OK:
    __all__ += [
        "bs_european_price",
        "build_vol_grid",
        "fetch_dividend_yield",
        "fetch_risk_free_rate",
        "fetch_spot",
        "implied_vol_from_price",
        "select_expiries",
    ]

__version__ = "0.2.1"
