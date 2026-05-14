# spy-asian-pricer

Arithmetic-average Asian option pricing on SPY under **Dupire local volatility**, calibrated from an **SVI / JWSVI** implied vol surface fit to live Yahoo Finance option chains.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/linhzhang217/asian_option_pricer/blob/main/notebooks/demo.ipynb)

---

## What problem does this solve?

Pricing path-dependent payoffs on a real index requires more than Black-Scholes — you need a vol surface that's smooth in both strike and time, free of static arbitrage, and consistent with every market-quoted vanilla option. This package handles the full pipeline end-to-end: pull SPY option chains from Yahoo Finance, fit an SVI smile per expiry, interpolate across maturities via JWSVI, derive the Dupire local volatility surface, and price arithmetic-average Asian options by Monte Carlo with antithetic variates.

```
Yahoo Finance option chains
   -> SVI calibration (per expiry slice)
   -> SVI -> JWSVI conversion
   -> JWSVI time interpolation (wing-derived nu_tilda)
   -> Arbitrage diagnostics (butterfly / calendar / spread)
   -> Dupire local vol  (cubic-spline dw/dT, Gatheral denominator)
   -> Monte Carlo  (Euler-Maruyama + antithetic variates)
   -> Asian price, Greeks, convergence diagnostics
```

---

## Background

### What is an Asian option?

A standard ("European") call pays $\max(S_T - K, 0)$ at maturity — its value depends only on the spot price on one date. An **arithmetic-average Asian call** pays

$$\max\!\Bigl(\bar{A} - K, 0\Bigr), \qquad \bar{A} = \tfrac{1}{N}\sum_{i=1}^{N} S_{t_i}$$

i.e. it depends on the **average** of $N$ prices over the option's lifetime. Averaging dampens terminal-price noise, so Asian options are cheaper than their European counterparts and are widely used to hedge cost-averaged exposures (energy, FX, commodity flows). The catch: the distribution of an arithmetic average of lognormals has no closed form, so pricing requires numerical methods.

### Why local volatility instead of constant volatility?

Black-Scholes assumes a single constant $\sigma$. The market disagrees: every strike–maturity pair has its own implied volatility, and that surface has skew and term structure. **Dupire (1994)** showed that there exists a **deterministic** local volatility function $\sigma_{\text{loc}}(S, t)$ such that the diffusion

$$dS_t = (r - q)\, S_t\, dt + \sigma_{\text{loc}}(S_t, t)\, S_t\, dW_t$$

reproduces *every* observed European option price simultaneously, given by

$$\sigma_{\text{loc}}^2(K, T) = \frac{\partial_T C(K, T)}{\tfrac{1}{2} K^2\, \partial_{KK} C(K, T)}$$

Once we have $\sigma_{\text{loc}}$, we can price **any** payoff on $S$ — including path-dependent ones like Asians — consistently with the market's vanilla quotes.

### Why SVI for the smile?

We need a parametric form for the implied vol smile that (a) fits market quotes well, (b) has smooth $\partial_K$ and $\partial_{KK}$ (needed for Dupire's denominator), and (c) is parsimonious. **Gatheral (2004) raw SVI** parameterizes total variance $w = \sigma_{\mathrm{iv}}^2 T$ as a 5-parameter function of log-forward-moneyness $y = \ln(K/F)$:

$$w(y) = a + b\,\Bigl(\rho\,(y - m) + \sqrt{(y - m)^2 + \sigma^2}\Bigr)$$

The five parameters $(a, b, \rho, m, \sigma)$ control the level, slope, asymmetry, location and convexity of the smile. SVI fits real equity-index smiles to within a few basis points and admits closed-form first and second derivatives — exactly what Dupire needs.

### Why JWSVI for time interpolation?

Raw SVI parameters $(a, b, \rho, m, \sigma)$ have no direct financial interpretation, so interpolating them across maturities is unstable. **Gatheral & Jacquier (2014)** introduced the **Jump-Wing** reparameterization (JWSVI), a mathematically equivalent 6-tuple $(\nu, \phi, p, c, \tilde\nu, \mathrm{conv})$ with clear meaning:

| Param | Meaning |
|---|---|
| $\nu$ | ATM total variance per unit time |
| $\phi$ | ATM skew |
| $p$ | put-wing slope (left tail) |
| $c$ | call-wing slope (right tail) |
| $\tilde\nu$ | minimum total variance (smile floor) |
| $\mathrm{conv}$ | convexity |

Because each parameter is a *financial* quantity, smooth time interpolation in JWSVI space produces a sensible, calendar-arbitrage-free surface even with sparse expiry coverage. We interpolate $(\nu T, \phi, p, c)$ linearly (cubic if $>3$ slices), and $\tilde\nu$ is re-derived from the wing slopes via $\tilde\nu = 4\nu p c / (p+c)^2$, then clipped to $[10^{-6},\, 0.99\,\nu]$ (the `NUTILDA_FLOOR` constant) so the SVI radius stays well-defined and $\nu - \tilde\nu \ge 0$ always.

### Why SSVI for joint surface fitting?

JWSVI fits each expiry independently and only checks calendar arbitrage *after the fact*. On Bloomberg-quality data with trader overrides this is fine, but on free-feed Yahoo chains adjacent slices routinely disagree enough that calendar arb leaks into the surface — and once that happens the Dupire numerator $\partial_T w$ goes negative and the local-vol grid has to clamp itself out of the corner. **Gatheral (2014) SSVI** removes the failure mode at the source by sharing skew/curvature across every tenor: only the ATM term structure $\theta(t)$ varies tenor-by-tenor, and three global parameters describe the smile shape everywhere.

$$w(k, t) = \tfrac{\theta(t)}{2}\,\Bigl(1 + \rho\,\phi(\theta) k + \sqrt{(\phi(\theta) k + \rho)^2 + 1 - \rho^2}\Bigr),\quad \phi(\theta) = \eta\,\theta^{-\gamma}$$

| Param | Meaning |
|---|---|
| $\eta$ | skew curvature scale (overall smile width) |
| $\rho$ | global spot-vol correlation; equity skew is negative ($\rho < 0$ at every tenor) |
| $\gamma \in [0, 1/2]$ | term-structure decay exponent on $\phi(\theta) = \eta\,\theta^{-\gamma}$; controls how the smile flattens with maturity |
| $\theta(t)$ | ATM total variance term structure (one knot per calibrated tenor) |

The surface is **calendar-arbitrage-free by construction** when $\theta(t)$ is monotone non-decreasing and $\gamma \in [0, 1/2]$ — both enforced by the calibrator. Total parameter count is $3 + N$ instead of JWSVI's $5 \times N$, and the $3$ globals give the optimiser strong noise rejection: a single bad call-wing quote on one slice can no longer flip the local skew sign because $\rho$ is shared across every tenor.

At each fixed tenor SSVI reduces in *closed form* to a raw SVI slice (Gatheral & Jacquier 2014, Theorem 4.1), so an `SSVISurface` exposes the same `get_svi_at(dcf)` API as `JWSVIVolSurface` and plugs into `DupireLocalVol` unchanged.

When to pick which:

| Situation | Pick |
|---|---|
| Clean Bloomberg / vendor data, pricing per-tenor European vanillas | per-slice JWSVI (preserves per-tenor flexibility, vega-bucket-friendly) |
| Noisy Yahoo data, want surface clean by construction | SSVI `full` |
| Want to reproduce a JWSVI surface as a baseline before SSVI | SSVI `pinned` (matches per-slice ATM exactly) |
| Path-dependent products (Asian, barrier) on noisy data | SSVI `full` (calendar-arb-free → cleaner Dupire) |

### Why three arbitrage checks?

A vol surface that violates static arbitrage allows a model-free strategy with positive expected payoff and zero cost — pricing on top of it is meaningless. We check all three classical conditions:

1. **Butterfly** — Gatheral's density discriminant $g(y) \ge 0$ for every slice (the implied risk-neutral PDF must be non-negative).
2. **Calendar** — total variance $w(K, T)$ is non-decreasing in $T$ for every strike.
3. **Spread** — call prices are monotone in $K$ with slope bounded below by $-e^{-rT}$ (no two-call portfolio dominates a third).

### Why Monte Carlo with antithetic variates?

Under local vol, the spot path has no closed-form distribution, so we discretize the SDE (Euler-Maruyama on log-price) and average the discounted payoff over many simulated paths. Plain MC has standard error $O(1/\sqrt{n})$. We use **antithetic variates** — for each Brownian draw $Z$ we also use $-Z$, halving the noise from symmetric paths essentially for free.

> **Why no control variate?** A natural CV is the geometric-Asian Kemna–Vorst (1990) closed form. It works perfectly under flat vol, but on a skewed local-vol surface the K-V target (which assumes GBM with a single flat ATM vol) systematically misses the true expectation of the geometric leg by several percent — a bias far larger than the MC error it removes. Recentering the target with an independent pilot MC introduces enough pilot variance to dominate the variance gain, making the recentered estimator worse than no-CV at the same compute budget. We therefore drop CV and rely on antithetic + path count. The closed form is still exposed as `geometric_asian_call_price` for flat-vol benchmarking.

---

## How it works

Each pipeline stage maps 1-to-1 onto a module in `spy_asian_pricer/`. Below is what each does and the key implementation choices.

### 1. Data ingestion (`data.py`)

`build_vol_grid()` pulls SPY chains from Yahoo Finance, selects expiries within `[min_dte, max_dte]` calendar days (default `[2, 365*7]` — standard "7-year cutoff" convention used in production; the library does NOT decide for you which Yahoo tenors are too noisy, see "Production-style cleanup" below), and constructs the per-expiry IV grid by:

- **Forward with dividends**: $F = S\, e^{(r-q) T}$. The continuous-equivalent dividend yield $q$ is pulled from `Ticker.info` via `fetch_dividend_yield()` (yfinance returns it as either a fraction or a percent depending on version; the helper normalizes). Default fallback is 1.3% (SPY).
- **OTM-only filter**: puts with $K \le F$ + calls with $K > F$. ITM IVs are noisy (small extrinsic value).
- **Re-imply IV from bid/ask mid** (with `lastPrice` fallback): Yahoo's own `impliedVolatility` column is **discarded** and replaced via `implied_vol_from_price()` using the same $r, q$ that the rest of the pipeline uses. Mid quotes are fresher than last trades — for an illiquid strike the last print can be days old while the bid/ask snapshot is at most a 15-min delayed NBBO. Where bid or ask are absent (weekends / pre-market) the helper falls back to `lastPrice`.
- **Two-tier quote-quality filter** (default; `use_mid=True`).  Yahoo's bid/ask snapshot disappears outside US market hours, so we use a soft fallback:

  - **Tier A** (preferred — market hours): `bid > 0` AND `ask > bid` AND `(ask - bid)/mid < max_rel_spread` (default 50%).  Wing strikes with stale wide quotes get dropped.
  - **Tier B** (fallback — weekend / pre-market when bid/ask are 0/NaN): accept the strike if `lastPrice > 0`.  We already used `lastPrice` to reverse-imply the IV via the mid-fallback path, so any strike that survived IV-reversal has a usable price.
  - **Liquidity sanity** (independent of which tier): `volume > 0` OR `openInterest > 0` (some sign of life on this strike).

  Net behaviour: market hours runs the strict spread filter that drops stale wing data; weekends fall back to a lastPrice-only path that matches the old behaviour.  Pass `use_mid=False` to disable both tiers and force the legacy `lastPrice`-only flow with a flat `volume > min_volume` filter; not recommended.
- **Moneyness band**: keep only $K \in [0.7\, S, 1.3\, S]$ to avoid deep-tail garbage.
- **Per-slice minimum**: discard any expiry with fewer than 5 surviving strikes.

Output: a `dict` of DataFrames keyed by expiry string, each carrying `strike`, `impliedVolatility` (re-implied), `dcf`, `fwd`, `logMoneyness`.

### 2. SVI calibration (`svi.py`)

For each slice we solve

$$\min_{a, b, \rho, m, \sigma} \sum_i w_i \bigl[w_{\mathrm{SVI}}(y_i) - \sigma_{\mathrm{iv}, i}^2 T\bigr]^2$$

via `scipy.optimize.least_squares` with the **TRF** (trust-region reflective) algorithm and parameter bounds $b \ge 0$, $|\rho| < 1$, $\sigma > 0$. Fit weights $w_i = e^{-y_i^2 / 2 \cdot 0.3^2}$ are vega-like — they emphasize ATM where the smile is best-quoted. Each slice fits in milliseconds.

### 3. SVI → JWSVI conversion (`svi.py`)

Closed-form transformation, no fitting. The 5 raw SVI parameters map to the 6 JWSVI parameters $(\nu, \phi, p, c, \tilde\nu, \mathrm{conv})$ via the formulas in Gatheral & Jacquier (2014). The inverse `JWSVIParam.to_svi()` round-trips with floating-point error.

### 4. Time interpolation (`surface.py`)

`JWSVIVolSurface` interpolates each JWSVI parameter across tenors and re-derives $\tilde\nu$ from the wing slopes at every target tenor:

- $(\nu T, \phi, p, c)$ are interpolated **linearly** when $\le 3$ tenors, **cubic** otherwise. Linear keeps $\nu T$ monotone non-decreasing, which is necessary for calendar-arbitrage-free reconstruction.
- $\tilde\nu$ is **re-derived** from wing slopes at every target tenor: $\tilde\nu = 4 \nu p c / (p + c)^2$, then **clipped to $[10^{-6},\, 0.99\,\nu]$ (the `NUTILDA_FLOOR` constant)** so the SVI radius reconstruction in `to_svi` always sees $\nu - \tilde\nu > 0$.
- `conv` is recomputed for diagnostics but does NOT enter the SVI reconstruction.
- The forward used by `implied_vol` / `total_variance` is $F = S\, e^{(r-q) T}$ (the dividend yield is stored on the surface).
- This re-derivation makes the surface **lossy at calibrated knots**: feeding `svi_orig.to_jwsvi(t)` into the surface and reading it back via `get_svi_at(t)` does NOT round-trip exactly. The notebook ships a "surface stability check" cell that quantifies this gap on a sensible moneyness band ($|y|\le 0.1$) split by tenor bucket.

### 4b. SSVI joint calibration (`ssvi.py`)

`calibrate_ssvi` is an alternative to the per-slice SVI → JWSVI → time-interpolate path of §2-§4. Instead of fitting each slice independently and then interpolating, it solves *one* joint least-squares problem over the entire `(K, T)` cloud:

$$\min_{\eta,\, \rho,\, \gamma,\, \theta(\cdot)}\; \sum_{i, j} w_{ij}\,\bigl[\, w_{\text{SSVI}}(k_{ij}, \theta(t_j);\; \eta, \rho, \gamma) \;-\; \sigma_{\text{iv},\,ij}^2\, t_j \,\bigr]^2$$

with the same vega-like Gaussian weights $w_{ij} = \exp(-k_{ij}^2 / (2 \cdot 0.3^2))$ as the per-slice SVI fit. Implementation choices:

- **Two modes** controlled by `mode=`:
  - **`'pinned'` (3 free params, default)**: $\theta(t_i)$ is pinned to each slice's market ATM total variance ($\theta_i := \sigma_{\text{ATM},\,i}^2 \cdot t_i$, computed by linearly interpolating the input IV grid at the forward); only $(\eta, \rho, \gamma)$ are optimised. Fast (3 unknowns, L-BFGS-B converges in a few iterations) and matches per-slice ATM exactly, but inherits any noise in the ATM term structure.
  - **`'full'` (3 + N free params)**: $\theta_1, \ldots, \theta_N$ are optimised jointly with $(\eta, \rho, \gamma)$.
- **Monotone $\theta(t)$ via cumsum-of-squares**. Calendar-arb-free SSVI requires $\theta_{i+1} \ge \theta_i$. Rather than feeding L-BFGS-B a constrained problem, we reparameterise $\theta_i = \sum_{j \le i} d_j^2$ with the unbounded variables $d_j$ — squaring kills the sign and cumsum kills the monotonicity constraint, so the optimiser sees a fully unconstrained problem in the $d_j$. Initial guess is $d_j^{(0)} = \sqrt{\theta^{\text{ATM}}_j - \theta^{\text{ATM}}_{j-1}}$ from the pinned-mode init.
- **Bounds** $\eta \in [0.01, 50]$, $\rho \in (-0.999, 0.999)$, $\gamma \in [0, 0.499]$ enforce calendar-arbitrage-free structure (the upper bound on $\gamma$ is the strict version of $\le 1/2$).
- **Closed-form SSVI → SVI mapping at fixed $t$** (Gatheral & Jacquier 2014, Thm 4.1): $a = \tfrac{\theta(1-\rho^2)}{2}$, $b = \tfrac{\theta\,\phi}{2}$, $\rho_{\text{SVI}} = \rho$, $m = -\rho/\phi$, $\sigma = \sqrt{1 - \rho^2}/\phi$. This is what `get_svi_at(dcf)` returns, so `SSVISurface` plugs into `DupireLocalVol` via the same duck-typed contract as `JWSVIVolSurface`.
- **`theta_at(dcf)`** linearly interpolates between the calibrated $\theta_i$ knots; below the first knot it clips at $\theta_1$ (no extrapolation toward zero where $\phi(\theta) = \eta\,\theta^{-\gamma}$ blows up).

### 5. Dupire local volatility (`dupire.py`)

The numerator of Dupire's formula is $\partial_T w$. For each of 160 evenly-spaced strikes on $K \in [0.5\, S_0,\, 1.5\, S_0]$, we collect total variance at every benchmark tenor and fit a **natural cubic spline in $T$** with `scipy.interpolate.CubicSpline`. The derivative is then the spline's analytic derivative — no finite differences, no noise amplification. Forwards inside the grid use $F = S_0\, e^{(r-q) T}$; the time grid runs from the shortest calibrated tenor to the longest (no extrapolation below the first benchmark).

The denominator uses Gatheral's butterfly formula:

$$\mathrm{denom}(K, T) = \Bigl(1 - \tfrac{y w'}{2 w}\Bigr)^2 - \tfrac{(w')^2}{4}\Bigl(\tfrac{1}{w} + \tfrac{1}{4}\Bigr) + \tfrac{w''}{2}$$

with $w', w''$ from SVI's closed-form derivatives. To handle numerical edge cases at extreme strikes / short maturities, we apply **four safety clamps**:

| Clamp | Constant | Purpose |
|---|---|---|
| `DWDT_FLOOR` | `1e-4` | Prevent negative/zero numerator from noisy spline |
| `DENOM1_FLOOR` | `0.2` | Prevent denominator collapse near deep wings |
| `DENS_RATIO_CAP` | `5.0` | Cap density-correction ratio (was `0.75`, which silently biased convex short-dated smiles low; the higher cap still prevents pathological blow-ups while letting legitimate density show through) |
| `LV_OVER_ATM_CAP` | `10.0` | Cap local vol at $10 \times$ ATM IV |

Hit rates for each clamp are recorded in `DupireLocalVol.clamp_stats` — if any one fires on $> 5\%$ of the grid, your input data is probably the problem.

The final $(T, K)$ grid is wrapped in a `RectBivariateSpline` for fast vectorized lookup inside the MC loop. `local_vol` and `local_vol_vec` floor any negative spline-overshoot at zero before returning.

### 6. Monte Carlo pricing (`mc.py`)

**Path simulation.** Euler-Maruyama on log-price:

$$S_{t + \Delta t} = S_t \exp\!\Bigl(\bigl(r - q - \tfrac{1}{2}\sigma_{\mathrm{loc}}^2(S_t, t)\bigr) \Delta t + \sigma_{\mathrm{loc}}(S_t, t)\sqrt{\Delta t}\, Z\Bigr)$$

We sample $\sigma_{\mathrm{loc}}$ at the **midpoint** $t + \Delta t/2$ for $O(\Delta t^2)$ weak-error improvement. The averaging grid (`n_obs` dates) and simulation grid (`n_steps_per_obs` Euler sub-steps) are **decoupled** so coarse averaging (e.g. monthly) keeps a fine Euler step.

**Observation schedule (trading-day default).** By default `AsianMCPricer` puts the `n_obs` averaging dates on the next `n_obs` weekdays after `start_date` (today by default), so consecutive intervals are 1 calendar day on weekday-to-weekday and 3 calendar days across a weekend. The Euler-Maruyama scheme uses the per-interval $\Delta t$, so a Friday-to-Monday gap is integrated with $\sqrt{3\,\Delta t_{\text{wd}}}$ Brownian increment — the correct scaling — instead of being mis-treated as a regular day. Holidays are *not* skipped (no built-in market calendar); for an exact NYSE schedule including holidays, build the dcfs externally (`pandas_market_calendars` or similar) and pass `obs_dcfs=` directly. In trading mode the user-supplied `T` is informational only; the effective maturity is taken from the last observation (e.g. 126 trading days from today on a Wednesday lands at $T \approx 176/365 \approx 0.482$, not exactly $0.5$). Pass `obs_schedule='calendar'` to recover legacy uniform-in-T spacing — required when comparing to the Kemna-Vorst closed form, which assumes uniform $t_i = i\,T/N$.

**Antithetic variates.** For each Brownian draw $Z$ we also use $-Z$, doubling sample size at no extra cost and cancelling first-order symmetric noise.

**No control variate.** See "Why no control variate?" above. The estimator is the plain antithetic average of discounted arithmetic payoffs:

$$\widehat{C}_{\mathrm{arith}} = \frac{1}{n}\sum_{i=1}^{n} e^{-rT}\,\bigl(\bar{A}^{(i)} - K\bigr)^+$$

with $n$ split half/half across $Z$ and $-Z$. Standard error is $O(1/\sqrt{n})$; bump `n_paths` if you need tighter CIs.

**Greeks.** `compute_greeks()` runs central finite differences with **common random numbers** (`np.random.seed(42)` reset before each bump), so simulation noise cancels between $S \pm \delta S$ pairs. Vega applies an **absolute parallel shift** (`vol_bump_abs`) to the local-vol diffusion. Pass `call=False` for put greeks (delta < 0, etc.); the small-$T$ intrinsic fallback uses the forward-based payoff for the requested option type.

---

## What can it price?

### Why SPY (and not single stocks)?

This package is built and tested on SPY because it's uniquely suited to a **fully-automated pipeline running off Yahoo Finance** — i.e. retail-quality data with no trader curation:

- **Liquidity**: the most-traded equity options market in the world (~3M contracts/day), with tight bid-ask, hundreds of strikes per expiry, and minute-fresh `lastPrice` even on a free feed.
- **Smooth smile**: a basket of 500 stocks averages away idiosyncratic noise, so the empirical IV surface is statistically smooth — the regime where SVI's parametric form is a good fit.
- **No event risk**: no earnings, no M&A, no dividend surprises beyond a small continuous yield.  Each expiry's smile is pure index vol, not company-specific shocks.

Run the same pipeline on a single name like **AAPL or TSLA** and the diagnostics fall apart — typically: catastrophic per-slice SVI fits ($g_{\min} \ll -10$), 40–70% butterfly arb failure rate, $20$+ vol-point IV errors on wings, and a Dupire surface where >50% of grid points hit the safety clamps.  Single stocks need **Bloomberg-quality bid/ask + manual trader mark-overrides** to feed a stable surface — neither of which a free-data demo package can provide.  Liquid index ETFs (QQQ, IWM, EFA) sit somewhere in between SPY and single stocks and may work with parameter tightening.

In short: **SPY is the "easy mode" where the methodology is what's being tested, not the data quality**.

### Products supported

| Capability | Yes | No |
|---|---|---|
| Arithmetic-average Asian, **call** | ✅ `call=True` | |
| Arithmetic-average Asian, **put** | ✅ `call=False` | |
| Discrete monitoring (any frequency) | ✅ via `n_obs` | |
| European vanilla | ✅ as a 1-obs Asian | |
| Continuous-monitoring Asian | ⚠️ approximate via large `n_obs` | exact CV not implemented |
| American / Bermudan Asian | | ❌ no early exercise |
| Basket / rainbow / spread Asian | | ❌ single-asset only |

### Strike range

The Dupire surface is built on $K \in [0.5\, S_0,\, 1.5\, S_0]$ (160 grid points). Pricing at strikes in this band is reliable; outside, the local-vol grid extrapolates and accuracy degrades. Recommended: $|K/S_0 - 1| \le 0.4$. Calling `local_vol(S, dcf)` outside the grid emits a `RuntimeWarning`; the MC inner loop (`local_vol_vec`) silently clips to the boundary instead of warning per Euler step.

### Maturity range

Lower bound: shortest calibrated tenor (default `min_dte=2`, often `min_dte=14` in practice on Yahoo).  Upper bound: longest available expiry (default `max_dte=365*7`, ~2–3 years effective on SPY chains).  Pricing past the longest tenor is extrapolation and not advised.

### Underlying

`SPY` is the **tested and recommended** ticker.  Other liquid index ETFs (`QQQ`, `IWM`, `EFA`) work with the same defaults but should be sanity-checked via `clamp_stats` and `check_*_arbitrage`.  Single stocks (`AAPL`, `TSLA`, `NVDA`, etc.) are accepted by `data.py` but **will routinely fail butterfly / calendar arb** under the default thresholds — see "Why SPY" above.  If you need single stocks, expect to (a) tighten `min_dte` to drop earnings-distorted weeklies, (b) tighten `filter_butterfly_arbitrage` threshold to `-0.01`, (c) prefer SSVI over JWSVI for its joint denoising, and (d) accept that some expiries will be unfittable without a manual mark.

**Sensitivity outputs**

| Greek | Method |
|---|---|
| Delta | Central FD, $\pm 1\%$ spot bump |
| Gamma | Central FD, second-order $\pm 1\%$ spot bump |
| Vega  | Central FD, $\pm 1$ vol point (absolute, e.g. 20% → 21%) |
| Theta | One-sided, $-1$ calendar day |

All under common random numbers.

---

## Limitations & assumptions

We are explicit about what this package does NOT model so you can decide whether it fits your use case.

### Model assumptions

- **Pure local volatility.** The diffusion is $dS = (r - q) S\,dt + \sigma_{\mathrm{loc}}(S, t) S\, dW$. There is no stochastic vol component (no Heston, SABR, rough vol), no vol-of-vol, no jumps, no leverage effect beyond what's baked into the surface. For long-dated forward-skew-sensitive payoffs this matters — local vol is well-known to under-price forward-start volatility.
- **Constant (single-tenor) risk-free rate.** $r$ is fetched once via `fetch_risk_free_rate(tenor_years=...)` from Yahoo's Treasury curve (`^IRX` / `^FVX` / `^TNX` / `^TYX`) at the option's tenor, then held constant through the simulation. No term structure inside the SDE, no stochastic rates. Fine for short-to-medium SPY tenors; for multi-year LEAPS or tenors that straddle large rate moves you'd want a full curve.
- **Dividends as a continuous yield.** $q$ is fetched from `Ticker.info['dividendYield']` (or `trailingAnnualDividendYield`) via `fetch_dividend_yield()`, fallback 1.3% for SPY. Discrete dividends, ex-div jumps, and irregular payment schedules are not modeled — the continuous-yield approximation is good enough for index ETFs but biased for single names with concentrated payouts.
- **Last-price IV (not mid).** We re-imply IV from Yahoo's `lastPrice` (last trade) via `implied_vol_from_price()`, NOT from bid/ask mid. The `lastPrice` is fresher than Yahoo's own `impliedVolatility` column outside market hours, but for illiquid strikes it can still be stale by hours or days, so the calibrated surface reflects the most recent traded prints rather than live mid quotes. yfinance does not expose reliable bid/ask, so a true mid-IV path would require a different data source (Polygon, Tradier, IB).

### Numerical caveats

- **Dupire safety clamps mask data quality.** The four clamps (see `DupireLocalVol.clamp_stats`) prevent crashes when the input surface has noisy second derivatives, but they also hide problems. If `denom1_floor_pct > 5%` or `lv_cap_pct > 5%`, the local vol surface is not trustworthy in those regions — investigate the input IVs.
- **Antithetic only — no CV.** A geometric-Asian K-V control variate was tried and removed (bias under skew, see "Why no control variate?" above). If you need tighter CIs, scale `n_paths`. For a step up, dropping in Sobol/QMC paths in `simulate()` typically buys another order-of-magnitude variance reduction without bias risk; not implemented here.
- **Discretization.** Euler-Maruyama is weak order 1; for very long maturities or high-vol regimes consider increasing `n_steps_per_obs`. Default `n_steps_per_obs=1` is appropriate for daily-or-finer averaging.
- **Throughput.** 200k paths × 126 obs × 1 sub-step takes ~5-10 seconds on a modern laptop. PDE methods would be faster for vanilla payoffs but lose generality for path-dependence.

### Data caveats

- **Yahoo Finance reliability.** Free data — stale quotes off-hours, missing volumes on illiquid strikes, occasional NaN IVs. The package filters but does not repair bad quotes.
- **Why arbitrage violations appear on raw market data.** Yahoo Finance provides raw mid-quote IVs without the cleaning, smoothing, or manual marking that occurs on a trading desk. Real vol surfaces in production are the result of (1) multi-source quote validation, (2) bid-ask spread + volume filtering, (3) parametric smoothing (SVI/SSVI), and (4) trader manual override of anomalous quotes — especially on short-dated and far-OTM strikes. Without these layers, raw Yahoo IVs will produce butterfly/calendar arbitrage violations on noisy slices. This is a feature, not a bug: our `check_*_arbitrage()` diagnostics surface exactly where the data fails, allowing the user to filter (e.g. `min_dte=14`) or apply additional smoothing as needed.

#### Production-style cleanup

In a sell-side bank or HF the workflow after each per-slice SVI fit is:

1. The trader (or a junior quant on the surface team) eyeballs every smile.
2. Any slice whose density discriminant $g(y)$ is catastrophically negative gets pulled out of the calibration set — either re-fit with tighter bounds / different weights, or marked by hand from a parallel bid/ask source.
3. Only the cleaned set goes into the time-interpolated surface (here JWSVI).

This package automates step 2 with `filter_butterfly_arbitrage(svi_slices, threshold=-0.05)`:

```python
from spy_asian_pricer import calibrate_svi, filter_butterfly_arbitrage, JWSVIVolSurface

svi_slices = {exp: (calibrate_svi(...), dcf) for ...}
svi_clean, dropped = filter_butterfly_arbitrage(svi_slices, threshold=-0.05)
jwsvi_slices = {k: (svi.to_jwsvi(dcf), dcf) for k, (svi, dcf) in svi_clean.items()}
surface = JWSVIVolSurface(jwsvi_slices, spot=spot, r=r, q=q)
```

Default threshold `-0.05` is conservative (only catastrophic fits). Tighten to `-0.01` for a stricter cut; loosen to `-0.5` to only catch the absolute worst. The JWSVI time interpolator bridges the dropped tenors smoothly using the surviving neighbouring knots.
- **No SSVI smoothing.** We fit raw SVI per slice independently; we do not enforce calendar-arbitrage-free interpolation in calibration (only check it after the fact). For most SPY surfaces this is fine; for very steep skew or short tenors butterfly violations can occur — `check_butterfly_arbitrage()` will tell you.
- **Single-snapshot.** Each call to `build_vol_grid()` is one point in time. No historical surface storage.

### Scope

- **European-style only.** No early exercise.
- **Single-asset.** No basket, spread, rainbow, or quanto Asians.
- **Vanilla averaging.** Equal-weighted arithmetic mean. Default schedule is the next `n_obs` weekdays after the valuation date (1 calendar day apart on weekdays, 3 across weekends); pass `obs_schedule='calendar'` for uniform-in-T spacing, or `obs_dcfs=` for a custom (e.g. holiday-aware) schedule. No weighted averaging, no float-strike, no in-progress (already-observed) paths.

---

## Installation

```bash
pip install spy-asian-pricer                   # core (numpy, scipy)
pip install "spy-asian-pricer[data]"           # + yfinance, pandas (chain fetching)
pip install "spy-asian-pricer[plot]"           # + matplotlib (notebook charts)
pip install "spy-asian-pricer[data,plot]"     # full — recommended for the demo notebook
```

> **Note on quoting**: the brackets in `[data,plot]` are glob characters in zsh (macOS default since 10.15) and will produce `no matches found` if unquoted.  Quoting works in both `bash` and `zsh`, so always wrap the package spec in double quotes when using extras.

---

## Quick start

```python
import numpy as np
from spy_asian_pricer import (
    calibrate_svi, JWSVIVolSurface, DupireLocalVol, AsianMCPricer,
    fetch_spot, fetch_dividend_yield, fetch_risk_free_rate, build_vol_grid,
)

spot = fetch_spot("SPY")
r = fetch_risk_free_rate(tenor_years=0.5)  # Yahoo ^IRX/^FVX interpolation
q = fetch_dividend_yield("SPY")            # ~1.3% for SPY (auto-normalized)
vol_data = build_vol_grid("SPY", spot=spot, r=r, q=q)

# Per-expiry SVI -> JWSVI -> surface
jwsvi_slices = {}
for exp_str, df in vol_data.items():
    svi = calibrate_svi(df["logMoneyness"].values,
                        df["impliedVolatility"].values,
                        df["dcf"].iloc[0])
    jwsvi_slices[exp_str] = (svi.to_jwsvi(df["dcf"].iloc[0]),
                             df["dcf"].iloc[0])

surface = JWSVIVolSurface(jwsvi_slices, spot=spot, r=r, q=q)
local_vol = DupireLocalVol(surface)        # inherits r, q

# Price a 6-month ATM Asian call with daily averaging.
# Default obs_schedule='trading' puts the 126 observation dates on the
# next 126 weekdays from today; the effective maturity becomes whatever
# calendar dcf the 126th business day lands at (~0.48-0.50 depending on
# the day of week today happens to be).  Pass obs_schedule='calendar'
# to force exact T=0.5 and uniform spacing.
T, n_obs = 0.5, 126
pricer = AsianMCPricer(S0=spot, r=r, T=T, n_obs=n_obs,
                       vol_surface=surface, local_vol_surface=local_vol)  # q from surface
np.random.seed(42)
res_call = pricer.price_asian(K=spot, n_paths=200_000)
np.random.seed(42)
res_put  = pricer.price_asian(K=spot, n_paths=200_000, call=False)
print(f"Asian call: ${res_call['price']:.4f}  +/- ${res_call['std_err']:.4f}")
print(f"Asian put : ${res_put ['price']:.4f}  +/- ${res_put ['std_err']:.4f}")
```

### Compute Greeks (call or put)

```python
from spy_asian_pricer import compute_greeks
g_call = compute_greeks(spot, K=spot, r=r, T=T, n_obs=n_obs,
                        vol_surface=surface, local_vol_surface=local_vol,
                        n_paths=150_000, call=True)    # q taken from surface
g_put  = compute_greeks(spot, K=spot, r=r, T=T, n_obs=n_obs,
                        vol_surface=surface, local_vol_surface=local_vol,
                        n_paths=150_000, call=False)
print("call delta:", g_call["delta"], "  put delta:", g_put["delta"])  # opposite signs
print("vegas:", g_call["vega"], g_put["vega"])                          # same sign (positive)
```

Greeks use central finite differences under common random numbers (same `seed=42` for both sides of every bump), so the discretization noise cancels and small bumps are stable. Vega is an absolute parallel vol shift applied identically to the local-vol diffusion and the geometric-Asian CV closed-form vol.

### Run all three arbitrage checks

```python
from spy_asian_pricer import (
    check_butterfly_arbitrage, check_calendar_arbitrage, check_spread_arbitrage,
)
import numpy as np

K = np.linspace(spot * 0.8, spot * 1.2, 80)
print("Calendar:", check_calendar_arbitrage(surface, K))
for exp, (jw, dcf) in jwsvi_slices.items():
    print(exp, check_butterfly_arbitrage(jw.to_svi(dcf), dcf),
                check_spread_arbitrage(surface, dcf, K, r))   # q taken from surface
```

---

## API reference

### Data ingestion (optional `[data]` extra)

| Object | Description |
|---|---|
| `fetch_spot(ticker="SPY") -> float` | Most recent close from yfinance. |
| `fetch_risk_free_rate(tenor_years=0.25, default=0.043) -> float` | Risk-free rate from Yahoo Treasury yields (`^IRX` 13W → `^FVX` 5Y → `^TNX` 10Y → `^TYX` 30Y), linearly interpolated by tenor. Yahoo quotes percent; helper divides by 100. Falls back to `default` on missing fields or network errors. |
| `fetch_dividend_yield(ticker="SPY", default=0.013) -> float` | Continuous-equivalent dividend yield from `Ticker.info`. Auto-normalizes the yfinance "fraction vs percent" inconsistency. Falls back to `default` on missing fields or network errors. |
| `build_vol_grid(ticker="SPY", spot=None, r=0.043, q=None, min_dte=2, max_dte=365*7, ...)` | Pulls option chains and builds the per-expiry IV grid.  IV column is re-implied from bid/ask mid (or `lastPrice` fallback) via `implied_vol_from_price` (Yahoo's own IV is discarded).  Tenor band defaults to `[2, 365*7]` days (standard 7y cutoff).  On noisy Yahoo data tighten via `min_dte=14, max_dte=365*5` and/or post-process with `filter_butterfly_arbitrage`.  When `q` is None it is fetched via `fetch_dividend_yield`. Returns `{expiry_str: DataFrame}`. |
| `select_expiries(ticker="SPY", max_expiries=None, min_dte=2, max_dte=365*7)` | Helper to pick expiries within a DTE band.  Defaults follow the industry-standard "7-year cutoff" convention; the library does NOT decide for you which Yahoo tenors are too noisy.  Tighten with `min_dte=14, max_dte=365*5` for cleaner Yahoo input.  `max_expiries=None` returns every expiry in the band; pass an integer to evenly sub-sample. |
| `bs_european_price(S, K, r, q, sigma, T, call=True) -> float` | Black-Scholes European option price with continuous dividend yield. |
| `implied_vol_from_price(price, S, K, r, q, T, call=True) -> float` | Reverse-imply BS volatility from an observed option price (Brent on `[1e-4, 5.0]`).  Returns `NaN` if price is below forward intrinsic or solver brackets fail. |

### Calibration

| Object | Description |
|---|---|
| `calibrate_svi(y, iv, dcf, weights=None) -> SVIParam` | Weighted least-squares fit of raw SVI to one expiry slice. Default weights are vega-like (Gaussian in `y`). |
| `SVIParam(a, b, rho, m, sigma)` | Raw SVI 5-tuple. Methods: `total_variance(y)`, `implied_vol(y, dcf)`, `dw_dy(y)`, `d2w_dy2(y)`, `to_jwsvi(t)`. |
| `JWSVIParam(nu, phi, p, c, nu_tilda, conv)` | Jump-Wing 6-tuple. Method: `to_svi(t)`. |

### Surface

| Object | Description |
|---|---|
| `JWSVIVolSurface(jwsvi_slices, spot, r, q=0.0)` | Per-slice JWSVI with wing-derived $\tilde\nu$ time interpolation. `q` is continuous dividend yield used in `forward(dcf) = spot * exp((r - q) * dcf)`. |
| `SSVISurface(dcfs, thetas, eta, rho, gamma, spot, r, q=0.0)` | Gatheral SSVI surface (joint $(\eta, \rho, \gamma)$ + monotone $\theta(t)$).  Calendar-arb-free by construction.  Same consumer API as `JWSVIVolSurface`; plugs into `DupireLocalVol` via duck-typed `get_svi_at(dcf)`. |
| `calibrate_ssvi(vol_data, spot, r, q=0.0, mode='pinned')` | Build an `SSVISurface` from `vol_data`.  `mode='pinned'`: 3-param fit, $\theta(t)$ = market ATM.  `mode='full'`: 3+N param fit with monotone $\theta(t)$ (cumsum-of-squares parameterisation). |
| `.forward(dcf) -> float` | Forward price at tenor `dcf`. |
| `.implied_vol(K, dcf)` | Implied vol at strike `K` and time `dcf`. |
| `.total_variance(K, dcf)` | Total variance $w = \sigma_{\mathrm{iv}}^2 T$. |
| `.implied_vol_grid(K_arr, dcf_arr)` | 2-D grid eval, used for surface plots. |

### Local vol

| Object | Description |
|---|---|
| `DupireLocalVol(vol_surface)` | Build the local vol grid via per-strike cubic-spline $dw/dT$ + Gatheral's butterfly denominator. Inherits `r, q` from the surface. Numerical clamps reported in `clamp_stats`. |
| `.local_vol(S, dcf) -> float` | Scalar local vol at $(S, t)$; warns if `(S, dcf)` is outside the grid. Floors at 0. |
| `.local_vol_vec(S_arr, dcf) -> np.ndarray` | Vectorized local vol across many spots at one time (used inside MC). Silently clips to grid; floors at 0. |

### Pricing

| Object | Description |
|---|---|
| `AsianMCPricer(S0, r, T=None, n_obs=None, vol_surface, local_vol_surface, n_steps_per_obs=1, vol_scale=1.0, vol_bump_abs=0.0, q=None, obs_schedule='trading', obs_dcfs=None, start_date=None)` | Monte Carlo pricer (antithetic variates only — no CV). **Observation schedule**: `'trading'` (default) → next `n_obs` weekdays after `start_date` (today); `'calendar'` → uniform $t_i = i\,T/N$ (legacy behavior, required for KV closed-form comparison); `obs_dcfs=` → fully explicit (overrides everything else, `T` derived from `obs_dcfs[-1]`). `n_steps_per_obs` decouples Euler sub-steps from averaging dates and is applied per-interval, so weekend gaps get the correct $\sqrt{3\,\Delta t}$ scaling. `vol_scale` is multiplicative; `vol_bump_abs` is an absolute parallel shift applied AFTER `vol_scale` (used by Vega in `compute_greeks`). `q` defaults to `vol_surface.q`. |
| `.simulate(n_paths, antithetic=True)` | Returns spot at every averaging date, shape `(n, n_obs)`. |
| `.price_asian(K, n_paths=100_000, call=True, antithetic=True)` | Returns dict: `price, std_err, ci_95, n_paths`. Set `call=False` for puts. With `antithetic=True` (default) the SE is computed from pair averages `(f(Z) + f(-Z))/2` so the variance reduction is reflected; with `antithetic=False` the SE is the usual `std(payoff)/sqrt(n)`. |
| `trading_day_obs_dcfs(n_obs, start_date=None, calendar_basis=365.0)` | Calendar-year dcfs for the next `n_obs` weekdays after `start_date` (defaults to today). Holidays are NOT skipped — for an exact NYSE schedule build dcfs externally and pass via `obs_dcfs=`. |
| `geometric_asian_call_price(S0, K, r, sigma, T=None, n_obs=None, call=True, q=0.0, *, obs_dcfs=None)` | Kemna-Vorst closed form. Drift uses $r-q$; discount uses $r$. Two calling conventions: `(T, n_obs)` for the legacy uniform-spacing formula, or `obs_dcfs=` for arbitrary $t_i$ via $\mu_G T = (r-q-\sigma^2/2)\bar t + \sigma_G^2 T/2$ and $\sigma_G^2 T = (\sigma^2/N^2)\sum_{i,j}\min(t_i, t_j)$. Provided as a flat-vol benchmarking utility — not used internally by `price_asian`. |
| `compute_greeks(S0, K, r, T, n_obs, vol_surface, local_vol_surface, n_paths=150_000, n_steps_per_obs=1, seed=42, call=True, q=None, obs_schedule='trading', obs_dcfs=None, start_date=None)` | Finite-difference Delta / Gamma / Vega / Theta under common random numbers. Observation schedule is resolved once and reused across bumps so CRN holds exactly; theta shifts the entire schedule by -1 calendar day. Pass `call=False` for put greeks. `q` defaults to `vol_surface.q`. Vega uses an absolute parallel vol shift on the diffusion. |

### Arbitrage diagnostics

| Object | Description |
|---|---|
| `check_butterfly_arbitrage(svi, dcf)` | Per-slice PDF positivity (Gatheral $g(y) \ge 0$). Returns `(ok, n_violations, g_min)`. |
| `check_calendar_arbitrage(vol_surface, K)` | Total variance non-decreasing in $T$ at every strike. Returns `(ok, n_violations, details)`. |
| `check_spread_arbitrage(vol_surface, dcf, K, r, q=None)` | Call prices monotone in $K$ with slope $\ge -e^{-rT}$, forward $F = S\, e^{(r-q) T}$. `q` defaults to `vol_surface.q`. Returns `(ok, n_violations, details)`. |
| `filter_butterfly_arbitrage(svi_slices, threshold=-0.05, ...)` | Trader-style cleanup: drop slices whose Gatheral $g_{\min}$ is below `threshold`. Returns `(kept_slices, dropped_info)`.  See "Production-style cleanup" below. |

---

## Demo notebook

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) runs the full pipeline end-to-end with charts. Open it in Colab via the badge at the top — the first cell `pip install`s the package, so no local setup is required.

---

## References

- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives. *Global Derivatives & Risk Management*.
- Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59-71.
- Dupire, B. (1994). Pricing with a smile. *Risk*, 7(1), 18-20.
- Kemna, A. G. Z., & Vorst, A. C. F. (1990). A pricing method for options based on average asset values. *Journal of Banking & Finance*, 14(1), 113-129.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.

### Related literature (extensions beyond pure local vol)

- Corsaro, S., Kyriakou, I., Marazzina, D., & Marino, Z. (2019). A general framework for pricing Asian options under stochastic volatility on parallel architectures. *European Journal of Operational Research*, 272(3), 1082-1095.
- Lei, Z., Zhou, Q., & Xiao, W. (2025). Continuous-time Markov chain approximation for pricing Asian options under rough stochastic local volatility models. *Communications in Statistics - Simulation and Computation*, 54(8), 3096-3117.

---

## License

MIT — see [LICENSE](LICENSE).
