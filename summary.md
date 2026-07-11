# Predict2Optimize — Final Results Summary (Weeks 3-5 + Robustness Extension)

## Reproducibility

- **Ticker universe:** `AAPL, MSFT, GOOG, AMZN, TSLA` (Week 1's long-term 5-asset universe)
- **Date range:** `2015-01-01` to `2024-01-01` (price history); walk-forward
  out-of-sample evaluation and backtest run from `2017-01-27` to `2023-12-29`
  (the first ~2 years are consumed as initial training window)
- **Seed:** `42`, fixed across NumPy, Python `random`, and PyTorch (`common/pipeline.py::set_seed`)
- **Code:** `common/pipeline.py` (data + features + walk-forward splitter + metrics),
  `common/optimize.py` (classical + robust Markowitz, shared by Weeks 4-5),
  `week3/task3.ipynb`, `week4/task4.ipynb`, `week5/task5.ipynb`
- **Rerun:** `.venv` with `torch`, `cvxpy`, `scikit-learn`, `yfinance` installed;
  price data cached to `data/prices_5asset_2015_2024.csv` after first download

## Predictor comparison (Week 3, pooled walk-forward, 27 expanding windows, 63-day test segments)

| model | MSE | pooled IC | pooled rank-IC | directional accuracy | mean daily cross-sectional rank-IC |
|---|---|---|---|---|---|
| baseline (historical mean) | 0.0006 | -0.013 | -0.008 | 53.3% | -0.018 |
| ridge | 0.0006 | -0.036 | -0.035 | 51.6% | -0.034 |
| **mlp** | 0.0006 | **0.006** | -0.007 | 52.0% | -0.005 |

All three predictors land at essentially the same MSE, and every IC/rank-IC
value is close to zero. This is the expected, honest result for next-day
return prediction on liquid large-cap single-name equities — daily returns
are close to a random walk, so none of these models found meaningful
exploitable signal. The MLP's slightly less-negative rank-IC is not
statistically distinguishable from the baseline given the noise level.

## Strategy backtest (Week 5, weekly rebalance, 337 rebalances, 2017-02-27 to 2023-12-29)

Risk-free rate assumed **0%** (long-only equity backtest, no cash-yield
modeling — Sharpe = annualized return / annualized volatility).
Transaction cost: **10 bps per unit L1 turnover**, deducted at each rebalance.

| strategy | ann. return | ann. vol | Sharpe | max drawdown | Calmar | avg turnover/rebalance | cumulative return | vs baseline (pp) |
|---|---|---|---|---|---|---|---|---|
| (a) mean-return baseline | 26.16% | 26.83% | 0.975 | -38.30% | 0.683 | 0.039 | 389.5% | — |
| (b) classical Markowitz on predictions | 25.20% | 27.40% | 0.920 | -45.37% | 0.555 | 0.618 | 364.5% | -25.0 |
| (c) robust Markowitz on predictions | 27.46% | 26.99% | 1.018 | -41.12% | 0.668 | 0.078 | 425.0% | +35.5 |

**Did the robust variant reduce turnover / improve stability vs classical?**
Yes, decisively: **87.3% lower average turnover** (0.078 vs 0.618 per
rebalance) for a comparable risk profile. This is the mechanism, not a
coincidence — classical Markowitz takes the MLP's near-zero-rank-IC
predictions at face value, and an unconstrained mean-variance optimizer
chases that noise into large weight swings every rebalance. The robust
penalty discounts each asset's predicted return by its own historical
estimation error, which directly damps noise-chasing. This was predicted
*before* running the backtest by Week 4's Monte Carlo allocation-stability
test (perturbing predictions and re-solving both programs 300 times): robust
showed lower weight-dispersion than classical for every single asset (mean
weight std 0.158 vs 0.177).

## Limitations (honest)

- **Transaction cost realism:** 10 bps flat proportional cost is a reasonable
  approximation for large-cap US equities but ignores bid-ask spread
  variation, market impact at size, and the fact that classical Markowitz's
  87% higher turnover would face materially worse realistic slippage than
  this linear model captures — its live-cost disadvantage is likely
  understated here.
- **Overfitting / signal risk:** with predictor rank-IC this close to zero,
  none of the return differences between the three strategies should be read
  as "the MLP beats the market" — the informative, robust finding here is the
  turnover/stability mechanism (Sections above), not the raw return ranking,
  which could plausibly flip under a different seed, cost assumption, or
  rebalance frequency.
- **Sample period:** 2017-2023 was an unusually strong bull market for
  mega-cap tech (all 5 tickers), so absolute return/Sharpe numbers are not
  representative of a general regime and should not be cited outside this
  specific universe and window.
