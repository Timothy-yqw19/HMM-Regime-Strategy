# HMM Regime-Switching Multi-Asset Strategy

A quantitative research project that combines **multi-lookback momentum** with **Hidden Markov Model (HMM) regime detection** to dynamically allocate across equities, bonds, and gold.

## Project Overview

This project studies whether market regime detection can improve the risk-adjusted performance of a cross-asset momentum strategy.

The strategy allocates across:

- **SPY**: US equities
- **QQQ**: tech / growth equities
- **TLT**: long-duration Treasuries
- **GLD**: gold

The main workflow is:

1. Build momentum signals using 3M / 6M / 12M lookbacks
2. Use an HMM to infer hidden market regimes from SPY return and volatility
3. Label regimes as **Bull / Bear / Neutral**
4. Switch allocation rules depending on the regime
5. Compare performance against SPY, Equal Weight, and 60/40 benchmarks

---

## Methodology

### 1. Momentum Signal
Momentum is computed using three lookback windows:

- 63 trading days
- 126 trading days
- 252 trading days

The final momentum score is the average across windows.

### 2. Regime Detection
A **Gaussian Hidden Markov Model (HMM)** is trained on:

- SPY daily returns
- SPY 20-day rolling annualized volatility

The inferred hidden states are interpreted post hoc as:

- **Bull**
- **Bear**
- **Neutral**

### 3. Regime-Based Allocation
- **Bull**: allocate to the top 2 assets by momentum
- **Bear**: allocate defensively to TLT and GLD
- **Neutral**: use a balanced top-2 momentum allocation

### 4. Backtest Assumptions
- Monthly rebalancing
- 1-day lag to avoid look-ahead bias
- 10 bps transaction cost

---

## Main Results

### Static HMM Strategy

| Metric | HMM Strategy | SPY | Equal Weight | 60/40 |
|---|---:|---:|---:|---:|
| Annual Return | 11.84% | 13.65% | 10.51% | 10.01% |
| Annual Volatility | 12.94% | 17.33% | 10.25% | 10.11% |
| Sharpe Ratio | **0.92** | 0.79 | 1.03 | 0.99 |
| Max Drawdown | **-25.9%** | -33.7% | -25.2% | -27.2% |
| Calmar Ratio | **0.46** | 0.40 | 0.42 | 0.37 |

### Key Findings
- The HMM-based regime strategy improves **risk-adjusted performance** relative to SPY
- It significantly reduces volatility and drawdown
- It outperforms a traditional **60/40 portfolio**
- Equal-weight diversification remains a strong baseline

---

## Robustness Checks

This project also evaluates two more demanding extensions:

### Out-of-Sample HMM
The HMM is trained only on 2010–2018 and tested on 2019–2024.

**Finding:** performance deteriorates out-of-sample, suggesting regime instability and sensitivity to market structure shifts.

### Rolling HMM
The HMM is retrained annually in a walk-forward setting with richer engineered features.

**Finding:** the rolling HMM underperforms due to unstable regime classification, showing that greater model complexity does not necessarily improve real-world investment performance.

---

## Why This Project Matters

This project is not just a trading strategy demo. It is a **quantitative research workflow** that includes:

- baseline momentum strategies
- multi-asset portfolio construction
- HMM-based regime detection
- transaction-cost-aware backtesting
- benchmark comparison
- out-of-sample validation
- failure analysis of more complex models

---

## Repository Structure

```text
notebooks/
    01_main_hmm_strategy.ipynb
    02_robustness_checks.ipynb

src/
    data_loader.py
    features.py
    regime_hmm.py
    momentum.py
    backtest.py
    benchmarks.py
    metrics.py
    plotting.py
