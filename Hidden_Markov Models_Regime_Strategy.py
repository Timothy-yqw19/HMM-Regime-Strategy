# -*- coding: utf-8 -*-
"""
Project: Hidden Markov Models Detection Regime Strategy

Author: Yuqin Wang
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

#(1) STATIC HMM Detection Strategy
##################################################################################

# =========================================
# 1. Download data
# =========================================
tickers = ["SPY", "QQQ", "TLT", "GLD"]
data = yf.download(tickers, start="2010-01-01", end="2024-01-01", auto_adjust=True)["Close"]
data = data.dropna()

returns = data.pct_change().dropna()

# =========================================
# 2. Build SPY features for HMM (determine bull/bear/neutral market based SPY)
# =========================================
spy_ret = returns["SPY"]#daily return of SPY
spy_vol20 = spy_ret.rolling(20).std() * np.sqrt(252)

features = pd.DataFrame({
    "spy_ret": spy_ret,
    "spy_vol20": spy_vol20
}).dropna()

# -----------------------------------------
# Split train / test
# -----------------------------------------
train_end = "2018-12-31"

features_train = features.loc[:train_end].copy()
features_test = features.loc["2019-01-01":].copy()

# =========================================
# 3. Standardize features using TRAIN only
# =========================================
scaler = StandardScaler()
X_train = scaler.fit_transform(features_train.values)
X_test = scaler.transform(features_test.values)

# =========================================
# 4. Fit HMM on TRAIN only
# =========================================
hmm = GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=200,
    random_state=42
)

hmm.fit(X_train)

# Predict hidden states
train_states = hmm.predict(X_train)
test_states = hmm.predict(X_test)

features_train["state"] = train_states
features_test["state"] = test_states

# =========================================
# 5. Label states using TRAIN only
# =========================================
state_summary_train = features_train.groupby("state").agg(
    mean_ret=("spy_ret", "mean"),
    vol=("spy_vol20", "mean"),
    count=("spy_ret", "size")
).sort_values("mean_ret")

print("Train state summary:")
print(state_summary_train)

ordered_states = state_summary_train.index.tolist()

state_labels = {}
state_labels[ordered_states[0]] = "Bear"
state_labels[ordered_states[-1]] = "Bull"

middle_state = [s for s in ordered_states if s not in [ordered_states[0], ordered_states[-1]]][0]
state_labels[middle_state] = "Neutral"

print("\nState label mapping (from TRAIN):")
print(state_labels)

features_train["regime"] = features_train["state"].map(state_labels)
features_test["regime"] = features_test["state"].map(state_labels)

# =========================================
# 6. Daily regime series
# =========================================
daily_regime_train = features_train["regime"].reindex(data.index).ffill()
daily_regime_test = features_test["regime"].reindex(data.index).ffill()

# Only use test period for backtest
test_data = data.loc["2019-01-01":].copy()
test_returns = test_data.pct_change().dropna()

test_daily_regime = features_test["regime"].reindex(test_data.index).ffill()
monthly_regime_test = test_daily_regime.resample("ME").last()

# =========================================
# 7. Multi-lookback momentum on TEST period
#    (using full historical prices up to each point is okay in backtest;
#     but final backtest starts in test period)
# =========================================
lookbacks = [63, 126, 252]
momentum_score = sum([data.pct_change(lb) for lb in lookbacks]) / len(lookbacks)

monthly_score_test = momentum_score.loc["2019-01-01":].resample("ME").last()

# =========================================
# 8. Regime-based allocation on TEST period
# =========================================
monthly_weights_test = pd.DataFrame(0.0, index=monthly_score_test.index, columns=test_data.columns)

for dt in monthly_score_test.index:
    if dt not in monthly_regime_test.index:
        continue

    regime = monthly_regime_test.loc[dt]
    row = monthly_score_test.loc[dt].dropna()

    if len(row) == 0:
        continue

    if regime == "Bull":
        top2 = row.nlargest(2).index
        monthly_weights_test.loc[dt, top2] = 0.5

    elif regime == "Bear":
        monthly_weights_test.loc[dt, "TLT"] = 0.5
        monthly_weights_test.loc[dt, "GLD"] = 0.5

    else:  # Neutral
        top2 = row.nlargest(2).index
        monthly_weights_test.loc[dt, top2] = 0.5

# =========================================
# 9. Daily weights + transaction cost
# =========================================
weights_test = monthly_weights_test.reindex(test_returns.index, method="ffill").fillna(0.0)
weights_test = weights_test.shift(1).fillna(0.0)

cost = 0.001
turnover_test = weights_test.diff().abs().sum(axis=1)

strategy_returns_test = (weights_test * test_returns).sum(axis=1) - cost * turnover_test

# Benchmarks on TEST only
spy_returns_test = test_returns["SPY"]
ew_returns_test = test_returns.mean(axis=1)
portfolio_6040_test = 0.6 * test_returns["SPY"] + 0.4 * test_returns["TLT"]

# =========================================
# 10. Performance stats
# =========================================
def performance_stats(r):
    r = r.dropna()

    ann_return = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    max_dd = drawdown.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan
    sortino = ann_return / (r[r < 0].std() * np.sqrt(252)) if (r < 0).any() else np.nan
    win_rate = (r > 0).mean()

    return pd.Series({
        "Annual Return": ann_return,
        "Annual Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Sortino": sortino,
        "Win Rate": win_rate,
        "Final Value": wealth.iloc[-1]
    })

summary_test = pd.DataFrame({
    "HMM_OOS": performance_stats(strategy_returns_test),
    "SPY_OOS": performance_stats(spy_returns_test),
    "EqualWeight_OOS": performance_stats(ew_returns_test),
    "60_40_OOS": performance_stats(portfolio_6040_test)
})

print("\nOut-of-sample performance summary:")
print(summary_test)

# =========================================
# 11. Plot cumulative returns
# =========================================
cum_hmm = (1 + strategy_returns_test).cumprod()
cum_spy = (1 + spy_returns_test).cumprod()
cum_ew = (1 + ew_returns_test).cumprod()
cum_6040 = (1 + portfolio_6040_test).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(cum_hmm, label="HMM OOS Strategy")
plt.plot(cum_spy, label="SPY OOS")
plt.plot(cum_ew, label="Equal Weight OOS")
plt.plot(cum_6040, label="60/40 OOS")
plt.legend()
plt.title("Out-of-Sample HMM Regime Strategy")
plt.show()

# =========================================
# 12. Regime counts in TEST
# =========================================
print("\nTest regime counts:")
print(features_test["regime"].value_counts())




#ROLLING HMM Detection Strategy
#################################################################################
# =========================================
# 1. Download data
# =========================================
tickers = ["SPY", "QQQ", "TLT", "GLD"]
data = yf.download(tickers, start="2010-01-01", end="2024-12-31", auto_adjust=True)["Close"]
data = data.dropna()

returns = data.pct_change().dropna()

# =========================================
# 2. Feature engineering
# =========================================
spy = data["SPY"]
tlt = data["TLT"]
gld = data["GLD"]

features = pd.DataFrame(index=data.index)
features["SPY_ret"] = spy.pct_change()
features["SPY_vol20"] = features["SPY_ret"].rolling(20).std() * np.sqrt(252)
features["SPY_ma200"] = spy.rolling(200).mean()
features["SPY_trend"] = spy / features["SPY_ma200"] - 1
features["TLT_ret"] = tlt.pct_change()
features["GLD_ret"] = gld.pct_change()

features = features.dropna()

# =========================================
# 3. Multi-lookback momentum
# =========================================
lookbacks = [63, 126, 252]
momentum_score = sum([data.pct_change(lb) for lb in lookbacks]) / len(lookbacks)

# =========================================
# 4. Performance stats function
# =========================================
def performance_stats(r):
    r = r.dropna()
    ann_return = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    max_dd = drawdown.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = ann_return / downside if downside != 0 else np.nan
    win_rate = (r > 0).mean()

    return pd.Series({
        "Annual Return": ann_return,
        "Annual Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Sortino": sortino,
        "Win Rate": win_rate,
        "Final Value": wealth.iloc[-1]
    })

# =========================================
# 5. Walk-forward HMM strategy
# =========================================
years = range(2019, 2025)
all_strategy_returns = []
regime_history = []

for year in years:
    train_end = f"{year-1}-12-31"
    test_start = f"{year}-01-01"
    test_end = f"{year}-12-31"

    # ----- Split train / test
    feat_train = features.loc[:train_end].copy()
    feat_test = features.loc[test_start:test_end].copy()

    if len(feat_train) < 500 or len(feat_test) < 50:
        continue

    # ----- Standardize on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(feat_train.values)
    X_test = scaler.transform(feat_test.values)

    # ----- Fit HMM
    hmm = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        n_iter=200,
        random_state=42
    )
    hmm.fit(X_train)

    train_states = hmm.predict(X_train)
    test_states = hmm.predict(X_test)

    feat_train["state"] = train_states
    feat_test["state"] = test_states

    # ----- Label states using TRAIN only
    state_summary = feat_train.groupby("state").agg(
        mean_ret=("SPY_ret", "mean"),
        vol=("SPY_vol20", "mean"),
        count=("SPY_ret", "size")
    ).sort_values("mean_ret")

    ordered_states = state_summary.index.tolist()

    state_labels = {}
    state_labels[ordered_states[0]] = "Bear"
    state_labels[ordered_states[-1]] = "Bull"
    middle_state = [s for s in ordered_states if s not in [ordered_states[0], ordered_states[-1]]][0]
    state_labels[middle_state] = "Neutral"

    feat_test["regime"] = feat_test["state"].map(state_labels)

    # save regime counts for review
    yearly_regime_count = feat_test["regime"].value_counts()
    regime_history.append(pd.DataFrame({
        "year": year,
        "regime": yearly_regime_count.index,
        "count": yearly_regime_count.values
    }))

    # ----- Monthly regime in test year
    monthly_regime = feat_test["regime"].resample("ME").last()

    # ----- Monthly momentum score in test year
    monthly_score = momentum_score.loc[test_start:test_end].resample("ME").last()

    test_returns = returns.loc[test_start:test_end].copy()

    monthly_weights = pd.DataFrame(0.0, index=monthly_score.index, columns=data.columns)

    for dt in monthly_score.index:
        if dt not in monthly_regime.index:
            continue

        regime = monthly_regime.loc[dt]
        row = monthly_score.loc[dt].dropna()

        if len(row) == 0:
            continue

        # -------------------------------
        # Strategy Switching
        # -------------------------------
        if regime == "Bull":
            # Use Top 2 momentum across all assets
            top2 = row.nlargest(2).index
            monthly_weights.loc[dt, top2] = 0.5

        elif regime == "Bear":
            # Defensive strategy
            monthly_weights.loc[dt, "TLT"] = 0.6
            monthly_weights.loc[dt, "GLD"] = 0.4

        elif regime == "Neutral":
            # More conservative rotation:
            # choose top 2 but cap equity concentration
            top2 = row.nlargest(2).index
            if "SPY" in top2 and "QQQ" in top2:
                monthly_weights.loc[dt, "SPY"] = 0.35
                monthly_weights.loc[dt, "QQQ"] = 0.25
                monthly_weights.loc[dt, "TLT"] = 0.20
                monthly_weights.loc[dt, "GLD"] = 0.20
            else:
                monthly_weights.loc[dt, top2] = 0.4
                remaining = [c for c in ["TLT", "GLD"] if c not in top2]
                if len(remaining) == 2:
                    monthly_weights.loc[dt, remaining] = 0.1

    # ----- Convert to daily weights
    weights = monthly_weights.reindex(test_returns.index, method="ffill").fillna(0.0)
    weights = weights.shift(1).fillna(0.0)

    # ----- Transaction cost
    cost = 0.001
    turnover = weights.diff().abs().sum(axis=1)

    strategy_returns = (weights * test_returns).sum(axis=1) - cost * turnover
    all_strategy_returns.append(strategy_returns)

# =========================================
# 6. Concatenate OOS returns
# =========================================
walkforward_returns = pd.concat(all_strategy_returns).sort_index()

# Benchmarks on same dates
benchmark_returns = returns["SPY"].reindex(walkforward_returns.index).dropna()
ew_returns = returns.mean(axis=1).reindex(walkforward_returns.index).dropna()
portfolio_6040_returns = (0.6 * returns["SPY"] + 0.4 * returns["TLT"]).reindex(walkforward_returns.index).dropna()

# Align
common_idx = walkforward_returns.index.intersection(benchmark_returns.index)
walkforward_returns = walkforward_returns.loc[common_idx]
benchmark_returns = benchmark_returns.loc[common_idx]
ew_returns = ew_returns.loc[common_idx]
portfolio_6040_returns = portfolio_6040_returns.loc[common_idx]

# =========================================
# 7. Summary table
# =========================================
summary = pd.DataFrame({
    "Rolling_HMM_Strategy": performance_stats(walkforward_returns),
    "SPY": performance_stats(benchmark_returns),
    "EqualWeight": performance_stats(ew_returns),
    "60_40": performance_stats(portfolio_6040_returns)
})

print(summary)

# =========================================
# 8. Plot cumulative returns
# =========================================
cum_strategy = (1 + walkforward_returns).cumprod()
cum_spy = (1 + benchmark_returns).cumprod()
cum_ew = (1 + ew_returns).cumprod()
cum_6040 = (1 + portfolio_6040_returns).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(cum_strategy, label="Rolling HMM Strategy")
plt.plot(cum_spy, label="SPY")
plt.plot(cum_ew, label="Equal Weight")
plt.plot(cum_6040, label="60/40")
plt.legend()
plt.title("Rolling HMM + Feature Engineering + Strategy Switching")
plt.show()

# =========================================
# 9. Regime counts
# =========================================
regime_history_df = pd.concat(regime_history, ignore_index=True)
print(regime_history_df)