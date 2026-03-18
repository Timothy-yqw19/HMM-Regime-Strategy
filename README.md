# HMM-Regime-Strategy
Summarizing key comparisons between static Hidden Markov Models vs rolling HMM regime Strategies 
# Regime-Switching Multi-Asset Momentum Strategy with HMM
This notebook develops a multi-asset trading strategy that combines:

- multi-lookback momentum
- Hidden Markov Model (HMM) regime detection
- regime-conditioned asset allocation
- transaction costs and benchmark comparison

The strategy dynamically allocates across:

- SPY (US equities)
- QQQ (growth / tech equities)
- TLT (long-term US Treasuries)
- GLD (gold)

The main idea is:

1. Use an HMM to infer hidden market regimes from SPY return and volatility
2. Interpret the hidden states as Bull / Bear / Neutral regimes
3. Adjust asset allocation depending on the regime
4. Compare performance against SPY, Equal Weight, and 60/40 portfolios

## 1. Import libraries

We import the main libraries for:

- data collection (`yfinance`)
- data handling (`pandas`, `numpy`)
- visualization (`matplotlib`)
- regime detection (`hmmlearn`)
- feature scaling (`sklearn`)

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
  
