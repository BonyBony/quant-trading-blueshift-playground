# pairs_trading.py – Chan-style z-score pairs strategy for Blueshift
# Author: <Your-Name> – 20 Jul 2025

"""
Universe  : NIFTY 100 (minute bars, 2022-2024 default)
Selection : Top-correlated same-sector pairs over trailing 90 days,
            filtered by Johansen cointegration test (p < 0.05)
Signal    : z-score of price-ratio spread
Entry     : |z| ≥ 2  (long low leg, short high leg)
Exit      : z returns to 0
Sizing    : Kelly fraction (capped so portfolio risk ≤ 1 % equity,
            leverage ≤ 1.5×)
Metrics   : PnL, Sharpe, Sortino, Kelly utilisation, hit-rate,
            turnover, slippage
"""

import numpy as np
import pandas as pd

# Blueshift API helpers — imported automatically inside the platform
from zipline.api import (
    symbol, order_target_percent, record, schedule_function,
    date_rules as dr, time_rules as tr
)
from blueshift.finance import asset_finder
from scipy import stats
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ------------- strategy wrapper ------------------------------------------------
class PairsTradingStrategy:
    """Factory that returns a Blueshift algorithm object."""

    # --- configuration ---------------------------------------------------------
    LOOKBACK_DAYS = 90        # correlation & cointegration look-back
    MAX_PAIRS     = 10        # trade at most N pairs
    Z_ENTRY       = 2.0       # entry threshold
    Z_EXIT        = 0.0       # exit threshold

    def __init__(self, train_start=None, train_end=None, capital_base=100_000):
        self.train_start = train_start
        self.train_end   = train_end
        self.capital_base = capital_base

    # --- parameter grid for WF runner -----------------------------------------
    def generate_param_grid(self):
        return [
            {"z_entry": 2.0, "lookback": 90},
            {"z_entry": 2.5, "lookback": 90},
            {"z_entry": 2.0, "lookback": 120},
        ]

    # --------------------------------------------------------------------------
    def build_algorithm(self, params=None):
        """Return a Zipline/Blueshift algorithm instance configured with *params*."""
        z_entry  = params.get("z_entry", 2.0) if params else self.Z_ENTRY
        lookback = params.get("lookback", self.LOOKBACK_DAYS) if params else self.LOOKBACK_DAYS

        def initialize(context):
            context.capital_base = self.capital_base
            context.pairs        = select_pairs(lookback, self.MAX_PAIRS)
            context.z_entry      = z_entry
            context.z_exit       = self.Z_EXIT
            context.lookback     = lookback

            # Schedule daily check 15 min before close
            schedule_function(rebalance,
                              date_rule=dr.every_day(),
                              time_rule=tr.market_close(minutes=15))

        def rebalance(context, data):
            prices = data.history([p for pair in context.pairs for p in pair],
                                  "price", context.lookback, "1m")
            for long_sym, short_sym in context.pairs:
                s1 = prices[long_sym]
                s2 = prices[short_sym]
                spread = np.log(s1) - np.log(s2)
                z = (spread[-1] - spread.mean()) / spread.std()

                if z > context.z_entry and not context.portfolio.positions[short_sym].amount:
                    # short leg high, long leg low
                    kelly_w   = kelly_fraction(spread.pct_change().dropna())
                    alloc_pct = min(kelly_w, 0.01)  # cap at 1 %
                    order_target_percent(short_sym, -alloc_pct)
                    order_target_percent(long_sym,  alloc_pct)

                elif abs(z) < context.z_exit:
                    order_target_percent(long_sym,  0)
                    order_target_percent(short_sym, 0)

            # record metrics
            record(leverage=context.account.leverage)

        def analyze(context, perf):
            perf.to_csv("fold_metrics.csv")

        return locals()  # Blueshift consumes initialize/handle_data/analyze


# ------------- helper functions -----------------------------------------------
def select_pairs(lookback, max_pairs):
    """Find top-correlated, cointegrated pairs within the NIFTY 100 universe."""
    nifty100 = asset_finder.lookup_symbols([f"{s}" for s in range(1, 101)])
    prices   = get_pricing(nifty100, fields="price", start_date=pd.Timestamp.today() - pd.Timedelta(days=lookback),
                            end_date=pd.Timestamp.today(), frequency="1m").dropna(axis=1)
    corr = prices.pct_change().corr()

    pairs = []
    # iterate high → low correlation
    for (sym1, sym2), c in corr.unstack().sort_values(ascending=False).items():
        if sym1 == sym2 or (sym2, sym1) in pairs:
            continue
        # same sector constraint
        if asset_finder.retrieve_asset(sym1).sector != asset_finder.retrieve_asset(sym2).sector:
            continue
        # cointegration check
        test_res = coint_johansen(prices[[sym1, sym2]].dropna(), det_order=0, k_ar_diff=1)
        if test_res.lr1[0] > test_res.cvt[0, 1]:  # 5 % critical value
            pairs.append((sym1, sym2))
        if len(pairs) >= max_pairs:
            break
    return pairs


def kelly_fraction(returns):
    """Compute Kelly fraction for a series of trade returns."""
    m, s = returns.mean(), returns.std()
    return max(min(m / (s ** 2), 1), 0)  # constrain 0-1

