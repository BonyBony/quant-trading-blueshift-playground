# wf_runner.py – generic walk-forward runner for Blueshift strategies
# Author: <Your-Name> – 20 Jul 2025
"""
Runs rolling train-6m / test-1m folds and aggregates key metrics.
"""

from datetime import timedelta
from typing import Type, List, Dict
import pandas as pd

class WalkForwardRunner:
    def __init__(
        self,
        strategy_cls: Type,
        train_months: int = 6,
        test_months: int = 1,
        start: pd.Timestamp = pd.Timestamp("2022-01-01"),
        end: pd.Timestamp = pd.Timestamp("2024-12-31"),
        max_param_sets: int = 3,
        capital_base: float = 100_000,
    ):
        self.strategy_cls  = strategy_cls
        self.train_months  = train_months
        self.test_months   = test_months
        self.start, self.end = start, end
        self.max_param_sets = max_param_sets
        self.capital_base   = capital_base
        self.metrics: List[pd.Series] = []

    def _folds(self):
        t0 = self.start
        while True:
            train_end = t0 + pd.DateOffset(months=self.train_months) - timedelta(days=1)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + pd.DateOffset(months=self.test_months) - timedelta(days=1)
            if test_end > self.end:
                break
            yield t0, train_end, test_start, test_end
            t0 = test_start

    def run(self):
        for fold, (ts, te, vs, ve) in enumerate(self._folds(), 1):
            print(f"Fold {fold}: train {ts.date()}→{te.date()} | test {vs.date()}→{ve.date()}")
            strat = self.strategy_cls(train_start=ts, train_end=te, capital_base=self.capital_base)
            for params in strat.generate_param_grid()[: self.max_param_sets]:
                algo = strat.build_algorithm(params)
                results = algo.run(start=vs, end=ve)  # Blueshift engine call
                self.metrics.append(results.iloc[-1].rename({"algo_param": "params"}))
        return pd.DataFrame(self.metrics)

if __name__ == "__main__":
    from pairs_trading import PairsTradingStrategy
    df = WalkForwardRunner(PairsTradingStrategy).run()
    df.to_csv("wf_metrics.csv")
    print(df.describe())
