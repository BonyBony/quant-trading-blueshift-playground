# Quant-Trading Blueshift Playground

Starter repo for non-techies to launch back-tests and walk-forward runs on Blueshift with **zero code editing**.

## Files
| File                | Purpose                                             |
|---------------------|------------------------------------------------------|
| `pairs_trading.py`  | Strategy logic, risk, parameter grid                 |
| `wf_runner.py`      | Train-6M / Test-1M walk-forward executor             |
| `README.md`         | Setup guide                                          |

## Quick-start (5 steps)

1. **Clone & branch**

   ```bash
   git clone https://github.com/<org>/quant-trading-blueshift-playground.git
   cd quant-trading-blueshift-playground
   git checkout -b init-pairs-wf-skeleton
