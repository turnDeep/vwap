# VWAP Trend Trading Strategy Implementation

This repository contains an implementation of the VWAP Trend Trading Strategy as described in the provided paper. It calculates the return using QQQ and TQQQ 1-minute historical data over the specific period provided in the CSV files.

The main calculation engine is built using `polars` for highly-performant vectorization data processing, and `numba` to handle the rapid sequential simulation tracking capital, accurate position sizing, and commission deductions without look-ahead bias.

## Execution
Run the logic using:
```bash
python3 calculate_return.py
```

## Advanced Strategy

Additionally, `calculate_improved_return.py` implements an optimized version of the VWAP strategy. By introducing a 0.3% threshold filter, the algorithm avoids noise and wash-out trades when the price oscillates around VWAP, significantly improving the TQQQ returns from 38% to 67%.

Run the improved logic using:
```bash
python3 calculate_improved_return.py
```
