# VWAP Trend Trading Strategy Implementation

This repository contains an implementation of the VWAP Trend Trading Strategy as described in the provided paper. It calculates the return using QQQ and TQQQ 1-minute historical data over the specific period provided in the CSV files.

The main calculation engine is built using `polars` for highly-performant vectorization data processing, and `numba` to handle the rapid sequential simulation tracking capital, accurate position sizing, and commission deductions without look-ahead bias.

## Execution
Run the logic using:
```bash
python3 calculate_return.py
```
