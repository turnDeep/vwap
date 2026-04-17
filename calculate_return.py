import polars as pl
import numba
import numpy as np

@numba.njit
def simulate_trading(dates, opens, closes, pos, initial_capital=25000.0, commission=0.0005):
    capital = initial_capital
    shares = 0.0
    curr_pos = 0

    n = len(opens)
    for i in range(n):
        target_pos = pos[i]

        # Close at EOD
        if i == n - 1 or dates[i+1] != dates[i]:
            target_pos = 0

        if target_pos != curr_pos:
            price = closes[i] if (target_pos == 0 and (i == n-1 or dates[i+1] != dates[i])) else opens[i]

            # Close existing
            if curr_pos == 1:
                capital += shares * price
                capital -= shares * commission
                shares = 0
            elif curr_pos == -1:
                capital -= shares * price
                capital -= shares * commission
                shares = 0

            # Open new position
            if target_pos == 1:
                shares = capital / price
                capital -= shares * price
                capital -= shares * commission
            elif target_pos == -1:
                shares = capital / price
                capital += shares * price
                capital -= shares * commission

            curr_pos = target_pos

    return capital

def run_strategy(file_path):
    df = pl.read_csv(file_path)
    # Convert date string to datetime
    df = df.with_columns(pl.col('date').str.to_datetime('%Y-%m-%d %H:%M:%S'))
    # Deduplicate and sort
    df = df.unique(subset=['date'], keep='first').sort('date')

    # Filter Regular Trading Hours: 9:30 to 15:59
    df = df.filter(
        ((pl.col('date').dt.hour() == 9) & (pl.col('date').dt.minute() >= 30)) |
        ((pl.col('date').dt.hour() >= 10) & (pl.col('date').dt.hour() <= 15))
    )

    df = df.with_columns(pl.col('date').dt.date().alias('day'))

    # VWAP Calculation
    df = df.with_columns(typical_price=(pl.col('high') + pl.col('low') + pl.col('close')) / 3.0)
    df = df.with_columns(vp=pl.col('typical_price') * pl.col('volume'))

    df = df.with_columns([
        pl.col('vp').cum_sum().over('day').alias('cum_vp'),
        pl.col('volume').cum_sum().over('day').alias('cum_vol')
    ])
    df = df.with_columns(vwap=pl.col('cum_vp') / pl.col('cum_vol'))

    # Generate Trading Signal
    # "initiates long positions when price is above the VWAP and short positions when it falls below the VWAP"
    df = df.with_columns(
        signal_raw=pl.when(pl.col('close') > pl.col('vwap')).then(1)
                .when(pl.col('close') < pl.col('vwap')).then(-1)
                .otherwise(0)
    )
    # Forward fill 0 to maintain previous signal if close == vwap
    df = df.with_columns(signal=pl.col('signal_raw').replace(0, None).forward_fill().fill_null(0).over('day'))

    # Shift signal by 1 minute to avoid future data leakage.
    # The signal from minute `t-1` close determines position for minute `t`.
    df = df.with_columns(pos=pl.col('signal').shift(1).fill_null(0).over('day'))

    # Extract data for simulation
    dates_int = df['day'].cast(pl.Int32).to_numpy()
    opens = df['open'].to_numpy()
    closes = df['close'].to_numpy()
    pos = df['pos'].to_numpy()

    # Execute Numba Simulation
    final_cap = simulate_trading(dates_int, opens, closes, pos)
    return final_cap

if __name__ == "__main__":
    qqq_cap = run_strategy('qqq_1min_historical_data.csv')
    tqqq_cap = run_strategy('tqqq_1min_historical_data.csv')

    print(f"QQQ Return: {(qqq_cap / 25000 - 1) * 100:.2f}%")
    print(f"TQQQ Return: {(tqqq_cap / 25000 - 1) * 100:.2f}%")
