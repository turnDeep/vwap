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
        if i == n - 1 or dates[i+1] != dates[i]:
            target_pos = 0
        if target_pos != curr_pos:
            price = closes[i] if (target_pos == 0 and (i == n-1 or dates[i+1] != dates[i])) else opens[i]
            if curr_pos == 1:
                capital += shares * price
                capital -= shares * commission
                shares = 0
            elif curr_pos == -1:
                capital -= shares * price
                capital -= shares * commission
                shares = 0
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

def run_improved_strategy(file_path):
    df = pl.read_csv(file_path)
    df = df.with_columns(pl.col('date').str.to_datetime('%Y-%m-%d %H:%M:%S'))
    df = df.unique(subset=['date'], keep='first').sort('date')

    df = df.filter(
        ((pl.col('date').dt.hour() == 9) & (pl.col('date').dt.minute() >= 30)) |
        ((pl.col('date').dt.hour() >= 10) & (pl.col('date').dt.hour() <= 15))
    )

    df = df.with_columns(pl.col('date').dt.date().alias('day'))

    df = df.with_columns(typical_price=(pl.col('high') + pl.col('low') + pl.col('close')) / 3.0)
    df = df.with_columns(vp=pl.col('typical_price') * pl.col('volume'))

    df = df.with_columns([
        pl.col('vp').cum_sum().over('day').alias('cum_vp'),
        pl.col('volume').cum_sum().over('day').alias('cum_vol')
    ])
    df = df.with_columns(vwap=pl.col('cum_vp') / pl.col('cum_vol'))

    # Improved Strategy: VWAP Trend Trading + 0.3% Threshold filter.
    # The original strategy trades immediately when price crosses VWAP, resulting in many whipsaws (false signals) and commission burn.
    # By requiring the close to be at least 0.3% beyond VWAP before changing the signal, we avoid "chop" while capturing strong momentum.
    threshold = 0.003
    df = df.with_columns(
        signal_raw=pl.when(pl.col('close') > pl.col('vwap') * (1 + threshold)).then(1)
                .when(pl.col('close') < pl.col('vwap') * (1 - threshold)).then(-1)
                .otherwise(0)
    )

    df = df.with_columns(signal=pl.col('signal_raw').replace(0, None).forward_fill().fill_null(0).over('day'))
    df = df.with_columns(pos=pl.col('signal').shift(1).fill_null(0).over('day'))

    dates_int = df['day'].cast(pl.Int32).to_numpy()
    opens = df['open'].to_numpy()
    closes = df['close'].to_numpy()
    pos = df['pos'].to_numpy()

    return simulate_trading(dates_int, opens, closes, pos)

if __name__ == "__main__":
    qqq_cap = run_improved_strategy('qqq_1min_historical_data.csv')
    tqqq_cap = run_improved_strategy('tqqq_1min_historical_data.csv')

    print(f"Improved VWAP TQQQ Return: {(tqqq_cap / 25000 - 1) * 100:.2f}%")
