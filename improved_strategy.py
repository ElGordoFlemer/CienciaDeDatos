
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import warnings
import logging
import os

# Configure logging and warnings
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

# --- Configuration Parameters ---
TRANSACTION_COST = 0.001
SHIFT = 5
MACD_FAST = 12
MACD_SLOW = 26
RSI_PERIOD = 14
ATR_PERIOD = 14
SMA_SLOW = 200
SMA_FAST = 50

CV_FOLDS = 3
TRAIN_SIZE = 0.70
MIN_DATA_DAYS = 100
INITIAL_CAPITAL = 100000

# Position Sizing & Risk Management
POSITION_SIZE_PCT = 0.05 # 5% per trade for diversification
RISK = 0.05   # 5% Stop Loss
REWARD = 0.15 # 15% Take Profit
SLIPPAGE = 0.0005
MAX_CONCURRENT_POSITIONS = 20

def load_insider_data(csv_file='insider_purchases.csv'):
    """Loads and processes insider trading data."""
    if os.path.exists('01_insiders_raw.csv') and os.path.exists('02_daily_signals.csv'):
        print("Loading existing insider data...")
        insider_df = pd.read_csv('01_insiders_raw.csv')
        daily_signals = pd.read_csv('02_daily_signals.csv')
        return insider_df, daily_signals

    try:
        insider_df = pd.read_csv(csv_file, skipinitialspace=True)
        # Clean column names
        insider_df.columns = [col.replace('\xa0', ' ').replace('  ', ' ').strip() for col in insider_df.columns]

        filing_date_col = 'Filing Date'
        if filing_date_col not in insider_df.columns:
            raise KeyError("Filing Date column not found.")

        insider_df['Filing_Date'] = pd.to_datetime(insider_df[filing_date_col]).dt.date

        # Filter for Purchases
        trade_type_col = 'Trade Type'
        if trade_type_col in insider_df.columns:
            insider_df = insider_df[insider_df[trade_type_col] == 'P - Purchase'].copy()

        if insider_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Clean Value column
        value_col = 'Value'
        if value_col in insider_df.columns:
            insider_df['Purchase_Value'] = insider_df[value_col].astype(str).str.replace(r'[\+$,]', '', regex=True)
            insider_df = insider_df[insider_df['Purchase_Value'] != '']
            insider_df['Purchase_Value'] = insider_df['Purchase_Value'].astype(float)
        else:
            insider_df['Purchase_Value'] = 0.0

        ticker_col = 'Ticker'
        daily_signals = insider_df.groupby([ticker_col, 'Filing_Date']).agg({'Purchase_Value': 'sum'}).reset_index()

        return insider_df, daily_signals

    except Exception as e:
        print(f"Error loading insider data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def download_ohlc_data(insider_df):
    """Downloads OHLC data for tickers in the insider dataset."""
    if os.path.exists('03_market_prices.csv'):
        print("Loading existing market prices...")
        return pd.read_csv('03_market_prices.csv')

    if insider_df.empty: return pd.DataFrame()

    tickers = insider_df['Ticker'].unique()

    all_data = []
    for ticker in tickers:
        try:
            yf_ticker = ticker.replace('.', '-')
            start_date = (insider_df[insider_df['Ticker'] == ticker]['Filing_Date'].min() - timedelta(days=365)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')

            data = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if not data.empty and len(data) >= MIN_DATA_DAYS:
                data = data.reset_index()
                data['Ticker'] = ticker
                data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]
                all_data.append(data)
        except Exception:
            pass

    if not all_data: return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

def compute_rsi(series, period=14):
    """Calculates RSI indicator."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period - 1, adjust=False).mean()
    ema_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    """Calculates ATR indicator."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.ewm(span=period, adjust=False).mean()

def add_technical_indicators(df, daily_signals):
    """Adds technical indicators and merges insider signals."""
    if df.empty or daily_signals.empty: return pd.DataFrame()

    df['Date_only'] = pd.to_datetime(df['Date']).dt.date
    daily_signals['Filing_Date'] = pd.to_datetime(daily_signals['Filing_Date']).dt.date

    df = df.merge(daily_signals, left_on=['Ticker', 'Date_only'], right_on=['Ticker', 'Filing_Date'], how='left')

    df['Purchase_Value'] = df['Purchase_Value'].fillna(0)
    df['Log_Purchase_Value'] = np.log1p(df['Purchase_Value'])

    # Insider Signal: Rolling 10-day sum of log purchase value
    df['Rolling_10D_Log_Value'] = df.groupby('Ticker')['Log_Purchase_Value'].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum()
    )

    df.drop(['Date_only', 'Filing_Date', 'Purchase_Value'], axis=1, inplace=True, errors='ignore')

    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    results = []
    for ticker in df['Ticker'].unique():
        mask = df['Ticker'] == ticker
        sub_df = df.loc[mask].copy()

        prices = sub_df['Close']

        # MACD
        ema12 = prices.ewm(span=MACD_FAST, adjust=False).mean()
        ema26 = prices.ewm(span=MACD_SLOW, adjust=False).mean()
        sub_df['MACD'] = ema12 - ema26
        sub_df['MACD'] = sub_df['MACD'].shift(1) # Prevent lookahead bias

        # RSI
        sub_df['RSI'] = compute_rsi(prices, RSI_PERIOD)
        sub_df['RSI'] = sub_df['RSI'].shift(1)

        # ATR
        sub_df['ATR'] = compute_atr(sub_df, ATR_PERIOD)
        sub_df['ATR'] = sub_df['ATR'].shift(1)

        # SMA Trend
        sub_df['SMA50'] = prices.rolling(window=SMA_FAST).mean().shift(1)
        sub_df['SMA200'] = prices.rolling(window=SMA_SLOW).mean().shift(1)

        results.append(sub_df)

    final_df = pd.concat(results)
    # Drop rows where indicators are NaN (warming up periods)
    final_df.dropna(subset=['MACD', 'RSI', 'ATR', 'SMA200', 'SMA50'], inplace=True)
    return final_df.dropna()

def split_data(df):
    """Splits data into train and test sets."""
    if df.empty: return pd.DataFrame(), pd.DataFrame()

    dates = sorted(df['Date'].unique())
    split_idx = int(len(dates) * TRAIN_SIZE)
    split_date = dates[split_idx]

    train = df[df['Date'] <= split_date].copy()
    test = df[df['Date'] > split_date].copy()
    return train, test

def run_strategy_simulation(test_df):
    """Runs the rule-based strategy simulation on the test set."""

    # --- Strategy Signals ---
    # 1. Insider Buying Activity (Rolling_10D_Log_Value > 0)
    # 2. Positive Momentum (MACD > 0)
    # 3. Uptrend (Close > SMA50)
    # 4. Not Overbought (RSI < 75)

    test_df['Signal'] = 0
    signal_mask = (test_df['Rolling_10D_Log_Value'] > 0) & \
                  (test_df['MACD'] > 0) & \
                  (test_df['Close'] > test_df['SMA50']) & \
                  (test_df['RSI'] < 75)

    test_df.loc[signal_mask, 'Signal'] = 1

    print(f"Total signals generated: {test_df['Signal'].sum()}")

    # Shift signal to trade on next open/close (simulated at close here for simplicity)
    test_df['Signal'] = test_df.groupby('Ticker')['Signal'].shift(1).fillna(0)

    # Benchmark Calculation (Buy & Hold equal weight)
    closes = test_df.pivot(index='Date', columns='Ticker', values='Close')
    mkt_ret = closes.pct_change().mean(axis=1).fillna(0)
    bnh_curve = INITIAL_CAPITAL * (1 + mkt_ret).cumprod()
    bnh_final = (bnh_curve.iloc[-1]/INITIAL_CAPITAL - 1)*100

    # Strategy Backtest Loop
    capital = INITIAL_CAPITAL
    curve = [capital]
    positions = {} # {ticker: {'price': float, 'shares': float}}
    trades_log = []

    dates = sorted(test_df['Date'].unique())
    for d in dates:
        day_data = test_df[test_df['Date'] == d]

        # 1. Check Exit Conditions for existing positions
        closed_pnl = 0
        for t in list(positions.keys()):
            if t in day_data['Ticker'].values:
                curr_row = day_data[day_data['Ticker'] == t].iloc[0]
                curr = curr_row['Close']
                entry = positions[t]['price']
                shares = positions[t]['shares']
                rsi = curr_row['RSI']

                ret = (curr / entry) - 1

                # Exit Rules:
                # - Stop Loss: -5%
                # - Take Profit: +15%
                # - Overbought: RSI > 80
                overbought = rsi > 80

                if ret <= -RISK or ret >= REWARD or overbought:
                    pnl = shares * (curr - entry)
                    closed_pnl += pnl
                    trades_log.append(ret)
                    del positions[t]

        capital += closed_pnl

        # 2. Check Entry Conditions
        buys = day_data[day_data['Signal'] == 1]

        for _, row in buys.iterrows():
            if len(positions) < MAX_CONCURRENT_POSITIONS and row['Ticker'] not in positions:
                # Fixed Fractional Position Sizing
                pos_size_value = capital * POSITION_SIZE_PCT
                shares = pos_size_value / row['Close']

                positions[row['Ticker']] = {
                    'price': row['Close'],
                    'shares': shares
                }

        curve.append(capital)

    # --- Performance Metrics ---
    strat_final = (curve[-1]/INITIAL_CAPITAL - 1)*100

    curve_series = pd.Series(curve)
    returns = curve_series.pct_change().dropna()

    if len(returns) > 0:
        volatility = returns.std() * np.sqrt(252)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = (curve_series / curve_series.cummax() - 1).min()
    else:
        volatility = 0
        sharpe = 0
        max_drawdown = 0

    print("-" * 30)
    print("STRATEGY PERFORMANCE REPORT")
    print("-" * 30)
    print(f"Retorno Estrategia:       {strat_final:.2f}%")
    print(f"Retorno Benchmark:        {bnh_final:.2f}%")
    print(f"Total Operaciones:        {len(trades_log)}")
    print(f"Volatilidad Anualizada:   {volatility:.2f}")
    print(f"Sharpe Ratio:             {sharpe:.2f}")
    print(f"Max Drawdown:             {max_drawdown:.2f}")
    print("-" * 30)

    # Save log for review
    # if trades_log:
    #     pd.DataFrame(trades_log, columns=['Return']).to_csv('backtest_trades_log_improved.csv', index=False)

if __name__ == "__main__":
    insider_df, daily_signals = load_insider_data()

    if not insider_df.empty:
        ohlc_df = download_ohlc_data(insider_df)

        if not ohlc_df.empty:
            features_df = add_technical_indicators(ohlc_df, daily_signals)

            train, test = split_data(features_df)

            # We run simulation on Test Set to validate
            run_strategy_simulation(test.copy())
