#!/usr/bin/env python3

#MY CUSTOM LIBS
import sheets_logger as MySheet
import telegram_alerts as tele
from fetch_data import createData
#EXTERNAL LIBS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# -----------------------------
# Config
# -----------------------------
client = MySheet.connect_gsheet()
spreadsheet = MySheet.open_spreadsheet(client,"AlgoTradingLog")
dataSheet = MySheet.open_sheet_tab(spreadsheet,"Sheet1")

#Configured User Variables
from config import *

# -----------------------------
# Data loading & cleaning
# -----------------------------
def load_and_clean(path):
    df = pd.read_csv(path)
    # print(df.info())
    df.rename(columns={'Price': 'Date'}, inplace=True)
    # print(df.info())
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.set_index('Date')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    # print(df.info())
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if 'Target' in non_numeric:
        df['Target'] = pd.to_numeric(df['Target'], errors='coerce')
        # print(df.info())
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if 'Close' in non_numeric:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        # print(df.info())
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')

    # for col in non_numeric:
    #     if col != 'Target':
    #         df.drop(columns=[col], inplace=True)
    df.dropna(inplace=True)
    return df

# -----------------------------
# Feature prep
# -----------------------------
def prepare_xy(df):
    if 'Target' not in df.columns:
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        df.dropna(inplace=True)
    X = df.drop(columns=['Target'])
    y = df['Target'].astype(int)
    return X, y

# -----------------------------
# Model training
# -----------------------------
def train_model(X_train, y_train):
    # model = GaussianNB()
    # model = RandomForestClassifier(
        # criterion="entropy",
        # n_estimators=600,
        # max_depth=20,
        # min_samples_split=6,
        # random_state=RANDOM_STATE,
        # n_jobs=-1
    model = xgb.XGBClassifier(n_estimators=200, max_depth=32, learning_rate=0.05)
    model.fit(X_train, y_train)
    return model

# -----------------------------
# Signal generator
# -----------------------------
def generate_signals_from_probs(probs):
    p1 = probs[:, 1] if probs.ndim == 2 else probs
    signals = []
    for p in p1:
        if p >= BUY_PROB_THRESHOLD:
            signals.append("BUY")
        elif p <= SELL_PROB_THRESHOLD:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    return signals, p1

# -----------------------------
# Paper trading with P/L dataset
# -----------------------------
def backtest_with_trades(df_prices, signals):
    capital = INITIAL_CAPITAL
    position = 0.0
    entry_price = None
    entry_date = None
    trade_id = 0
    trades = []
    portfolio_values = []

    for i in range(len(df_prices)):
        price = df_prices['Close'].iloc[i]
        date = df_prices.index[i]
        signal = signals[i]

        if position > 0:
            ret = (price - entry_price) / entry_price
            if ret <= -STOP_LOSS_PCT or ret >= TAKE_PROFIT_PCT or signal == "SELL":
                proceeds = position * price
                pnl_amount = proceeds - (position * entry_price)
                pnl_pct = pnl_amount / (position * entry_price)
                capital += proceeds
                trade_id += 1
                trades.append({
                    'Trade_ID': trade_id,
                    'Entry_Date': entry_date,
                    'Entry_Price': entry_price,
                    'Exit_Date': date,
                    'Exit_Price': price,
                    'P/L_Amount': round(pnl_amount, 2),
                    'P/L_%': round(pnl_pct * 100, 2)
                })
                position = 0
                entry_price = None
                entry_date = None

        if position == 0 and signal == "BUY":
            alloc = capital * POSITION_SIZE_FRAC
            if alloc >= price:
                position = alloc / price
                entry_price = price
                entry_date = date
                capital -= alloc

        pv = capital + (position * price)
        portfolio_values.append(pv)

    if position > 0:
        last_price = df_prices['Close'].iloc[-1]
        last_date = df_prices.index[-1]
        proceeds = position * last_price
        pnl_amount = proceeds - (position * entry_price)
        pnl_pct = pnl_amount / (position * entry_price)
        capital += proceeds
        trade_id += 1
        trades.append({
            'Trade_ID': trade_id,
            'Entry_Date': entry_date,
            'Entry_Price': entry_price,
            'Exit_Date': last_date,
            'Exit_Price': last_price,
            'P/L_Amount': round(pnl_amount, 2),
            'P/L_%': round(pnl_pct * 100, 2)
        })
        position = 0

    return capital, trades, portfolio_values

# -----------------------------
# Plotting & Save Images
# -----------------------------

def plot_results(price_df, trades_df, portfolio_values, output_prefix="results"):
    # Ensure index is datetime
    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)

    # 1) Price chart with BUY (entry) and SELL (exit) markers
    fig1, ax1 = plt.subplots(figsize=(14,6))
    ax1.plot(price_df.index, price_df['Close'], label='Close Price')
    # Plot entry and exit points
    if not trades_df.empty:
        # convert trade dates to datetime
        trades_df['Entry_Date'] = pd.to_datetime(trades_df['Entry_Date'])
        trades_df['Exit_Date'] = pd.to_datetime(trades_df['Exit_Date'])
        entries = trades_df[['Entry_Date', 'Entry_Price']].dropna()
        exits = trades_df[['Exit_Date', 'Exit_Price']].dropna()
        if not entries.empty:
            ax1.scatter(entries['Entry_Date'], entries['Entry_Price'], marker='^', s=80, label='ENTRY', zorder=5)
        if not exits.empty:
            ax1.scatter(exits['Exit_Date'], exits['Exit_Price'], marker='v', s=80, label='EXIT', zorder=5)
    ax1.set_title('Price with Entry/Exit Signals')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(f"{output_prefix}_price_with_signals.png")
    plt.close(fig1)

    # 2) Equity curve (portfolio value over time)
    fig2, ax2 = plt.subplots(figsize=(12,5))
    # portfolio_values should align with price_df index length
    idx = price_df.index[:len(portfolio_values)]
    ax2.plot(idx, portfolio_values, label='Portfolio Value')
    ax2.axhline(y=INITIAL_CAPITAL, linestyle='--', label='Initial Capital')
    ax2.set_title('Equity Curve (Portfolio Value over Time)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Portfolio Value')
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"{output_prefix}_equity_curve.png")
    plt.close(fig2)

    # 3) Trade P/L bar chart (amount)
    fig3, ax3 = plt.subplots(figsize=(12,5))
    if not trades_df.empty:
        trades_df_sorted = trades_df.sort_values('Trade_ID')
        ax3.bar(trades_df_sorted['Trade_ID'].astype(str), trades_df_sorted['P/L_Amount'])
    ax3.set_title('Individual Trade P/L Amount')
    ax3.set_xlabel('Trade ID')
    ax3.set_ylabel('P/L Amount')
    fig3.tight_layout()
    fig3.savefig(f"{output_prefix}_trade_pnl_bars.png")
    plt.close(fig3)

    # 4) Cumulative P/L across trades
    fig4, ax4 = plt.subplots(figsize=(12,5))
    if not trades_df.empty:
        trades_df_sorted = trades_df.sort_values('Trade_ID')
        cum_pnl = trades_df_sorted['P/L_Amount'].cumsum()
        ax4.plot(trades_df_sorted['Trade_ID'].astype(int), cum_pnl, marker='o')
    ax4.set_title('Cumulative P/L Over Trades')
    ax4.set_xlabel('Trade ID')
    ax4.set_ylabel('Cumulative P/L')
    fig4.tight_layout()
    fig4.savefig(f"{output_prefix}_cumulative_pnl.png")
    plt.close(fig4)

    print(f"Saved plots: {output_prefix}_price_with_signals.png, {output_prefix}_equity_curve.png, "
          f"{output_prefix}_trade_pnl_bars.png, {output_prefix}_cumulative_pnl.png")


try:
    plot_results(price_df, trades_df, portfolio_values, output_prefix="surya_results")
except Exception as e:
    print("Plotting failed:", e)

# -----------------------------
# Main pipeline
# -----------------------------
def main(DATASET_PATH):
    global MODEL_PATH
    MODEL_PATH = DATASET_PATH.split(".")[0]+"_model.pkl"
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    print("Loading dataset...")
    df = load_and_clean(DATASET_PATH)
    # print(df.info())
    # Restrict to last N months
    cutoff_date = df.index.max() - pd.DateOffset(months=BACKTEST_MONTHS)
    df_bt = df[df.index >= cutoff_date]
    
    price_df = df_bt[['Close']].copy()
    X, y = prepare_xy(df_bt)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("Training model...")
    model = train_model(X_train, y_train)
    print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")

    probs = model.predict_proba(X)
    signals, _ = generate_signals_from_probs(probs)

    final_capital, trades, portfolio_values = backtest_with_trades(price_df, signals)

    trades_df = pd.DataFrame(trades)
    MySheet.write_dataframe(dataSheet,trades_df)
    print("Successfully Written to Google Sheet: AlgoTradingLog") # Uncomment to save in Google Sheet
    trades_df.to_csv("profit_loss_report.csv", index=False) # Uncomment to save file locally

    print("\n=== Backtest Report ===")
    print(f"Initial Capital: {INITIAL_CAPITAL}")
    print(f"Final Capital: {round(final_capital, 2)}")
    print(f"Total Profit/Loss: {round(final_capital - INITIAL_CAPITAL, 2)}")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Trade log saved: profit_loss_report.csv")
    
    #============LIVE GRAPH ANIMATION FOR ENTRY AND EXIT DATA======
    
    # # ensure signals is a list and price index is datetime
    # signals = list(signals)
    # fig, (ax_price, ax_equity) = plt.subplots(2, 1, figsize=(12, 8))
    # plt.tight_layout(pad=3)

    # timestamps = []
    # live_prices = []
    # live_signals = []
    # live_equity = []

    # # Use mutable state dict instead of nonlocal
    # state = {
    #     'capital': INITIAL_CAPITAL,
    #     'position': 0.0,
    #     'entry_price': None,
    #     'start_time': time.time()
    # }
    # DURATION_SEC = 60
    # INTERVAL_MS = 500  # update interval in milliseconds

    # def update(frame):
    #     elapsed = time.time() - state['start_time']
    #     if elapsed > DURATION_SEC:
    #         plt.close(fig)
    #         return

    #     # pick a price from historical price_df to simulate a live stream
    #     idx = min(frame, len(price_df) - 1)
        
    #     price = float(price_df['Close'].iloc[idx])
    #     sig = signals[idx]

    #     timestamps.append(price_df.index[idx])
    #     live_prices.append(price)
    #     live_signals.append(sig)

    #     # simple paper-trade simulation using the state dict
    #     if state['position'] > 0:
    #         ret = (price - state['entry_price']) / state['entry_price']
    #         if ret <= -STOP_LOSS_PCT or ret >= TAKE_PROFIT_PCT or sig == "SELL":
    #             state['capital'] += state['position'] * price
    #             state['position'] = 0.0
    #             state['entry_price'] = None

    #     if state['position'] == 0 and sig == "BUY":
    #         alloc = state['capital'] * POSITION_SIZE_FRAC
    #         if alloc >= price:
    #             state['position'] = alloc / price
    #             state['entry_price'] = price
    #             state['capital'] -= alloc

    #     pv = state['capital'] + state['position'] * price
    #     live_equity.append(pv)

    #     # plotting
    #     ax_price.clear()
    #     ax_equity.clear()

    #     ax_price.plot(timestamps, live_prices, label='Close Price')
    #     buys = [timestamps[i] for i, s in enumerate(live_signals) if s == 'BUY']
    #     buy_prices = [live_prices[i] for i, s in enumerate(live_signals) if s == 'BUY']
    #     sells = [timestamps[i] for i, s in enumerate(live_signals) if s == 'SELL']
    #     sell_prices = [live_prices[i] for i, s in enumerate(live_signals) if s == 'SELL']
    #     if buys:
    #         ax_price.scatter(buys, buy_prices, marker='^', label='BUY', zorder=5)
    #     if sells:
    #         ax_price.scatter(sells, sell_prices, marker='v', label='SELL', zorder=5)

    #     ax_price.set_title(f'Live Price Stream (Elapsed: {elapsed:.1f}s)')
    #     ax_price.set_ylabel('Price')
    #     ax_price.legend(loc='upper left')

    #     ax_equity.plot(timestamps, live_equity, label='Portfolio Value')
    #     ax_equity.axhline(y=INITIAL_CAPITAL, linestyle='--', label='Initial Capital')
    #     ax_equity.set_ylabel('Portfolio Value')
    #     ax_equity.set_xlabel('Time')
    #     ax_equity.legend(loc='upper left')

    # ani = FuncAnimation(fig, update, interval=INTERVAL_MS)
    # plt.show()
    
    #============LIVE GRAPH ANIMATION FOR ENTRY AND EXIT DATA======
    
    
    # Next action prediction
    last_X = X.iloc[[-1]]
    last_prob = model.predict_proba(last_X)
    print(last_prob)
    last_prob = last_prob[0][1]
    last_signal = "BUY" if last_prob >= BUY_PROB_THRESHOLD else ("SELL" if last_prob <= SELL_PROB_THRESHOLD else "HOLD")
    plot_results(price_df,trades_df,portfolio_values,output_prefix=DATASET_PATH.split(".")[0])
    print("\n=== Next Action ===")
    print(f"Date: {price_df.index[-1]}")
    print(f"Probability up: {last_prob:.4f}")
    print(f"Action: {last_signal}")
    
    alert_string = f"""
    \n===Stock: {DATASET_PATH.split(".")[0]}===\n
    Date: {price_df.index[-1]}
    Probability up: {last_prob:.4f}
    Action: {last_signal}
    """
    
    tele.send_alert(alert_string)
    
if __name__ == "__main__":
    TICKERS = ["INFY.NS","RELIANCE.NS","TATAPOWER.NS"]
    
    for stock in TICKERS:
        filename = createData(stock)
        main(filename)
