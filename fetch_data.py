import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def createData(ticker,start_date="2020-01-01",end_date="2025-01-01"):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # ===== Step 2: Technical Indicators =====

    # SMA & EMA
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # RSI
    def RSI(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = RSI(df['Close'])

    # MACD
    EMA12 = df['Close'].ewm(span=12, adjust=False).mean()
    EMA26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA12 - EMA26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    middle_bb = df['Close'].rolling(window=20).mean()  # This is a Series
    std_bb = df['Close'].rolling(window=20).std()      # This is a Series

    df['Middle_BB'] = middle_bb
    df['Upper_BB'] = middle_bb + (2 * std_bb)
    df['Lower_BB'] = middle_bb - (2 * std_bb)

    # VWAP
    df['Cum_Vol_Price'] = (df['Close'] * df['Volume']).cumsum()
    df['Cum_Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cum_Vol_Price'] / df['Cum_Volume']

    # Momentum (ROC)
    df['ROC'] = df['Close'].pct_change(periods=10) * 100

    # ATR (Average True Range) for Volatility
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)

    # Target Column (Next day up/down)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    # ===== Step 3: Clean Data =====
    df.dropna(inplace=True)

    # ===== Step 4: Save as CSV =====
    file_name = f"{ticker}_technical_dataset.csv"
    df.to_csv(file_name)

    print(f"Dataset saved as {file_name}")
    print(df.head())
    return file_name

if __name__=="__main__":
    # ===== Step 1: Fetch data until today =====
    ticker = input("Enter Stock Name (ex. IDEA.NS): ")  # Enter your stock symbol (STOCK.NS, STOCK.BS, etc)
    start_date = input("Enter Date (ex. 2023-01-23): ")
    end_date = datetime.today().strftime('%Y-%m-%d')
    createData(ticker,start_date=start_date,end_date=end_date)