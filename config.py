STOCKS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
SHEET_NAME = "AlgoTradingLog"

TELEGRAM_TOKEN = "REPLACE_YOUR_TOKEN"

TELEGRAM_CHAT_ID = "1676503822" # REPLACE by YOUR CHAT ID

SHEET_EMAIL = "REPLACE BY YOUR GOOGLE CLOUD CONSOLE SERVICE ACCOUNT API EMAIL PROVIDED TO YOU" # this is NOT general email (abc@gmail.com)

TICKERS = ["INFY.NS","RELIANCE.NS","TATAPOWER.NS"]

# DATASET_PATH = input("Enter Your CSV Dataset: ")  # your technical indicators CSV
MODEL_PATH = ""
RANDOM_STATE = 42

# Backtest config
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_FRAC = 1.0
STOP_LOSS_PCT = 1.0
TAKE_PROFIT_PCT = 0.5

# Decision thresholds
BUY_PROB_THRESHOLD = 0.8
SELL_PROB_THRESHOLD = 0.2

# Backtest period config
BACKTEST_MONTHS = 6 # int(input("Enter PaperTrading Duration in months (max: 12): "))  # how far back to test
