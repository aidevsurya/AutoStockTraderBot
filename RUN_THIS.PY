import model
from datetime import datetime, time
from time import sleep
from config import TICKERS

# Define time range
start_time = time(9, 15)
end_time = time(15, 30)

# Get current time
now = datetime.now().time()

# Check if current time is between the range
while 1:
    if start_time <= now <= end_time:
        for stock in TICKERS:
            filename = model.createData(stock)
            model.main(filename)
    else:
        print("STOCK MARKET IS CLOSED... waiting for 1 min")
        sleep(60)
