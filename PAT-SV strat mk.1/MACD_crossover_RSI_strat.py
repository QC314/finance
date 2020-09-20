import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from matplotlib.dates import date2num
import pandas as pd
from function_library import *
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.utils import dropna
from tqdm import tqdm

# Pull and cut down data

total_start_time = time.time()

start_datetime = '2020-07-09 10:30:00'
end_datetime = '2020-07-09 11:00:00'

root_dir = 'D:\\network_share\\finance\\raw_data\\'
root_dir_2 = '/Users/timrawling/Desktop/Projects/Finance/stock_analysis/raw_data/FX-EUR-USD/

file_last = 'FX-EUR-USD/EURUSD_T_LAST_012020-082020.csv'
file_bid = 'FX-EUR-USD/EURUSD_T_BID_012020-082020.csv'
file_ask = 'FX-EUR-USD/EURUSD_T_ASK_012020-082020.csv'

total_start_time = time.time()

#EURUSD_T_LAST_FX = pd.read_csv(root_dir + file_last, index_col = 0, parse_dates = ['Date'])
EURUSD_T_BID_FX = pd.read_csv(root_dir + file_bid, index_col = 0, parse_dates = ['Date'])
EURUSD_T_ASK_FX = pd.read_csv(root_dir + file_ask, index_col = 0, parse_dates = ['Date'])

#LAST_FX_data = EURUSD_T_LAST_FX.loc[start_datetime:end_datetime]
BID_FX_data = EURUSD_T_BID_FX.loc[start_datetime:end_datetime]
ASK_FX_data = EURUSD_T_ASK_FX.loc[start_datetime:end_datetime]

# Combine all data into one dataframe

FX_data = pd.merge(BID_FX_data, ASK_FX_data, how='inner', left_index=True, right_index=True)

# Create a 'last' price which is average of bid/ask

Last = []

for i in range(0, len(FX_data['Bid'])):
    Last.append((FX_data['Bid'][i] + FX_data['Ask'][i])/2)

FX_data['Last'] = Last

print(FX_data.head())

print("Data loaded")
print("###########")

# Calculate the macd and macd_sig lines.

# Start an instance of a MACD object

FX_data_MACD = MACD(close = FX_data["Last"])

# Set up new columns in df to hold macd method outputs for macd line and the signal line

FX_data['MACD'] = FX_data_MACD.macd()
FX_data['MACD_sig'] = FX_data_MACD.macd_signal()

# Run the macd line and signal line though the cross-over trigger function

MACD_crossover_signal = MACD_crossover_trigger(FX_data.Last, FX_data['MACD'], FX_data['MACD_sig'])

MACD_buy_sig = MACD_crossover_signal[0]
MACD_sell_sig = MACD_crossover_signal[1]

length = len(MACD_buy_sig)

for i in range(0, length):

    if str(MACD_buy_sig[i]) == 'nan':
        MACD_buy_sig[i] = 0
    else:
        pass

    if str(MACD_sell_sig[i]) == 'nan':
        MACD_sell_sig[i] = 0
    else:
        pass

# Set up RSI calculation

FX_data_RSI = RSIIndicator(close = FX_data["Last"])

RSI_line = FX_data_RSI.rsi()

RSI_indicator_buy_trigger = RSI_indicator_trigger(RSI_line)[0]
RSI_indicator_sell_trigger = RSI_indicator_trigger(RSI_line)[1]

# Set up a restricted buy signal that considers the RSI and the MACD cross-over triggers

buy_prices = []
buy_sell_signal = []
sell_prices = []

for i in range(0, len(RSI_indicator_buy_trigger)):

    # Classify buy and sell signals

    if MACD_buy_sig[i] != 0 and RSI_indicator_buy_trigger[i] == 1:

        buy_prices.append(MACD_buy_sig[i])
        buy_sell_signal.append(1)

    elif MACD_sell_sig[i] != 0 and RSI_indicator_sell_trigger[i] == 1:

        sell_prices.append(MACD_sell_sig[i])
        buy_sell_signal.append(-1)

    else:
        sell_prices.append(np.NAN)
        buy_prices.append(np.NAN)
        buy_sell_signal.append(np.NAN)

# Run signals through 'walk forward' algorithim for first cut at profitability. Note that this function looks forward to 
# check what profits would have been if trade was executed...

FX_data["action_signals"] = buy_sell_signal

total_WF_start_time = time.time()   
    
trade_results = walk_forward_bid_ask(FX_data.Bid, FX_data.Ask, buy_sell_signal, slippage = 4, stop = 10)

total_WF_time = time.time() - total_WF_start_time
total_time = time.time() - total_start_time

print("Walk forward processing time took: " + str(total_WF_time))
print("Total processing time took: " + str(total_time))

# Plot trade actions across period

total_charting_start_time = time.time()  

fig = plt.figure(constrained_layout=False, figsize = (13.8, 4.5))
grid_spec = fig.add_gridspec(nrows = 8, ncols = 1)

ax1 = fig.add_subplot(grid_spec[0:4,0])
ax1.set_title('EUR-USD MACD X-OVER STRAT')

ax2 = fig.add_subplot(grid_spec[4:6,0], sharex = ax1)
ax3 = fig.add_subplot(grid_spec[6:8,0], sharex = ax1)

# Price and buy/sell signal chart
ax1.scatter(FX_data.index, buy_prices, color = 'green', label = 'Buy', marker = '^', alpha = 1)
ax1.scatter(FX_data.index, sell_prices, color = 'red', label = 'Sell', marker = 'v', alpha = 1)
ax1.plot(FX_data.Last, label = 'Last Price', linewidth = 1, alpha = 0.5)
ax1.legend(loc = 'upper right', prop={"size":7.5})

# MACD Chart
ax2.plot(FX_data.index, FX_data['MACD'], label = 'EUR-USD T MACD', color = 'red', linewidth = 1, alpha = 0.5)
ax2.plot(FX_data.index, FX_data['MACD_sig'], label = 'EUR-USD T Signal', color = 'blue', linewidth = 1, alpha = 0.5)
ax2.legend(loc = 'upper right', prop={"size":7.5})

# RSI Chart
ax3.plot(FX_data.index, RSI_line, label = 'EUR-USD T RSI', color = 'black', linewidth = 1, alpha = 0.5)
ax3.hlines(70, min(FX_data.index), max(FX_data.index), colors='red', linewidth = 1)
ax3.hlines(30, min(FX_data.index), max(FX_data.index), colors='green', linewidth = 1)

plt.setp(ax1.get_xticklabels(), visible = False)
plt.setp(ax2.get_xticklabels(), visible = False)

plt.show()

total_charting_time = time.time() - total_charting_start_time
print("Total charting time took: " + str(total_charting_time))