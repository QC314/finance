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

start_datetime = '2020-06-19 06:00:00'
end_datetime = '2020-06-19 08:00:00'

root_dir = 'D:\\network_share\\finance\\raw_data\\'
file = 'FX-EUR-USD/EURUSD_T_012020-082020.csv'

total_start_time = time.time()

EURUSD_T_FX = pd.read_csv(root_dir + file, index_col = 0, parse_dates = ['Date'])
#EURUSD_T_FX = pd.read_csv(root_dir + file)
print("Data loaded")

EURUSD_T_FX = EURUSD_T_FX.loc[start_datetime:end_datetime]

#pd.to_datetime(EURUSD_T_FX.index)

print(EURUSD_T_FX.head())

#input()


# Calculate the macd and macd_sig lines.

# Start an instance of a MACD object

EURUSD_T_FX_MACD = MACD(close = EURUSD_T_FX["Last"])

# Set up new columns in df to hold macd method outputs for macd line and the signal line

EURUSD_T_FX['MACD'] = EURUSD_T_FX_MACD.macd()
EURUSD_T_FX['MACD_sig'] = EURUSD_T_FX_MACD.macd_signal()

# Run the macd line and signal line though the cross-over trigger function

MACD_crossover_signal = MACD_crossover_trigger(EURUSD_T_FX.Last, EURUSD_T_FX['MACD'], EURUSD_T_FX['MACD_sig'])

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

# Run the macd line inflection point trigger function

# ------- THIS IS WORK IN PROGRESS -------------------------------------------------------------------------------------

#MACD_inflection_trigger = MACD_inflection_trigger(EURUSD_T_FX.Last, EURUSD_T_FX['MACD'])

# ----------------------------------------------------------------------------------------------------------------------

# Set up RSI calculation

EURUSD_T_FX_RSI = RSIIndicator(close = EURUSD_T_FX["Last"])

RSI_line = EURUSD_T_FX_RSI.rsi()

#print(RSI_line.tolist())

RSI_indicator_buy_trigger = RSI_indicator_trigger(RSI_line)[0]
RSI_indicator_sell_trigger = RSI_indicator_trigger(RSI_line)[1]

# Set up a restricted buy signal that considers the RSI and the MACD cross-over triggers

buy_prices = []
sell_prices = []

#SOMETHING IS WRONG HERE - NOT WORKING ON CHART.....
count = 0

for i in zip(RSI_indicator_buy_trigger, RSI_indicator_sell_trigger, MACD_buy_sig, MACD_sell_sig):

    # Classify buy signals

    if MACD_buy_sig[count] != 0 and RSI_indicator_buy_trigger[count] == 1:

        buy_prices.append(MACD_buy_sig[count])

    else:

        buy_prices.append(np.NAN)

    # Classify sell signals

    if MACD_sell_sig[count] != 0 and RSI_indicator_sell_trigger[count] == 1:

        sell_prices.append(MACD_sell_sig[count])

    else:

        sell_prices.append(np.NAN)

    count += 1


# Run the MW pattern trigger function to get pattern buy/sell signal

#MW_pattern_signals = general_harmonic_signal(EURUSD_T_FX, err_allowed = 0.1, order = 5)
#MW_action_price_patterns = MW_pattern_signals[0]
#MW_action_prices = MW_pattern_signals[1]
#MW_harmonic_signals = MW_pattern_signals[2]

fig = plt.figure(constrained_layout=False, figsize = (13.8, 4.5))
grid_spec = fig.add_gridspec(nrows = 8, ncols = 1)

ax1 = fig.add_subplot(grid_spec[0:4,0])
ax1.set_title('EUR-USD MACD X-OVER STRAT')

ax2 = fig.add_subplot(grid_spec[4:6,0], sharex = ax1)
ax3 = fig.add_subplot(grid_spec[6:8,0], sharex = ax1)

# Price and buy/sell signal chart
#ax1.scatter(EURUSD_T_FX.index, MACD_inflection_trigger[0], color = 'yellow', label = 'buy', marker = '^', alpha = 1)
#ax1.scatter(EURUSD_T_FX.index, MACD_inflection_trigger[0], color = 'blue', label = 'sell', marker = 'v', alpha = 1)
ax1.scatter(EURUSD_T_FX.index, buy_prices, color = 'green', label = 'Buy', marker = '^', alpha = 1)
ax1.scatter(EURUSD_T_FX.index, sell_prices, color = 'red', label = 'Sell', marker = 'v', alpha = 1)
ax1.plot(EURUSD_T_FX.Last, label = 'Last Price', linewidth = 1, alpha = 0.5)
ax1.legend(loc = 'upper right', prop={"size":7.5})

# MACD Chart

ax2.plot(EURUSD_T_FX.index, EURUSD_T_FX['MACD'], label = 'EUR-USD T MACD', color = 'red', linewidth = 1, alpha = 0.5)
ax2.plot(EURUSD_T_FX.index, EURUSD_T_FX['MACD_sig'], label = 'EUR-USD T Signal', color = 'blue', linewidth = 1, alpha = 0.5)
ax2.legend(loc = 'upper right', prop={"size":7.5})

# RSI Chart
ax3.plot(EURUSD_T_FX.index, RSI_line, label = 'EUR-USD T RSI', color = 'black', linewidth = 1, alpha = 0.5)
ax3.hlines(70, min(EURUSD_T_FX.index), max(EURUSD_T_FX.index), colors='red', linewidth = 1)
ax3.hlines(30, min(EURUSD_T_FX.index), max(EURUSD_T_FX.index), colors='green', linewidth = 1)

plt.setp(ax1.get_xticklabels(), visible = False)
plt.setp(ax2.get_xticklabels(), visible = False)

plt.show()

#block = False