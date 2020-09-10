import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import pandas as pd
from FOREX_ML_PPT_MASTER import *

# Pull and cut down data

start_datetime = '2020-08-10 00:00:00'
end_datetime = '2020-08-20 00:00:00'

file = '/Users/timrawling/Desktop/Projects/Finance/stock_analysis/raw_data/FX-USD-JPY/USDJPY_M1_2015-082020.csv'

total_start_time = time.time()

USDJPY_FX = pd.read_csv(file, index_col=0, parse_dates=['Date'])
USDJPY_FX = USDJPY_FX.loc[start_datetime:end_datetime]

USDJPY_FX.drop_duplicates(subset='OPEN', inplace = False)

print(USDJPY_FX.head())

# Use this variable to run different FX pairs

df = USDJPY_FX

#print(df.head())

# Calculate moving average

ma_30 = df.CLOSE.rolling(center = False, window = 30).mean()

# Reference master file function to build heikenashi candles

HA_results = heikenashi(df,[1])
HA = HA_results.candles[1]
#print(HA)

# Reference master file function to detrend the data

#f = sine_coefficient_calculator(df, [10, 15], method = 'difference')

results = wadl(df, [15])

line = results.wadl[15]

print(line.head())