import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from HARMONIC_FUNCTIONS import *
# TQDM allows monitoring of loop progress
from tqdm import tqdm
#=======================================================================================================================
# #This strategy uses the concept of FX price movement
#harmonics to identify certain patterns and set buy/sell signals based on this
#=======================================================================================================================

# Import data and restrict time considered

start_datetime = '2020-01-20 00:00:00'
end_datetime = '2020-08-20 16:00:00'

file = '/Users/timrawling/Desktop/Projects/Finance/stock_analysis/raw_data/FX-AUD-USD/AUDUSD_M1_2015-082020.csv'

AUDUSD_FX = pd.read_csv(file, index_col=0, parse_dates=['Date'])
AUDUSD_FX = AUDUSD_FX.loc[start_datetime:end_datetime]

AUDUSD_FX.drop_duplicates(subset='OPEN', inplace = False)

print(AUDUSD_FX.head())
print(AUDUSD_FX.shape)

price = AUDUSD_FX['CLOSE']

err_allowed = 0.1

# As an experiment, set up a variable that captures gains/losses. It considers 15 time segments post trade.
# Can't use this in the strategy as this would constitute data snooping bias.

#pips = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

pnl = []
trade_datetimes = []
correct_pats = 0
pats = 0

plt.ion()

for i in tqdm(range(100, len(price.values))):

    current_idx, current_pat, start, end = peak_detector(price.values[:i], order = 10)

    # So now create the pattern requirements  to check the items against to find a harmonic (up-down-up-down).

    XA = current_pat[1] - current_pat[0]
    AB = current_pat[2] - current_pat[1]
    BC = current_pat[3] - current_pat[2]
    CD = current_pat[4] - current_pat[3]

    moves = [XA, AB, BC, CD]

    # Reference functions script for functions for each pattern. Only difference is the range ratios.

    gart_result = gartly_iden(moves, err_allowed)
    butt_result = butterfly_iden(moves, err_allowed)
    bat_result = bat_iden(moves, err_allowed)
    crab_result = crab_iden(moves, err_allowed)

    # Create an array of results.

    harmonics = np.array([gart_result, butt_result, bat_result, crab_result])
    labels = ['gartly', 'butterfly', 'bat', 'crab']

    if np.any(harmonics == 1) or np.any(harmonics == -1):

        pats += 1

        for j in range(0, len(harmonics)):

            if harmonics[j] == 1 or harmonics[j] == -1:

                sense = 'Bearish ' if harmonics[j] == -1 else 'Bullish '
                label = sense + labels[j] + ' found'
                #print(label)

                # So now collect the datetime of the trades as they happen

                start = np.array(current_idx).min()
                end = np.array(current_idx).max()
                trade_datetime = AUDUSD_FX.iloc[end].name

                trade_datetimes.append(trade_datetime)

                pips = walk_forward(price.values[end:], harmonics[j], slippage = 4, stop = 25)

                pnl = np.append(pnl,pips)

                cum_pips = pnl.cumsum()

                #print(cum_pips[-1])

                if pips > 0:

                    correct_pats += 1

                lbl = 'Accuracy ' + str(100*float(correct_pats)/float(pats)) + '%'

                plt.clf()

                fig = plt.figure(figsize=(13.8, 4.5))

                ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
                ax2 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)

                ax1.plot(np.arange(start, i + 100), price.values[start:i + 100])
                ax1.plot(current_idx, current_pat, color='red', linewidth=1)

                ax2.plot(cum_pips, label = lbl)
                ax2.legend()
                #plt.show(block=False)
                plt.pause(0.05)
                #plt.close('all')
                """
                if harmonics[j] == 1:
                    count = 1
                    for k in pips:
                        k += 1000*(price[end + count]-price[end])
                        pips[count - 1] = k
                        count += 1

                elif harmonics[j] == -1:
                    count = 1
                    for k in pips:
                        k += 1000*(price[end] - price[end + count])
                        pips[count - 1] = k
                        count += 1

                print(pips)

                # Create a dynamic bar chart of gains/losses depending on exit point
                #plt.clf()
                #plt.bar(np.arange(1,16), pips)
                #plt.show(block=False)
                #plt.pause(0.05)
                #plt.close('all')

                # Now plot the found harmonic pattern
                #plt.title(label)
                #plt.plot(np.arange(start, i + 100), price.values[start:i + 100])
                #plt.plot(current_idx, current_pat, color='red', linewidth=1)
                #plt.show(block=False)
                #plt.pause(0.025)
                #plt.close('all')
                """