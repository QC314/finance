import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib.finance import candlestick
from matplotlib.dates import date2num
import pandas as pd
import time
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
from sklearn.linear_model import LinearRegression
from datetime import datetime
import warnings
import math

plt.style.use('fivethirtyeight')

class holder:
    1

def heikenashi(prices, periods):

    # =============================================== DESCRIPTION ======================================================

    # Heiken Ashi candelesticks - altered candlestick that capture momentum

    # Param (1) - prices: dataframe of OHLC & volume data
    # Param (2) - periods: periods for which to create the candles
    # return (1) - heikenashi OHLC candles for chosen period as a dictionary of dataframes

    results = holder()
    dict = {}

    HA_close = prices[['OPEN', 'HIGH', 'LOW', 'CLOSE']].sum(axis = 1)/4
    HA_open = HA_close.copy()
    HA_high = HA_close.copy()
    HA_low = HA_close.copy()

    for i in range(1, len(prices)):

        HA_open.iloc[i] = (HA_open.iloc[i - 1]+HA_close.iloc[i - 1])/2
        HA_high.iloc[i] = np.array([prices.HIGH.iloc[i], HA_open.iloc[i], HA_close.iloc[i]]).max()
        HA_high.iloc[i] = np.array([prices.LOW.iloc[i], HA_open.iloc[i], HA_close.iloc[i]]).min()

    df = pd.concat((HA_open, HA_high, HA_low, HA_close), axis = 1)
    df.columns = [['HA_open', 'HA_high', 'HA_low', 'HA_close']]

    dict[periods[0]] = df

    results.candles = dict
    return results

def detrender(prices, method = 'difference'):

    # =============================================== DESCRIPTION ======================================================

    # Detrend price data using 'linear' or difference 'method'

    # Param (1) - prices: dataframe of OHLC & volume data
    # Param (2) - method: method by which to detrend the data, 'linear' or 'difference'
    # return (1) - the detrended price series

    if method == 'difference':

        detrended = prices.CLOSE[1:] - prices.CLOSE[:-1].values

    elif method == 'linear':

        x =np.arange(0, len(prices))
        y = prices.CLOSE.values

        # Make the linear regression line to fit between CLOSE price points. Note that need to reshape the values to
        # suit required input for sklearn. Then we reshape it back to a 1D array for processing

        model = LinearRegression()
        model.fit(x.reshape(-1,1), y.reshape(-1,1))

        trend = model.predict(x.reshape(-1,1))
        trend = trend.reshape((len(prices),))

        # Take the difference between 'our' y and the linear model's y to detrend the price

        detrended = prices.CLOSE - trend

    else:

        print('You did not input a valid method for detrending. Available options are linear and difference')

    return detrended

def fourier_series(x, a0, a1, b1, w):

    # =============================================== DESCRIPTION ======================================================

    # Fourier function generator

    # Param (1) - x: time period (independant variable)
    # Param (2) - a0: first fourier series coefficient
    # Param (3) - a1: second fourier series coefficient
    # Param (4) - b1: third fourier series coefficient
    # Param (5) - w: fourier series frequency
    # return (1) - F, the value of the fourier function as time x

    f = a0 + a1 * np.cos(w*x) + b1 * np.sin(w*x)

    return f

def sine_series(x, a0, b1, w):

    # =============================================== DESCRIPTION ======================================================

    # Sine function generator

    # param (1) - x: time period (independant variable)
    # param (2) - a0: first sine series coefficient
    # param (3) - b1: second sine series coefficient
    # param (4) - w: sine series frequency
    # return (1) - F, the value of the fourier function as time x

    f = a0 + b1 * np.sin(w*x)

    return f

def fourier_coefficient_calculator(prices, periods, method = 'difference'):

    # =============================================== DESCRIPTION ======================================================

    # Function to fit the general fourier function to price data

    # param (1) - prices: OHLC candlestick data as a df
    # param (2) - periods: list of periods for which to fit the fourier series. This is the window of fit consideration (2, 5, 10 etc.)
    # param (3) - method: method by which to detrend the data
    # return (1) - dictionary of dataframes containing the coefficients of the fitted fourier equation for the period

    results = holder()
    dict = {}

    # Set an option to plot the expansion fit for each iteration

    plot = True

    # Compute the coefficients of the series. First thing to do is to detrend the price data

    detrended = detrender(prices, method)

    for i in range(0, len(periods)):

        coeffs =[]

        for j in range(periods[i], len(prices) - periods[i]):

            x = np.arange(0, periods[i])
            # y is the shifting window for consideration as we move through j, periods[i] is the length of the window
            y = detrended.iloc[j - periods[i]:j]

            # this is a filter to catch any error created by the scipy.optimize package that we're using to fit the curves.
            # It is likely to spit an error when it can't fit the curves, but we don't want this to kill the program.
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)

                try:

                    res = scipy.optimize.curve_fit(fourier_series, x, y)

                except (RuntimeError, OptimizeWarning):

                    #size set to match the shape of what we're returning, which is [a0, a1, b1, w], 1 x 4

                    res = np.empty((1,4))
                    res[0:] = np.NAN

            if plot == True:

                xt = np.linspace(0, periods[i], 100)
                yt = fourier_series(xt, res[0][0], res[0][1], res[0][2], res[0][3])

                plt.plot(x,y, 'b')
                plt.plot(xt, yt, 'r')
                plt.show()

            coeffs = np.append(coeffs, res[0], axis = 0)

        # Filter out VDW for numpy....
        warnings.filterwarnings('ignore', category = np.VisibleDeprecationWarning)

        coeffs = np.array(coeffs).reshape(((len(coeffs)/4, 4)))

        df = pd.DataFrame(coeffs, index = prices.iloc[periods[i]:-periods[i]])

        df.columns = [['a0', 'a1', 'b1', 'w']]

        # for each NAN value encountered, the below will replace it with the closest values from above/below it
        df = df.fillna(method = 'bfill')

        dict[periods[i]] = df

    results.coeffs = dict

    return results

def sine_coefficient_calculator(prices, periods, method = 'difference'):

    # =============================================== DESCRIPTION ======================================================

    # Function to fit the general sine function to price data

    # param (1) - prices: OHLC candlestick data as a df
    # param (2) - periods: list of periods for which to fit the sine series. This is the window of fit consideration (2, 5, 10 etc.)
    # param (3) - method: method by which to detrend the data
    # return (1) - dictionary of dataframes containing the coefficients of the fitted sine equation for the period

    results = holder()
    dict = {}

    # Set an option to plot the expansion fit for each iteration

    plot = True

    # Compute the coefficients of the series. First thing to do is to detrend the price data

    detrended = detrender(prices, method)

    for i in range(0, len(periods)):

        coeffs =[]

        for j in range(periods[i], len(prices) - periods[i]):

            x = np.arange(0, periods[i])
            # y is the shifting window for consideration as we move through j, periods[i] is the length of the window
            y = detrended.iloc[j - periods[i]:j]

            # this is a filter to catch any error created by the scipy.optimize package that we're using to fit the curves.
            # It is likely to spit an error when it can't fit the curves, but we don't want this to kill the program.
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)

                try:

                    res = scipy.optimize.curve_fit(sine_series, x, y)

                except (RuntimeError, OptimizeWarning):

                    #size set to match the shape of what we're returning, which is [a0, b1, w], 1 x 4

                    res = np.empty((1,3))
                    res[0:] = np.NAN

            if plot == True:

                xt = np.linspace(0, periods[i], 100)
                yt = sine_series(xt, res[0][0], res[0][1], res[0][2])

                plt.plot(x,y, 'b')
                plt.plot(xt, yt, 'r')
                plt.show()

            coeffs = np.append(coeffs, res[0], axis = 0)

        # Filter out VDW for numpy....
        warnings.filterwarnings('ignore', category = np.VisibleDeprecationWarning)

        coeffs = np.array(coeffs).reshape(((len(coeffs)/3, 3)))

        df = pd.DataFrame(coeffs, index = prices.iloc[periods[i]:-periods[i]])

        df.columns = [['a0', 'b1', 'w']]

        # for each NAN value encountered, the below will replace it with the closest values from above/below it
        df = df.fillna(method = 'bfill')

        dict[periods[i]] = df

    results.coeffs = dict

    return results

def wadl(prices, periods):

    # =============================================== DESCRIPTION ======================================================

    # Williams Accumulation Distribution which acts as supply/demand indicator

    #### NOTE THIS HAS NOT BEEN CHECKED GRAPHICALLY YET, AS DON"T CURRENTLY DON'T HAVE VOLUME VALUES

    # Param (1) - prices: dataframe of OHLC & volume data
    # Param (2) - periods: periods for which to calculate the function for varying period lengths (5, 10, 15,...)
    # return (1) - WALD lines for each period

    results = holder()
    dict = {}

    for i in range(0, len(periods)):

        WAD = []

        for j in range(periods[i], len(prices) - periods[i]):

            # Calculate the true range high & low - a sub definition of the WADL

            TR_high = np.array([prices.HIGH.iloc[j],prices.CLOSE.iloc[j-1]]).max()
            TR_low = np.array([prices.LOW.iloc[j], prices.CLOSE.iloc[j - 1]]).min()

            # Below is the logic for the price move calc, as per WADL calc

            if prices.CLOSE.iloc[j] > prices.CLOSE.iloc[j-1]:

                PM = prices.CLOSE.iloc[j] - TR_low

            elif prices.CLOSE.iloc[j] < prices.CLOSE.iloc[j-1]:

                PM = prices.CLOSE.iloc[j] - TR_high

            elif prices.CLOSE.iloc[j] ==prices.CLOSE.iloc[j-1]:

                PM = 0

            else:

                print('Unknown error ocurred, check the WADL calc...')

            # Now calculate the accumulation distribution

            AD = PM * prices.Volume.iloc[j]

            WAD = np.append(WAD, AD)

        WAD = WAD.cumsum()

        WAD = pd.DataFrame(WAD, index = prices.iloc[periods[i]:-periods[i]].index)

        WAD.columns = [['close']]

        dict[periods[i]] = WAD

    results.wadl = dict

    return results

def OHLC_resample(dataframe, timeframe, column = 'ask'):

    # =============================================== DESCRIPTION ======================================================

    # OHLC Resampler to turn raw data into candlestick data if required, and to switch timeframes, e.g. from 1M to 1H.

    # param (1) - dataframe: dataframe containing the data we want to resample.
    # param (2) - timeframe: timeframe that we want for resample.
    # param (3) - column: which column we are resampling (bid or ask), default set to ask.
    # return (1) - resampled OHLC data for the given timeframe.

    grouped = dataframe.groupedby('Symbol')

    # The below is to turn raw data into candlestick format. This will be required if tick data

    if np.any(dataframe.columns == 'Ask'):

        if column == 'ask':
            ask = grouped['Ask'].resample(timeframe).ohlc()
            ask_vol = grouped['Ask_Vol'].resample(timeframe).count()
            resampled = pd.dataframe(ask)
            resampled['Ask_Vol'] = ask_vol

        elif column == 'bid':
            bid = grouped['Bid'].resample(timeframe).ohlc()
            bid_vol = grouped['Bid_Vol'].resample(timeframe).count()
            resampled = pd.dataframe(bid)
            resampled['Bid_Vol'] = bid_vol

        else:

            raise ValueError('Column must be a strong. Either ask or bid')

    # Now resample data that us already in 'candlestick' format (OHLC). To a higher time period

    elif np.any(dataframe.columns == 'CLOSE'):
        open = grouped['OPEN'].resample(timeframe).ohlc()
        close = grouped['CLOSE'].resample(timeframe).ohlc()
        high = grouped['HIGH'].resample(timeframe).ohlc()
        low = grouped['LOW'].resample(timeframe).ohlc()
        ask_vol = grouped['Volume'].resample(timeframe).count()

        resampled = pd.Dataframe(open)
        resampled['HIGH'] = high
        resampled['LOW'] = low
        resampled['CLOSE'] = close
        resampled['Volume'] = ask_vol

    resampled = resampled.dropna()

    return resampled

def MACD(prices, periods):


def pattern_finder():

# Incomplete from below

def momentum(prices, periods):

    # =============================================== DESCRIPTION ======================================================

    # Calculates the momentum of price movements

    # param (1) - prices: dataframe of OHLC data.
    # param (2) - periods: list of periods to calculate function over
    # return (1) - momentum indicator

    results = holder()
    open = {}
    close = {}

    for i in range(0, len(periods)):

        open[periods[i]] = pd.DataFrame(prices.OPEN.iloc[periods[i]:]-prices.OPEN.iloc[:-periods[i]].values,
                                        index = prices.iloc[periods[i]:].index)

        close[periods[i]] = pd.DataFrame(prices.CLOSE.iloc[periods[i]:] - prices.CLOSE.iloc[:-periods[i]].values,
                                        index=prices.iloc[periods[i]:].index)

        open[periods[i]].columns = [['open']]
        open[periods[i]].columns = [['close']]

    results.open = open
    results.close = close

    return results