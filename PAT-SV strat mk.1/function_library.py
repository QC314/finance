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
from scipy.signal import argrelextrema
from scipy.interpolate import UnivariateSpline



# Mathematical and general functions

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

    s = a0 + b1 * np.sin(w*x)

    return s

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

    # Now resample data that is already in 'candlestick' format (OHLC). To a higher time period

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

def curvature_splines(x, y=None, error=0.001):

    """Calculate the signed curvature of a 2D curve at each point
    using interpolating splines.
    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )
         or
         y=None and
         x is a numpy.array(dtype=complex) shape (n_points, )
         In the second case the curve is represented as a np.array
         of complex numbers.
    error : float
        The admisible error when interpolating the splines
    Returns
    -------
    curvature: numpy.array shape (n_points, )
    Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
    but more accurate, especially at the borders.
    """

    # handle list of complex case

    if y is None:
        x, y = x.real, x.imag

    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

    xˈ = fx.derivative(1)(t)
    xˈˈ = fx.derivative(2)(t)
    yˈ = fy.derivative(1)(t)
    yˈˈ = fy.derivative(2)(t)
    curvature = (xˈ* yˈˈ - yˈ* xˈˈ) / np.power(xˈ** 2 + yˈ** 2, 1.5)
    return curvature

# Pattern identification and trend fitting

def peak_set_detection(prices, order=10):

    # =============================================== DESCRIPTION ======================================================

    # Peak detector

    # param (1) - prices: close prices for use for detecting past peaks
    # param (2) - order: number of adjacent points to consider in peak calculations
    # return (1) - 'current_idx': index of the close prices for the last 5 peaks identified in the pricing sequence.
    # return (2) - 'current_price_pat': actual close prices for the last 5 peaks identified in the pricing sequence.
    # return (3) - start_index: index of prices df of the first peak in consideration
    # return (4) - end_index: index of prices df of the lest peak in consideration, which is also the current price.
    # return (4) - peak_pattern_movements: price movements magnitude for each key movement in the identified pattern.

    max_idx = list(argrelextrema(prices.values, np.greater, order=order)[0])
    min_idx = list(argrelextrema(prices.values, np.less, order=order)[0])

    # Make sure that the indexes for the peaks include the last value. We don't know if today's point will be a peak,
    # so we'll just have to assume that it is, and see if it fits the harmonic patterns anyway

    idx = max_idx + min_idx + [len(prices) - 1]
    idx.sort()

    current_idx = idx[-5:]

    start = min(current_idx)
    end = max(current_idx)

    current_price_pat = prices[current_idx]

    XA = current_price_pat[1] - current_price_pat[0]
    AB = current_price_pat[2] - current_price_pat[1]
    BC = current_price_pat[3] - current_price_pat[2]
    CD = current_price_pat[4] - current_price_pat[3]

    moves = [XA, AB, BC, CD]

    peak_indexes = current_idx
    peak_prices = current_price_pat
    peak_start_inx = start
    peak_end_inx = end

    return peak_indexes, peak_prices, peak_start_inx, peak_end_inx

def fourier_coefficient_calculator(prices, periods, method='difference'):

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

        coeffs = []

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

                    # size set to match the shape of what we're returning, which is [a0, a1, b1, w], 1 x 4

                    res = np.empty((1, 4))
                    res[0:] = np.NAN

            if plot == True:
                xt = np.linspace(0, periods[i], 100)
                yt = fourier_series(xt, res[0][0], res[0][1], res[0][2], res[0][3])

                plt.plot(x, y, 'b')
                plt.plot(xt, yt, 'r')
                plt.show()

            coeffs = np.append(coeffs, res[0], axis=0)

        # Filter out VDW for numpy....
        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        coeffs = np.array(coeffs).reshape(((len(coeffs) / 4, 4)))

        df = pd.DataFrame(coeffs, index=prices.iloc[periods[i]:-periods[i]])

        df.columns = [['a0', 'a1', 'b1', 'w']]

        # for each NAN value encountered, the below will replace it with the closest values from above/below it
        df = df.fillna(method='bfill')

        dict[periods[i]] = df

    results.coeffs = dict

    return results

def sine_coefficient_calculator(prices, periods, method='difference'):
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

        coeffs = []

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

                    # size set to match the shape of what we're returning, which is [a0, b1, w], 1 x 4

                    res = np.empty((1, 3))
                    res[0:] = np.NAN

            if plot == True:
                xt = np.linspace(0, periods[i], 100)
                yt = sine_series(xt, res[0][0], res[0][1], res[0][2])

                plt.plot(x, y, 'b')
                plt.plot(xt, yt, 'r')
                plt.show()

            coeffs = np.append(coeffs, res[0], axis=0)

        # Filter out VDW for numpy....
        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        coeffs = np.array(coeffs).reshape(((len(coeffs) / 3, 3)))

        df = pd.DataFrame(coeffs, index=prices.iloc[periods[i]:-periods[i]])

        df.columns = [['a0', 'b1', 'w']]

        # for each NAN value encountered, the below will replace it with the closest values from above/below it
        df = df.fillna(method='bfill')

        dict[periods[i]] = df

    self.coeffs = dict

    return results

def detrender(prices, method='difference'):
    # =============================================== DESCRIPTION ======================================================

    # Detrend price data using 'linear' or difference 'method'

    # Param (1) - prices: dataframe of OHLC & volume data
    # Param (2) - method: method by which to detrend the data, 'linear' or 'difference'
    # return (1) - the detrended price series

    if method == 'difference':

        detrended = prices.CLOSE[1:] - prices.CLOSE[:-1].values

    elif method == 'linear':

        x = np.arange(0, len(prices))
        y = prices.CLOSE.values

        # Make the linear regression line to fit between CLOSE price points. Note that need to reshape the values to
        # suit required input for sklearn. Then we reshape it back to a 1D array for processing

        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

        trend = model.predict(x.reshape(-1, 1))
        trend = trend.reshape((len(prices),))

        # Take the difference between 'our' y and the linear model's y to detrend the price

        detrended = prices.CLOSE - trend

    else:

        print('You did not input a valid method for detrending. Available options are linear and difference')

    return detrended

# Trigger functions

def MACD_crossover_trigger(prices, MACD_line, MACD_sig):

    buy = []
    sell = []
    flag = -1

    for i in range(0, len(prices)):

        if MACD_line[i] > MACD_sig[i]:
            sell.append(np.nan)

            if flag != 1:
                buy.append(prices[i])
                flag = 1

            else:
                buy.append(np.nan)

        elif MACD_line[i] < MACD_sig[i]:
            buy.append(np.nan)

            if flag != 0:
                sell.append(prices[i])
                flag = 0

            else:
                sell.append(np.nan)

        else:
            buy.append(np.nan)
            sell.append(np.nan)

    return buy, sell

def MACD_inflection_trigger(prices, MACD_line, smoothing = 2):

    # =============================================== DESCRIPTION ======================================================

    #######--------------  WIP  --------------

    # Use the last three MACD points to establish the inflection point

    # param (1) - prices: dataframe of OHLC & volume data
    # param (2) - MACD_line: needs to be previously calculated
    # return (1) - buy and sell signals/price

    buy = []
    sell = []

    for i in range(0, len(prices)):

        # Skip NANs as no MACD calculated for those points:

        if np.isnan(MACD_line[i]):

            buy.append(np.nan)
            sell.append(np.nan)

        else:

            # Create 'smoothed' rates of change for consideration
            try:

                sm_roc_1 = sum(MACD_line[i - smoothing:i]) / len(MACD_line[i - smoothing:i])
                sm_roc_2 = sum(MACD_line[i - 2 * smoothing:i - smoothing]) / len(MACD_line[i - 2 * smoothing:i - smoothing])
                sm_roc_3 = sum(MACD_line[i - 2 * smoothing - 3 * smoothing:i]) / len(MACD_line[i - 2 * smoothing:i - 3 * smoothing])
                sm_roc_4 = sum(MACD_line[i - 3 * smoothing - 4 * smoothing:i]) / len(MACD_line[i - 3 * smoothing:i - 4 * smoothing])

            except:

                buy.append(np.nan)
                sell.append(np.nan)

                continue

            # If it is, then check the direction of rotation (i.e. d2y/dx2 > 0, meaning downward inf. and sell signal,
            # or d2y/dx2 < 0, meaning upward inflection and buy signal.

            if (sm_roc_1 - sm_roc_2) > 0 and (sm_roc_3 - sm_roc_4) < 0:

                sell.append(prices[i])
                buy.append(np.nan)

            elif (sm_roc_1 - sm_roc_2) < 0 and (sm_roc_3 - sm_roc_4) > 0:

                buy.append(prices[i])
                sell.append(np.nan)

            else:
                buy.append(np.nan)
                sell.append(np.nan)

    return buy, sell

def RSI_indicator_trigger(RSI_line, upper_bound = float(70), lower_bound = float(30)):

    buy = []
    sell = []

    for i in range(0, len(RSI_line)):

        if str(RSI_line[i]) == 'NaN':

            buy.append(np.NAN)
            sell.append(np.NAN)

        elif RSI_line[i] > upper_bound:

            sell.append(1)
            buy.append(np.NAN)

        elif RSI_line[i] < lower_bound:

            buy.append(1)
            sell.append(np.NAN)

        else:
            buy.append(np.NAN)
            sell.append(np.NAN)

    return buy, sell

def general_harmonic_signal(prices, err_allowed = 0.1, use_custom_pattern_ranges = False, pattern_ranges = [0.618, 0.618, 0.382, 0.886, 1.27, 1.618], order = 10, start_point = 100):

    # =============================================== DESCRIPTION ======================================================

    # General harmonic pattern identifier

    # param (1) - prices: price action indexed by date as a dataframe
    # param (2) - error allowed: allowed difference between set pattern weights and found pattern moves.
    # param (3) - pattern_weights: weights/coefficient for moves comparison. Start with pure gartly patterns.
    # param (4) - order: number of adjacent points to consider in peak calculations to feed into peak detector function.

    # return (1) - signal: buy, sell or do nothing signal

    gartly_pattern_ranges = [0.618, 0.618, 0.382, 0.886, 1.27, 1.618]
    butterfly_pattern_ranges = [0.786, 0.786, 0.382, 0.886, 1.618, 2.618]
    bat_pattern_ranges = [0.382, 0.5, 0.382, 0.886, 1.618, 2.618]
    crab_pattern_ranges = [0.382, 0.618, 0.382, 0.886, 1.24, 3.618]

    if use_custom_pattern_ranges is True:
        pattern_ranges = pattern_ranges

    else:
        pattern_ranges = [gartly_pattern_ranges, butterfly_pattern_ranges, bat_pattern_ranges, crab_pattern_ranges]

    patt_count = 1

    action_price_patterns = []
    action_prices = []
    harmonic_signals = []

    # We are starting consideration of pattersn from the 'start_point' therefore we need to add 'start point'
    # numnber NAN entries to the start of our pattern/signal lists

    for k in range(0, start_point):
        action_price_patterns.append(0)
        action_prices.append(np.NAN)
        harmonic_signals.append(np.NAN)

    for j in pattern_ranges:

        for i in range(start_point, len(prices.Last.values)):

            current_idx, current_price_pat, start, end = peak_set_detection(prices.Last[:i], order)

            # So now create the pattern requirements  to check the items against to find a harmonic (up-down-up-down).

            XA = current_price_pat[1] - current_price_pat[0]
            AB = current_price_pat[2] - current_price_pat[1]
            BC = current_price_pat[3] - current_price_pat[2]
            CD = current_price_pat[4] - current_price_pat[3]

            # Create movement ranges for all patterns ranges

            AB_range = np.array([j[0] - err_allowed, j[1] + err_allowed]) * abs(XA)
            BC_range = np.array([j[2] - err_allowed, j[3] + err_allowed]) * abs(AB)
            CD_range = np.array([j[4] - err_allowed, j[5] + err_allowed]) * abs(BC)

            if XA > 0 and AB < 0 and BC > 0 and CD < 0:
                if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(
                        CD) < CD_range[1]:
                    signal = 1

                else:
                    signal = np.NAN

            elif XA < 0 and AB > 0 and BC < 0 and CD > 0:
                if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(
                        CD) < CD_range[1]:
                    signal = -1

                else:
                    signal = np.NAN

            else:
                signal = np.NAN

            if patt_count == 1:

                harmonic_signals.append(signal)


                if signal == 1 or signal == -1:
                    action_price_patterns.append(current_price_pat.values.tolist())
                    action_prices.append(prices.Last[i])
                else:
                    action_price_patterns.append(0)
                    action_prices.append(np.NAN)

            else:

                harmonic_signals[i] = signal

                if action_price_patterns[i] == 0:

                    if signal == 1 or signal == -1:
                        action_price_patterns[i] = current_price_pat.values.tolist()
                        action_prices[i] = prices.Last[i]

                    else:
                        pass
                else:
                    pass

        patt_count += 1

    return action_price_patterns, action_prices, harmonic_signals

def walk_forward(prices, signal, slippage = 4, stop = 10):

    slippage = float(slippage)/float(10000)
    stop_amount = float(stop)/float(10000)

    if signal == 1:

        initial_stop_loss = prices[-1] - stop_amount

        stop_loss = initial_stop_loss

        for i in range(1, len(price)):

            move = prices[i] - prices[i - 1]

            if move > 0 and (prices[i] - stop_amount) > initial_stop_loss:

                stop_loss = prices[i] - stop_amount

            elif prices[i] < stop_loss:

                return stop_loss - prices[i] - slippage

    elif signal == -1:

        initial_stop_loss = prices[-1] + stop_amount

        stop_loss = initial_stop_loss

        for i in range(1, len(prices)):

            move = prices[i] - prices[i-1]

            if move < 0 and (prices[i] + stop_amount) < initial_stop_loss:

                stop_loss = prices[i] + stop_amount

            elif prices[i] > stop_loss:

                return stop_loss - prices[i] - slippage

def walk_forward_2(prices, signal, slippage = 4, stop = 10):

    trade_results = []

    #Translate into pips (1/100th of 1% or 'one basis point')
    slippage = float(slippage)/float(10000)
    stop_amount = float(stop)/float(10000)

    count = 1

    for i in range(1, len(prices)):

        if str(signal[i]) == "nan":
            trade_results.append(0)

        elif signal[i] == 1:
            
            print("Buy trade number " + str(count) + " initiated")

            for j in range(i + 1, len(prices)):

                stop = prices[j-1] - stop_amount

                if prices[j] > stop:
                    
                    count += 1

                elif prices[j] < stop_amount:
                    # If this is the case, then we sell and take the result minus slippage
                    trade_result = price[j] - prices[i] - slippage 
                    trade_results.append(trade_result)
                    print("Buy trade number " + str(count) + " earned " + str(trade_result*10000) + " pips")
                    break                  
                
        elif signal[i] == -1:

            print("Sell trade number " + str(count) + " initiated")

            for j in range(i + 1, len(prices)):

                stop = prices[j-1] + stop_amount

                if prices[j] < stop_amount:
                    pass
                    flag = 1
                    
                elif prices[j] > stop_amount:
                    # If this is the case, then we sell and take the result minus slippage
                    trade_result = -prices[j] + prices[i] - slippage
                    trade_results.append(trade_result)
                    print("Sell trade number " + str(count) + " earned " + str(trade_result*10000) + " pips")
                    break
        count += 1
                      
    return trade_results    