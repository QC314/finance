import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
plt.style.use('fivethirtyeight')
import pandas as pd
import time
from functools import reduce

#pull and cut down data

start_datetime = '2020-08-10 00:00:00'
end_datetime = '2020-08-20 00:00:00'

file = '/Users/timrawling/Desktop/Projects/Finance/stock_analysis/raw_data/FX-USD-JPY/USDJPY_M1_2015-082020.csv'

total_start_time = time.time()

USDJPY_FX = pd.read_csv(file, index_col=0, parse_dates=['Date'])
USDJPY_FX = USDJPY_FX.loc[start_datetime:end_datetime]

USDJPY_FX.drop_duplicates(subset='OPEN', inplace = False)



def graph_price_data(df):

    fig = plt.figure(figsize=(13.8, 4.5))
    ax1 = plt.subplot2grid((40, 40), (0, 0), rowspan=40, colspan=40)

    ax1.plot(df['LOW'])
    ax1.plot(df['HIGH'])

    plt.show()

def percent_delta(start_point, current_point):
    try:
        x = ((current_point-start_point)/abs(start_point)) * 100
        if x == 0.0:
            return 0.000000001
        else:
            return x
    except:
        return 0.000000001

def pattern_storage():

    pattern_start_time = time.time()

    x = len(avg_line) - 60

    y = 31

    while y < x:

        pattern = []

        p1 = percent_delta(avg_line[y - 30], avg_line[y-29])
        p2 = percent_delta(avg_line[y - 30], avg_line[y - 28])
        p3 = percent_delta(avg_line[y - 30], avg_line[y - 27])
        p4 = percent_delta(avg_line[y - 30], avg_line[y - 26])
        p5 = percent_delta(avg_line[y - 30], avg_line[y - 25])
        p6 = percent_delta(avg_line[y - 30], avg_line[y - 24])
        p7 = percent_delta(avg_line[y - 30], avg_line[y - 23])
        p8 = percent_delta(avg_line[y - 30], avg_line[y - 22])
        p9 = percent_delta(avg_line[y - 30], avg_line[y - 21])
        p10 = percent_delta(avg_line[y - 30], avg_line[y - 20])

        p11 = percent_delta(avg_line[y - 30], avg_line[y - 19])
        p12 = percent_delta(avg_line[y - 30], avg_line[y - 18])
        p13 = percent_delta(avg_line[y - 30], avg_line[y - 17])
        p14 = percent_delta(avg_line[y - 30], avg_line[y - 16])
        p15 = percent_delta(avg_line[y - 30], avg_line[y - 15])
        p16 = percent_delta(avg_line[y - 30], avg_line[y - 14])
        p17 = percent_delta(avg_line[y - 30], avg_line[y - 13])
        p18 = percent_delta(avg_line[y - 30], avg_line[y - 12])
        p19 = percent_delta(avg_line[y - 30], avg_line[y - 11])
        p20 = percent_delta(avg_line[y - 30], avg_line[y - 10])

        p21 = percent_delta(avg_line[y - 30], avg_line[y - 9])
        p22 = percent_delta(avg_line[y - 30], avg_line[y - 8])
        p23 = percent_delta(avg_line[y - 30], avg_line[y - 7])
        p24 = percent_delta(avg_line[y - 30], avg_line[y - 6])
        p25 = percent_delta(avg_line[y - 30], avg_line[y - 5])
        p26 = percent_delta(avg_line[y - 30], avg_line[y - 4])
        p27 = percent_delta(avg_line[y - 30], avg_line[y - 3])
        p28 = percent_delta(avg_line[y - 30], avg_line[y - 2])
        p29 = percent_delta(avg_line[y - 30], avg_line[y - 1])
        p30 = percent_delta(avg_line[y - 30], avg_line[y])

        outcome_range = avg_line[y+20:y+30]
        current_point = avg_line[y]

        # Collect the average outcome, which is a percent of a percent. Mathematically could return inf%. So put it in
        # an exception and return 0 if this occurs.

        try:
            avg_outcome = reduce(lambda x, y: x+y, outcome_range)/len(outcome_range)
        except Exception as e:
            print(str(e))
            avg_outcome = 0

        future_outcome = percent_delta(current_point, avg_outcome)

        pattern.append(p1)
        pattern.append(p2)
        pattern.append(p3)
        pattern.append(p4)
        pattern.append(p5)
        pattern.append(p6)
        pattern.append(p7)
        pattern.append(p8)
        pattern.append(p9)
        pattern.append(p10)

        pattern.append(p11)
        pattern.append(p12)
        pattern.append(p13)
        pattern.append(p14)
        pattern.append(p15)
        pattern.append(p16)
        pattern.append(p17)
        pattern.append(p18)
        pattern.append(p19)
        pattern.append(p20)

        pattern.append(p21)
        pattern.append(p22)
        pattern.append(p23)
        pattern.append(p24)
        pattern.append(p25)
        pattern.append(p26)
        pattern.append(p27)
        pattern.append(p28)
        pattern.append(p29)
        pattern.append(p30)

        pattern_arr.append(pattern)
        performance_arr.append(future_outcome)

        y += 1

    pattern_end_time = time.time()

    print(len(pattern_arr))
    print(len(performance_arr))
    print('Pattern storage took ' + str(pattern_end_time - pattern_start_time) + ' seconds')

def current_pattern():

    cp1 = percent_delta(avg_line[-31], avg_line[-30])
    cp2 = percent_delta(avg_line[-31], avg_line[-29])
    cp3 = percent_delta(avg_line[-31], avg_line[-28])
    cp4 = percent_delta(avg_line[-31], avg_line[-27])
    cp5 = percent_delta(avg_line[-31], avg_line[-26])
    cp6 = percent_delta(avg_line[-31], avg_line[-25])
    cp7 = percent_delta(avg_line[-31], avg_line[-24])
    cp8 = percent_delta(avg_line[-31], avg_line[-23])
    cp9 = percent_delta(avg_line[-31], avg_line[-22])
    cp10 = percent_delta(avg_line[-31], avg_line[-21])

    cp11 = percent_delta(avg_line[-31], avg_line[-20])
    cp12 = percent_delta(avg_line[-31], avg_line[-19])
    cp13 = percent_delta(avg_line[-31], avg_line[-18])
    cp14 = percent_delta(avg_line[-31], avg_line[-17])
    cp15 = percent_delta(avg_line[-31], avg_line[-16])
    cp16 = percent_delta(avg_line[-31], avg_line[-15])
    cp17 = percent_delta(avg_line[-31], avg_line[-14])
    cp18 = percent_delta(avg_line[-31], avg_line[-13])
    cp19 = percent_delta(avg_line[-31], avg_line[-12])
    cp20 = percent_delta(avg_line[-31], avg_line[-11])

    cp21 = percent_delta(avg_line[-31], avg_line[-10])
    cp22 = percent_delta(avg_line[-31], avg_line[-9])
    cp23 = percent_delta(avg_line[-31], avg_line[-8])
    cp24 = percent_delta(avg_line[-31], avg_line[-7])
    cp25 = percent_delta(avg_line[-31], avg_line[-6])
    cp26 = percent_delta(avg_line[-31], avg_line[-5])
    cp27 = percent_delta(avg_line[-31], avg_line[-4])
    cp28 = percent_delta(avg_line[-31], avg_line[-3])
    cp29 = percent_delta(avg_line[-31], avg_line[-2])
    cp30 = percent_delta(avg_line[-31], avg_line[-1])

    pattern_for_recognition.append(cp1)
    pattern_for_recognition.append(cp2)
    pattern_for_recognition.append(cp3)
    pattern_for_recognition.append(cp4)
    pattern_for_recognition.append(cp5)
    pattern_for_recognition.append(cp6)
    pattern_for_recognition.append(cp7)
    pattern_for_recognition.append(cp8)
    pattern_for_recognition.append(cp9)
    pattern_for_recognition.append(cp10)

    pattern_for_recognition.append(cp11)
    pattern_for_recognition.append(cp12)
    pattern_for_recognition.append(cp13)
    pattern_for_recognition.append(cp14)
    pattern_for_recognition.append(cp15)
    pattern_for_recognition.append(cp16)
    pattern_for_recognition.append(cp17)
    pattern_for_recognition.append(cp18)
    pattern_for_recognition.append(cp19)
    pattern_for_recognition.append(cp20)

    pattern_for_recognition.append(cp21)
    pattern_for_recognition.append(cp22)
    pattern_for_recognition.append(cp23)
    pattern_for_recognition.append(cp24)
    pattern_for_recognition.append(cp25)
    pattern_for_recognition.append(cp26)
    pattern_for_recognition.append(cp27)
    pattern_for_recognition.append(cp28)
    pattern_for_recognition.append(cp29)
    pattern_for_recognition.append(cp30)

    print(pattern_for_recognition)

def pattern_recogniton():

    predicted_outcomes_arr = []
    pattern_found = 0
    plot_pattern_arr = []

    for each_pattern in pattern_arr:
        sim1 = 100.00 - abs(percent_delta(each_pattern[0], pattern_for_recognition[0]))
        sim2 = 100.00 - abs(percent_delta(each_pattern[1], pattern_for_recognition[1]))
        sim3 = 100.00 - abs(percent_delta(each_pattern[2], pattern_for_recognition[2]))
        sim4 = 100.00 - abs(percent_delta(each_pattern[3], pattern_for_recognition[3]))
        sim5 = 100.00 - abs(percent_delta(each_pattern[4], pattern_for_recognition[4]))
        sim6 = 100.00 - abs(percent_delta(each_pattern[5], pattern_for_recognition[5]))
        sim7 = 100.00 - abs(percent_delta(each_pattern[6], pattern_for_recognition[6]))
        sim8 = 100.00 - abs(percent_delta(each_pattern[7], pattern_for_recognition[7]))
        sim9 = 100.00 - abs(percent_delta(each_pattern[8], pattern_for_recognition[8]))
        sim10 = 100.00 - abs(percent_delta(each_pattern[9], pattern_for_recognition[9]))

        sim11 = 100.00 - abs(percent_delta(each_pattern[10], pattern_for_recognition[10]))
        sim12 = 100.00 - abs(percent_delta(each_pattern[11], pattern_for_recognition[11]))
        sim13 = 100.00 - abs(percent_delta(each_pattern[12], pattern_for_recognition[12]))
        sim14 = 100.00 - abs(percent_delta(each_pattern[13], pattern_for_recognition[13]))
        sim15 = 100.00 - abs(percent_delta(each_pattern[14], pattern_for_recognition[14]))
        sim16 = 100.00 - abs(percent_delta(each_pattern[15], pattern_for_recognition[15]))
        sim17 = 100.00 - abs(percent_delta(each_pattern[16], pattern_for_recognition[16]))
        sim18 = 100.00 - abs(percent_delta(each_pattern[17], pattern_for_recognition[17]))
        sim19 = 100.00 - abs(percent_delta(each_pattern[18], pattern_for_recognition[18]))
        sim20 = 100.00 - abs(percent_delta(each_pattern[19], pattern_for_recognition[19]))

        sim21 = 100.00 - abs(percent_delta(each_pattern[20], pattern_for_recognition[20]))
        sim22 = 100.00 - abs(percent_delta(each_pattern[21], pattern_for_recognition[21]))
        sim23 = 100.00 - abs(percent_delta(each_pattern[22], pattern_for_recognition[22]))
        sim24 = 100.00 - abs(percent_delta(each_pattern[23], pattern_for_recognition[23]))
        sim25 = 100.00 - abs(percent_delta(each_pattern[24], pattern_for_recognition[24]))
        sim26 = 100.00 - abs(percent_delta(each_pattern[25], pattern_for_recognition[25]))
        sim27 = 100.00 - abs(percent_delta(each_pattern[26], pattern_for_recognition[26]))
        sim28 = 100.00 - abs(percent_delta(each_pattern[27], pattern_for_recognition[27]))
        sim29 = 100.00 - abs(percent_delta(each_pattern[28], pattern_for_recognition[28]))
        sim30 = 100.00 - abs(percent_delta(each_pattern[29], pattern_for_recognition[29]))

        how_sim = (sim1 + sim2 + sim3 + sim4 + sim5 + sim6 + sim7+ sim8 + sim9 + sim10 +
                   sim11 + sim12 + sim13 + sim14 + sim15 + sim16 + sim17+ sim18 + sim19 + sim20 +
                   sim21 + sim22 + sim23 + sim24 + sim25 + sim26 + sim27+ sim28 + sim29 + sim30)/30.0

        if how_sim > 50:

            pattern_index = pattern_arr.index(each_pattern)
            pattern_found = 1

            #print('###############')
            #print(pattern_for_recognition)
            #print('===============')
            #print(each_pattern)
            #print('---------------')
            #print('Predicted outcome ' + str(performance_arr[pattern_index]))

            # Now plot each similar pattern against the original
            xp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
            plot_pattern_arr.append(each_pattern)

    if pattern_found == 1:
        fig = plt.figure(figsize = (10,5))

        for each_patt in plot_pattern_arr:

            future_points = pattern_arr.index(each_patt)

            if performance_arr[future_points] > pattern_for_recognition[29]:
                pcolour = '#24bc00'
            else:
                pcolour = '#d40000'

            plt.plot(xp, each_patt, linewidth = 1, alpha = 0.5)
            predicted_outcomes_arr.append(performance_arr[future_points])

            plt.scatter(35, performance_arr[future_points], c = pcolour, alpha = 0.5)

        real_outcome_range = all_data[to_what + 20: to_what + 30]
        real_average_outcome = reduce(lambda x, y: x + y, real_outcome_range)/len(real_outcome_range)
        real_movement = percent_delta(all_data[to_what], real_average_outcome)

        predicted_average_outcome = reduce(lambda x, y: x + y, predicted_outcomes_arr) / len(predicted_outcomes_arr)

        plt.scatter(40, real_movement, c = '#54fff7', s = 25)
        plt.scatter(40, predicted_average_outcome, c = 'b', s = 30)


        plt.plot(xp, pattern_for_recognition, '#54fff7', linewidth = 3)
        plt.grid(True)
        plt.title('Pattern Recognition')
        plt.show()

data_length = int(len(USDJPY_FX['OPEN']))
print('Data length is '+ str(data_length))

to_what = 1000
all_data = ((USDJPY_FX['OPEN'] + USDJPY_FX['CLOSE']) / 2)

while to_what < data_length:

    #avg_line = ((USDJPY_FX['OPEN'] + USDJPY_FX['CLOSE']) / 2)
    avg_line = all_data[:to_what]

    pattern_arr = []
    performance_arr = []
    pattern_for_recognition = []

    # Run all functions in order

    pattern_storage()
    current_pattern()
    pattern_recogniton()

    total_time = time.time() - total_start_time
    print('Total processing time took: ' + str(total_time))

    to_what += 1

