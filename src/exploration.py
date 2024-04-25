import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def limit_values(sensor, n, critical):
    # Search for the minimum value for this specific sensor, in the critical csv
    min_val = critical.loc[critical['Tag'] == sensor, 'Low Limit']
    max_val = critical.loc[critical['Tag'] == sensor, 'Upper Limit']
    
    print(min_val, max_val)

    MIN = n * [min_val]
    MAX = n * [max_val]

    return MIN, MAX

if __name__ == "__main__":

    # Retrieve the dataframes of each dataset

    df = pd.read_csv('../data/CGCE-2019-PivotHourCombined.csv')
    critical = pd.read_csv('../data/Critical_Tag.csv')
    failure_date = pd.read_csv('../data/Failure_date.csv')

    # Retrieve manually the name of the sensor that is causing the shutdown

    x = []
    y = []

    sensor = 'PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:PZT2468_EN.PNT'
    date = '2019-05-18'

    # For each hour in a day
    for i in range(2, 24): # 2-hours delay
        dateh = date + ' ' + str(i).zfill(2) + ':00:00'
        x.append(str(i).zfill(2) + ':00')
        index, value = df.loc[df['Time'] == dateh, sensor]
        y.append(value)

    # Retrieve the limit values ​​(values ​​from which the equipment shuts down) of each sensor
    MIN, MAX = limit_values(sensor, len(x), critical)

    x_num = np.arange(len(x))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.plot(x, y, color='tab:red')
    ax1.plot(x, MIN, color='tab:blue')
    ax1.plot(x, MAX, color='tab:green')
    plt.xticks(x[::5])  # Print every 5th hour
    ax1.tick_params(axis='y')
    plt.title("Variation of the sensor responsible of the shutdown of " + date)


    # Display
    fig.tight_layout()
    plt.savefig(date + ".png")

