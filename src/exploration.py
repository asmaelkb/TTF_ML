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

def common(cgce, critical):
    res = []
    for index, line in critical.iterrows():
        sensor = line['Tag']
        if sensor in cgce.columns:
            res.append(sensor)

    return res

if __name__ == "__main__":

    # Retrieve the dataframes of each dataset

    df = pd.read_csv('../data/CGCE-2019-PivotHourCombined.csv')
    critical = pd.read_csv('../data/Critical_Tag.csv')
    failure_date = pd.read_csv('../data/Failure_date.csv')


    # Displays sensors that are out of their allowed value range

    date = '2019-05-18 07:00:00'
    common_sensors = common(df, critical)

    print(date)

    for sensor in common_sensors:
        _, value = df.loc[df['Time'] == date, sensor]

        upper =  critical.loc[critical['Tag'] == sensor, 'Upper Limit']
        upper = int(upper.iloc[0])

        lower = critical.loc[critical['Tag'] == sensor, 'Low Limit']
        lower = int(lower.iloc[0])

        if value > upper or value < lower:

            print("Value : ", int(value))
            print("Upper : ", upper)
            print("Lower : ", lower)

            print(sensor)

