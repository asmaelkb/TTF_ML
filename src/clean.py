import pandas as pd
import numpy as np 
from exploration import common

df = pd.read_csv('../data/CGCE-2019-PivotHourCombined.csv')
dfc = pd.read_csv('../data/Cleaned-CGCE-2019-PivotHourCombined.csv')
critical = pd.read_csv('../data/Critical_Tag.csv')
failure_date = pd.read_csv('../data/Failure_date.csv')

def format(dates):
    """ Returns the correct formatted list of dates and hours """
    res = []
    for date in dates:
        left = date.split(":", 1)
        res.append(left[0] + ":00:00")
    
    return res


if __name__ == "__main__":

    # Remove duplicates
    # print("Shape before dropping duplicates : ", df.shape)
    # df.drop_duplicates(subset=['Time'], inplace=True)
    # print("Shape after dropping duplicates : ", df.shape)

    # df.to_csv('../data/Cleaned-CGCE-2019-PivotHourCombined.csv')

    empty_cells = dfc.isna().sum().sum()
    total = dfc.shape[0] * dfc.shape[1]
    print("Percentage of empty cells :", round(100 * empty_cells / total, 1), "%")

    # Remove the outliers (value out of range, but did not cause any shutdown)

    failure_dates = format(failure_date['Time'].tolist())
    print(failure_dates)
    common_sensors = common(dfc, critical)

    # Takes too long to run : to fix
    for sensor in common_sensors:           # Browse the list of common_sensors
        for date in dfc['Time']:     # For each sensor, look at its value for all time
            
            if date not in failure_dates:   # If, at this date and hour, no shutdown
                
                value = (dfc.loc[dfc['Time'] == date, sensor]).iloc[0]

                if not np.isnan(value):
                    value = int(value)

                    upper =  critical.loc[critical['Tag'] == sensor, 'Upper Limit']
                    MAX = int(upper.iloc[0])

                    lower = critical.loc[critical['Tag'] == sensor, 'Low Limit']
                    MIN = int(lower.iloc[0])

                    if value < MIN or value > MAX:      # If its value is out of range -> remove the cell
                        dfc.loc[dfc['Time'] == date, sensor] = np.nan


                
               
