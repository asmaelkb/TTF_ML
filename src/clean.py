import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from exploration import common

df = pd.read_csv('../data/CGCE-2019-PivotHourCombined.csv')
dfc = pd.read_csv('../data/Cleaned-CGCE-2019-PivotHourCombined.csv')
dfc2 = pd.read_csv('../data/Cleaned2-CGCE-2019-PivotHourCombined.csv')
critical = pd.read_csv('../data/Critical_Tag.csv')
failure_date = pd.read_csv('../data/Failure_date.csv')

def format(dates):
    """ Returns the correct formatted list of dates and hours """
    res = []
    for date in dates:
        left = date.split(":", 1)
        res.append(left[0] + ":00:00") # To have 23:00:00 instead of 23:18:00, to match correctly the format of the CGCE csv file
    
    return res


if __name__ == "__main__":

    empty_cells = df.isna().sum().sum()
    total = df.shape[0] * df.shape[1]
    print("Percentage of empty cells (missing values):", round(100 * empty_cells / total, 1), "%")

    # Remove duplicates
    # print("Shape before dropping duplicates : ", df.shape)
    # df.drop_duplicates(subset=['Time'], inplace=True)
    # print("Shape after dropping duplicates : ", df.shape)

    # df.to_csv('../data/Cleaned-CGCE-2019-PivotHourCombined.csv')

    empty_cells = dfc.isna().sum().sum()
    total = dfc.shape[0] * dfc.shape[1]
    print("Percentage of empty cells (missing values):", round(100 * empty_cells / total, 1), "%")

    # Remove the outliers (value out of range, but did not cause any shutdown)

    # failure_dates = format(failure_date['Time'].tolist())
    # common_sensors = common(dfc, critical)

    # for sensor in common_sensors:           # Browse the list of common_sensors
    #     for date in dfc['Time']:     # For each sensor, look at its value for all time
            
    #         if date not in failure_dates:   # If, at this date and hour, no shutdown
                
    #             value = (dfc.loc[dfc['Time'] == date, sensor]).iloc[0]

    #             if not np.isnan(value):
    #                 value = int(value)

    #                 upper =  critical.loc[critical['Tag'] == sensor, 'Upper Limit']
    #                 MAX = int(upper.iloc[0])

    #                 lower = critical.loc[critical['Tag'] == sensor, 'Low Limit']
    #                 MIN = int(lower.iloc[0])

    #                 if value < MIN or value > MAX:      # If its value is out of range -> remove the cell
    #                     dfc.loc[dfc['Time'] == date, sensor] = np.nan

    
    # dfc.to_csv('../data/Cleaned2-CGCE-2019-PivotHourCombined.csv')

    empty_cells = dfc2.isna().sum().sum()
    total = dfc2.shape[0] * dfc2.shape[1]
    print("Percentage of empty cells (missing values):", round(100 * empty_cells / total, 1), "%")

    ## Visualize how empty the dataset is
    # Extract the column names
    cols = dfc2.columns

    # Plot a heatmap of missing values with seaborn
    plt.figure(figsize = (10,5))
    heatmap = sns.heatmap(dfc2[cols].isnull())

    heatmap.get_figure().savefig('missing_values_heatmap.png')


                
               
