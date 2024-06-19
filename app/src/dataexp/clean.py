import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from exploration import common
import dask.dataframe as dd
import time
import countdown

df7k = pd.read_csv('data/CGCE7K-1920-0503-Pivot15M-updated.csv', parse_dates=['Time'])
df7k = df7k.set_index('Time')

# df = pd.read_csv('../data/CGCE-2019-PivotHourCombined.csv')
# dfc = pd.read_csv('../data/Cleaned-CGCE-2019-PivotHourCombined.csv')
# dfc2 = pd.read_csv('../data/Cleaned2-CGCE-2019-PivotHourCombined.csv')
# critical = pd.read_csv('../data/Critical_Tag.csv')
# failure_date = pd.read_csv('../data/Failure_date.csv')

def format(dates):
    """ Returns the correct formatted list of dates and hours """
    res = []
    for date in dates:
        left = date.split(":", 1)
        res.append(left[0] + ":00:00") # To have 23:00:00 instead of 23:18:00, to match correctly the format of the CGCE csv file
    
    return res

def may_july(df):

    start_date = pd.to_datetime('2019-05-01 00:00:00')
    end_date = pd.to_datetime('2019-06-30 00:00:00')
    df_index = df.index
    df_index = pd.to_datetime(df_index)

    df_index_filtered = df_index[df_index >= start_date]
    df_index_filtered = df_index_filtered[df_index_filtered <= end_date]
    df_res = df.loc[df_index_filtered]

    return df_res

def correlation(df, df_countdown):
    start_time = time.time()
    # ddf = dd.from_pandas(df, npartitions=4)

    # print("df index : ", df.index)
    # print("df countdown index : ", df_countdown.index)
    # df_tags = df.reset_index(drop=True)

    common_index = df.index.intersection(df_countdown.index)

    # Filter df_tags to keep only the common indices
    df_tags = df.loc[common_index]
    print(df_tags)

    correlation_matrix = df_tags.corrwith(df_countdown['Countdown'])
   
    # Plot the heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title("Heatmap of correlation matrix of highly available tags and countdown")
    plt.savefig('correlation_matrix_tags.png')  # Save the figure as PNG

    execution_time = time.time() - start_time
    print("Temps d'exécution pour la matrice : ", execution_time ," secondes") 
   
    # # Find pairs of tags with high correlation
    # highly_correlated_pairs = []
    # threshold = 0.8  # You can adjust the threshold as needed

    # for i in range(len(correlation_matrix.columns)):
    #     for j in range(i+1, len(correlation_matrix.columns)):
    #         if abs(correlation_matrix.iloc[i, j]) >= threshold:
    #             pair = (correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j])
    #             highly_correlated_pairs.append(pair)

    # # Print pairs of tags with high correlation
    # print("\nHighly Correlated Tag Pairs (correlation >= {:.2f}):".format(threshold))
    # for pair in highly_correlated_pairs:
    #     print(pair)

if __name__ == "__main__":

    ### Initial CGCE dataset
    # empty_cells = df.isna().sum().sum()
    # total = df.shape[0] * df.shape[1]
    # print("Percentage of empty cells (missing values):", round(100 * empty_cells / total, 1), "%")

    # Remove duplicates
    # print("Shape before dropping duplicates : ", df.shape)
    # df.drop_duplicates(subset=['Time'], inplace=True)
    # print("Shape after dropping duplicates : ", df.shape)

    # df.to_csv('../data/Cleaned-CGCE-2019-PivotHourCombined.csv')

    ### Cleaned CGCE dataset
    # empty_cells = dfc.isna().sum().sum()
    # total = dfc.shape[0] * dfc.shape[1]
    # print("Percentage of empty cells (missing values):", round(100 * empty_cells / total, 1), "%")

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

    ### Cleaned 2 CGCE dataset
    # empty_cells = dfc2.isna().sum().sum()
    # total = dfc2.shape[0] * dfc2.shape[1]
    # print("Percentage of empty cells (missing values):", round(100 * empty_cells / total, 1), "%")

    ### Visualize how empty the dataset is
    # Extract the column names
    # cols = dfc2.columns

    ### Between May and July, how many tags have more than 70% availability?

    df = may_july(df7k)
    # Tags with high availability (> 70% of non-empty cells)
    tags_high = 0

    selected_tags = []

    for col in df.columns:
        missing = df[col].isnull().sum()

        total_cells = df[col].size
        percentage_missing = (missing / total_cells) * 100 # Percentage of empty cells for this tag
        percentage_non_missing = 100 - percentage_missing

        if percentage_non_missing > 70:
            tags_high += 1

            # Create a new dataframe for the correlation matrix (only for highly available tags)
            selected_tags.append(col)
    
    df_high = df[selected_tags]
    
    print("Number of tags with high availability : ", tags_high)
    print("Total of tags : ", df.shape[1])

    ### Correlation between tags and countdown

    df_failure = pd.read_csv('data/Failure_date.csv', parse_dates=['Time'])
    df_failure['Time'] = pd.to_datetime(df_failure['Time'])
    df_failure['Time'] = df_failure['Time'].apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))
    
    df = countdown.init()

    for date in df_failure['Time'].tolist():
        df.loc[df['Time'] == date, 'Failure'] = 'YES'
        df = countdown.fcountdown(df, date)

    df_countdown = df

    correlation(df_high, df_countdown)

    # missing_values_count = df.isnull().sum().sum()

    # print("Number of empty cells between May and July : ", missing_values_count)
    # print("Percentage of empty cells (missing values):", round(100 * missing_values_count / total, 1), "%")

    ### Plot a heatmap of missing values with seaborn7k

    # cols = df.columns
    # plt.figure(figsize = (10,5))

    # heatmap = sns.heatmap(df[cols].isnull())
    # heatmap.get_figure().savefig('missing_values_heatmap_df7k.png')


                
               
