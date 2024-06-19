import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

def init(index):

    date_range = index
    df = pd.DataFrame(date_range, columns=['Time'])

    df['Failure'] = 'NO'
    df['Countdown'] = 0

    return df

def fcountdown(df, date):
    step = 0
    start = (df.loc[df['Time'] == date].index)[0]

    current = start - 1

    while current >= 0 and current < len(df):
        if df.at[current, 'Failure'] == 'YES':
            return df
        
        step += 1
        df.at[current, 'Countdown'] = step

        current -= 1
    
    return df

def may_july(df):

    start_date = pd.to_datetime('2019-05-01 00:00:00')
    end_date = pd.to_datetime('2019-06-30 23:00:00')
    df_index = df.index
    df_index = pd.to_datetime(df_index)

    df_index_filtered = df_index[df_index >= start_date]
    df_index_filtered = df_index_filtered[df_index_filtered <= end_date]
    df_res = df.loc[df_index_filtered]

    return df_res


def correlation(df, df_countdown):
    
    corr = {}
    df_countdown = df_countdown.set_index('Time')
    for col in df.columns:

        corr[col] = (df[col].fillna(df[col].mean())).corr(df_countdown.Countdown)
        if corr[col] > 0.3:
            print("Tag : ", col, "Value : ", corr[col])

    df_corr = pd.DataFrame.from_dict(corr, orient='index', columns=['Values'])
    df_corr = df_corr.reset_index().rename(columns={'index': 'Tag'})

    # Sort the columns
    df_corr = df_corr.sort_values(by='Values', ascending=False)
    df_corr.to_csv("correlation.csv", index=False)



if __name__ == "__main__":

    ### Retrieve the tags with high availability, between May and July
    df7k = pd.read_csv('data/CGCE7K-1920-0503-Pivot15M-updated.csv', parse_dates=['Time'])
    df7k = df7k.set_index('Time')
    df7k = may_july(df7k)

    ### Creating the countdown dataframe
    df_failure = pd.read_csv('data/Failure_date_May_July.csv', parse_dates=['Time'])
    df_failure['Time'] = pd.to_datetime(df_failure['Time'])

    df_failure['Time'] = df_failure['Time'].apply(lambda x: x.replace(minute=0, second=0, microsecond=0))
    
    df = init(df7k.index)

    for date in df_failure['Time'].tolist():
        df.loc[df['Time'] == date, 'Failure'] = 'YES'
        df = fcountdown(df, date)
    
    df['Time'] = pd.to_datetime(df['Time'])
    df_countdown = df.loc[(df['Time'] >= '2019-05-01') & (df['Time'] <= '2019-06-30')]
    df.to_csv("countdown_new.csv")
    # Tags with high availability (> 70% of non-empty cells)
    # tags_high = 0

    # selected_tags = []

    # for col in df7k.columns:
    #     missing = df7k[col].isnull().sum()

    #     total_cells = df7k[col].size
    #     percentage_missing = (missing / total_cells) * 100 # Percentage of empty cells for this tag
    #     percentage_non_missing = 100 - percentage_missing

    #     if percentage_non_missing > 70:
    #         tags_high += 1

    #         # Create a new dataframe for the correlation matrix (only for highly available tags)
    #         selected_tags.append(col)
    
    # df_high = df7k[selected_tags]


    # ### Creating the correlation matrix between the tags highly available and the countdown
    # correlation(df_high, df_countdown)






    ### Plot the countdown dataframe

    # df = pd.read_csv('countdown_may_july.csv', parse_dates=['Time'])
    # df['Time'] = pd.to_datetime(df['Time'])

    # plt.figure(figsize=(8, 6))
    # plt.plot(df['Time'], df['Countdown'])
    # plt.xticks(rotation=90, fontsize=8)
    # plt.subplots_adjust(bottom=0.2) 
    # plt.title("Countdown between May and July")
    # plt.savefig("Countdown_May_July.png")




        

