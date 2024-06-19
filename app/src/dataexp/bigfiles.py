import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from datetime import datetime
from collections import defaultdict

def relation(df, tags):
    ''' Find all the sensors that have the same SAP MAPPING value as the sensors in tags '''
    output = '../data/sap_mapping/'

    for tag in tags.keys():
        # Find the SAP MAPPING value of the tag
        sap = df.loc[df['Tag'] == tag, 'SAP MAPPING ']

        # Find all the tags related to the current tag
        rel = df.loc[df['SAP MAPPING '] == sap.iloc[0]]

        # # Complete the dataframe
        # if '6050' in sap.iloc[0]:
        #     for i, element in enumerate(rel['Tag']):
        #         df_final.at[i, tag] = element
        # else:
        # df_final[tag] = rel['Tag']

        # Write in the csv file the sensors related to the tag
        rel.to_csv(output + tag + ".csv", index=False)
        
        # df_final.to_csv(output, index=False)

def load_categories(f):
    df = pd.read_csv(f)

    res = df['Category'].tolist()
    # res = [element.upper().replace(' ', '') for element in res]
    res.remove("STATUS")

    if "Status" in res:
        res.remove("Status")

    if "Status " in res:
        res.remove("Status ")

    return list(set(res))

        
# Trends by category 
def trendcat(df):
    path = "data/sap_mapping/"

    files = ["PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH6050B.PNT.csv",
             "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH6050A.PNT.csv"]
             # "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6007D.CIN.csv"]

    failures = ['2019-05-18 11:00:00', '2019-05-25 9:00:00', '2019-05-30 06:00:00', '2019-06-19 12:00:00', '2019-06-30 03:00:00']

    # print("Compute")
    # df = df.compute()

    print("Re-sample")
    df = df.resample('15min').mean() 

    df_index = df.index
    
    # Defining the time (date) range
    start_date = pd.to_datetime('2019-05-01 00:00:00')
    end_date = pd.to_datetime('2019-07-01 00:00:00')

    df_index_filtered = df_index[df_index >= start_date]
    df_index_filtered = df_index_filtered[df_index_filtered <= end_date]
    df_2019c = df.loc[df_index_filtered]

    df_2019 = df_2019c[pd.to_datetime(df_2019c.index, errors='coerce').notnull()]

    # df_2019 = df.loc[(df_index >= start_date) & (df_index <= end_date)] # doesn't work 
    # date_range = pd.date_range(start='2019-05-01', end='2019-07-01')
    # df_2019 = df[df_index.isin(date_range)]

    # For each tag file
    for f in files:
        print(f)
        tags = pd.read_csv(path + f)

        categories = load_categories(path + f)

        fig, axes = plt.subplots(nrows=1, ncols=len(categories), figsize=(12, 6))

        legend_labels = set()

        for tag in tags['Tag']:
            print(tag)
            if tag in df_2019.columns.tolist():
                cat = tags.loc[tags['Tag'] == tag, 'Category'].iloc[0]
                
                if cat != "STATUS" and cat != "Status" and cat != "Status ": 
                    i = categories.index(cat)
                    print(cat)

                    if len(categories) != 1:
                        if tag not in legend_labels:
                            axes[i].plot(df_2019.index, df_2019[tag], label=tag)
                            axes[i].legend(loc='upper right')
                            axes[i].set_title("Trend for the category " + cat)
                            axes[i].tick_params(axis='x', labelsize=7)
                            legend_labels.add(tag)

                            for date in failures:
                                date = pd.to_datetime(date)
                                axes[i].axvline(x=date, color='red', linestyle="--")
                                
                    else:
                        if tag not in legend_labels:
                            axes.plot(df_2019.index, df_2019[tag], label=tag)
                            axes.legend(loc='upper right')
                            axes.set_title("Trend for the category " + cat)
                            axes.tick_params(axis='x', labelsize=7)
                            legend_labels.add(tag)

                            for date in failures:
                                date = pd.to_datetime(date)
                                axes.axvline(x=date, color='red', linestyle="--")

        fig.savefig('src/cat/' + f + '.png') 
        plt.close()


            # if cat == "PRESSURE":
            #     axes[0].plot(df_2019.index.compute(), df_2019[tag].compute(), label=tag)
            #     axes[0].legend(loc="upper right")
            #     axes[0].set_title("Trend for the category " + cat)
            #     axes[0].tick_params(axis='x', labelsize=7)

            #     for date in failures:
            #         date = pd.to_datetime(date)
            #         axes[0].axvline(x=date, color='red', linestyle="--", label=date)
            #         # axes[0].legend()
                
            # if cat == "LEVEL":
            #     axes[1].plot(df_2019.index.compute(), df_2019[tag].compute(), label=tag)
            #     axes[1].legend(loc="upper right")
            #     axes[1].set_title("Trend for the category " + cat)
            #     axes[1].tick_params(axis='x', labelsize=7)

            #     for date in failures:
            #         date = pd.to_datetime(date)
            #         axes[1].axvline(x=date, color='red', linestyle="--", label=date)
            #         # axes[1].legend()


# Trends by category 
def trendcat2(df):
    path = "data/sap_mapping/"

    files = ["PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6007D.CIN.csv"]

    failures = ['2019-05-18 11:00:00', '2019-05-25 9:00:00', '2019-05-30 06:00:00', '2019-06-19 12:00:00', '2019-06-30 03:00:00']

    # print("Compute")
    # df = df.compute()

    print("Re-sample")
    df = df.resample('15min').mean() 

    df_index = df.index
    
    # Defining the time (date) range
    start_date = pd.to_datetime('2019-05-01 00:00:00')
    end_date = pd.to_datetime('2019-07-01 00:00:00')

    df_index_filtered = df_index[df_index >= start_date]
    df_index_filtered = df_index_filtered[df_index_filtered <= end_date]
    df_2019c = df.loc[df_index_filtered]

    df_2019 = df_2019c[pd.to_datetime(df_2019c.index, errors='coerce').notnull()]

    # df_2019 = df.loc[(df_index >= start_date) & (df_index <= end_date)] # doesn't work 
    # date_range = pd.date_range(start='2019-05-01', end='2019-07-01')
    # df_2019 = df[df_index.isin(date_range)]

    # For each tag file
    for f in files:
        print(f)
        tags = pd.read_csv(path + f)

        categories = load_categories(path + f)

        legend_labels = set()

        # For each category
        for cat in categories:

            if cat != "STATUS" and cat != "Status" and cat != "Status ": 
                # Retrieve all the tags of this category
                cat_tags = tags.loc[tags['Category'] == cat, 'Tag'].unique().tolist()

                if cat == "FLOW":
                    desc = tags.loc[tags['Category'] == cat, 'Description'].unique().tolist()
                    print("Number of tags in FLOW : ", len(cat_tags))
                    print("Non null descriptions : ", len(desc))
                    print("Duplicates : ", len(desc) - len(set(desc)))

                for tag in cat_tags:
                    if tag in df_2019.columns.tolist():

                            if tag not in legend_labels:
                                plt.plot(df_2019.index, df_2019[tag])
                                #plt.legend(loc='upper right')
                                plt.title("Trend for the category " + cat)
                                plt.tick_params(axis='x', labelsize=7)
                                legend_labels.add(tag)

                                for date in failures:
                                    date = pd.to_datetime(date)
                                    plt.axvline(x=date, color='red', linestyle="--")
                
    
                # plt.savefig('src/cat/third/' + cat + "_" + f + '.png') 
                # plt.close()

def group_similar_desc(desc):
    groups = defaultdict(list)

    for word in desc:

        if any(char.isdigit() for char in word):
            # Group by prefix for words containing digits
            prefix = ''.join([char for char in word if not char.isdigit()])
            groups[prefix].append(word)
        elif ' ' in word:
            # Group by the first word in multi-word phrases
            prefix = word.split()[0]
            groups[prefix].append(word)
        else:
            # Each single word without digits goes in its own group
            groups[word].append(word)
    
    # Convert defaultdict to a list of lists
    return list(groups.values())

def divide(df):
    """ Dividing the tags (FLOW) based on their description """
    taglist = pd.read_csv("../data/sap_mapping/PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6007D.CIN.csv")

    failures = ['2019-05-18 11:00:00', '2019-05-25 9:00:00', '2019-05-30 06:00:00', '2019-06-19 12:00:00', '2019-06-30 03:00:00']
    print("Re-sample")
    df = df.resample('15min').mean() 
    df_index = df.index
    
    # Defining the time (date) range
    start_date = pd.to_datetime('2019-05-01 00:00:00')
    end_date = pd.to_datetime('2019-07-01 00:00:00')

    df_index_filtered = df_index[df_index >= start_date]
    df_index_filtered = df_index_filtered[df_index_filtered <= end_date]
    df_2019c = df.loc[df_index_filtered]
    df_2019 = df_2019c[pd.to_datetime(df_2019c.index, errors='coerce').notnull()]


    filtered_taglist = taglist[taglist['Category'] == 'FLOW']
    filtered_taglist = filtered_taglist.dropna(subset=['Description'])
    descriptions = filtered_taglist['Description'].tolist()

    # descriptions = taglist.loc[taglist['Category'] == 'FLOW', 'Description'].tolist()
    
    # filtered_taglist = taglist[taglist['Category'] == 'FLOW'].dropna(subset=['Description'])
    # descriptions = filtered_taglist['Description'].tolist()
    
    grp_descriptions = group_similar_desc(descriptions)

    for grp in grp_descriptions:
        print("Groupe :", grp)

        for d in grp:

            tag = taglist.loc[taglist['Description'] == d, 'Tag'].iloc[0]

            plt.plot(df_2019.index, df_2019[tag], label=tag)
            plt.legend()
            plt.title("Trend for tags of category FLOW")
            plt.tick_params(axis='x', labelsize=7)

            for date in failures:
                date = pd.to_datetime(date)
                plt.axvline(x=date, color='red', linestyle="--")

        plt.savefig('cat/flow/' + tag + '.png') 
        plt.close()


if __name__ == "__main__":

    # Loading the dataframe
    print("Loading the dataframe")
    # ddf = dd.read_csv('data/CGCE7K-1920-0503-Pivot15M-updated.csv', sample=1000000, assume_missing=True, parse_dates=['Time']).set_index('Time')
    df = pd.read_csv('../data/CGCE7K-1920-0503-Pivot15M-updated.csv', parse_dates=['Time'])
    df = df.set_index('Time')
    


    # relation(final, tags)
    # trendcat2(df)
    divide(df)


# print("Number of columns in tag :", len(critical_tags))
# print("Number of columns in tag2 :", len(tag2_columns))
# print("Number of columns in CGCE :", len(cgce_columns))
# print("Number of columns in CGCE7K :", len(cgce7k_columns))



# # Number of tags in common 
# n = 0
# for col in tag2_columns:
#     if col in cgce7k_columns:
#         n += 1

# print("Number of columns in common : ", n)

# # Correlation matrix
# df.corr(numeric_only=True)

# print(tag2.head(n=5).iloc[:, :5])



