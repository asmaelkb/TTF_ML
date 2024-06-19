import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Values of the tag in function of the hours of the day of the shutdown

def histo(df, date):
    
    tags = ["PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH2417B.PNT", "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:LZT2417B_EN.PNT"]
    for tag in tags:
        data = []
        x = []
        for i in range(2, 24): # 2-hours delay
            dateh = date + ' ' + str(i).zfill(2) + ':00:00'
            value = df.loc[df['Time'] == dateh, tag]

            if not value.isnull().any():
                data.append(value.iloc[0])
                x.append(i)
        

        plt.bar(x, data, color="green")

        # sns.regplot(x=x, y=data, scatter=False, color='red')

        z = np.polyfit(x, data, 1)
        p = np.poly1d(z)

        plt.plot(x, p(x), "r--")

        plt.xlabel('Hours')
        plt.ylabel('Values of the sensor')
        plt.title('Value of ' + tag + " \nduring " + date, fontsize=10)

        plt.savefig(date + "_" + tag + ".png")  

def hist_year(df):
    # Plot the values of all these tags 
    tags = ["PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:PZT2468_EN.PNT", 
            "PC.PMO.DLB.TCP.CGCE.CD563.FN_BN3500_SD",
            "PC.PMO.DULG-DP-B.DCS.CGCE_TCP_DI:B_BN3500_SD.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH60071.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6007D.CIN",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH6007.PNT",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:LZT2417B_EN.PNT",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH2417B.PNT",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6050.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH60501.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6050A.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6050B.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:BLZAHH6050A1.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:BLZAHH6050B1.CIN",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH6050A.PNT",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH6050B.PNT"]
    

    nrow = df.shape[0]
    col = ["2019-01", "2019-03", "2019-05", "2019-07", "2019-09", "2019-11"]

    for tag in tags:

        value = df.loc[0:nrow, tag]
        data = [x for x in value if not pd.isnull(x)]

        print(np.mean(data))

        # Plot for this tag
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.hist(data, color="green", alpha=0.5)

        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())

        x = np.arange(len(data))
        z = np.polyfit(x, data, 1)
        p = np.poly1d(z)

        ax.plot(x, p(x), color="red")

        # Adding labels and title
        plt.ylabel('Values of the sensor')
        plt.title('Values of the sensor \n' + tag + ' during 2019')

        espacement = fig.get_figwidth() / len(col)
        positions = [i * espacement for i in range(len(col))]
        # plt.xticks(positions, col) 

        plt.savefig("hist/" + tag + ".png") 
        plt.close()

def plot_year(df):
    # Plot the values of all these tags 
    tags = ["PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:PZT2468_EN.PNT", 
            "PC.PMO.DLB.TCP.CGCE.CD563.FN_BN3500_SD",
            "PC.PMO.DULG-DP-B.DCS.CGCE_TCP_DI:B_BN3500_SD.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH60071.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6007D.CIN",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH6007.PNT",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:LZT2417B_EN.PNT",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH2417B.PNT",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6050.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH60501.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6050A.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6050B.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:BLZAHH6050A1.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:BLZAHH6050B1.CIN",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH6050A.PNT",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH6050B.PNT"]
    
    for tag in tags:
        df[tag] = df[tag].dropna()
        plt.plot(df['Time'], df[tag])

        # Adding labels and title
        plt.ylabel('Values of the sensor')
        plt.title('Values of the sensor \n' + tag + ' during 2019')

        plt.savefig("hist/" + tag + "NEW.png") 


def hist_year2(df):
    # Number of non-null values for each month
    data = {}

    tags = ["PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:PZT2468_EN.PNT", 
            "PC.PMO.DLB.TCP.CGCE.CD563.FN_BN3500_SD",
            "PC.PMO.DULG-DP-B.DCS.CGCE_TCP_DI:B_BN3500_SD.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH60071.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6007D.CIN",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH6007.PNT",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:LZT2417B_EN.PNT",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH2417B.PNT",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6050.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH60501.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6050A.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6050B.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:BLZAHH6050A1.CIN",
            "PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:BLZAHH6050B1.CIN",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH6050A.PNT",
            "PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH6050B.PNT"]

    for tag in tags:
        for i in range(1,13):
            data[str(i).zfill(2)] = 0

        for col in df['Time']:

            month = col[5:7]

            value = df.loc[df['Time'] == col, tag]
            data[month] += 0 if value.isnull().any() else 1

        print(data.values())
        plt.bar(data.keys(), data.values(), alpha=0.5)


        x = np.arange(len(data.values()))
        z = np.polyfit(x, list(data.values()), 1)
        p = np.poly1d(z)

        plt.plot(x, p(x), color="red")
        plt.xlabel("Months")
        plt.title('General trend of \n' + tag)

        plt.savefig("bar/" + tag + ".png") 
        plt.close()

        


if __name__ == "__main__":

    df = pd.read_csv('../data/CGCE-2019-PivotHourCombined.csv')
    date = '2019-06-19'
    
    # plot_year(df)
    hist_year2(df)

    # histo(df, date)
    # histo_sensors(df, date + " 12:00:00")
