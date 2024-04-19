import pandas as pd 

full = pd.read_csv('~/CGCE-2019-PivotHourCombined.csv')
critical = pd.read_csv('../data/Critical_Tag.csv')
failure_date = pd.read_csv('../data/Failure_date.csv')

for sensor in full.columns:
    if "LZAHH6007" in sensor:
        print(sensor)

for index, line in full.iterrows():
    if line['Time'] == "2019-05-25 09:00:00":
        print(line["PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SLZAHH6007.PNT"])
        print(line['PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH60071.CIN'])
        print(line['PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_LZAHH6007D.CIN'])
        
        # print(line['PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_PZALL2453D.CIN'])
        # print(line['PC.PMO.DULG-DP-B.DCS.CGCE_CSDP:B_PZALL24531.CIN'])
        # print(line['PC.PMO.DULG-DP-B.DCS.CSDP_CGCE:SPZALL2453.PNT'])
        

