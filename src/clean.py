import pandas as pd

df = pd.read_csv('../data/CGCE-2019-PivotHourCombined.csv')
critical = pd.read_csv('../data/Critical_Tag.csv')
failure_date = pd.read_csv('../data/Failure_date.csv')

# Removing duplicate values

def remove_duplicate():
    