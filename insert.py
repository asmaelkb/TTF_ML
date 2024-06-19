from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from models import Base, DulangTagList, CGCE7K
import pandas as pd
import numpy as np

# Connexion to the database
engine = create_engine('mysql+pymysql://intern_asmae:asmae123@172.16.239.131:3306/mysql_server')
Session = sessionmaker(bind=engine)
session = Session()

# # Load CSV data into a dataframe
# df = pd.read_csv('app/data/DulangTagList.csv')

# df = df.replace(np.nan, None)

# # Insert DulangTagList dataframe data into the corresponding table
# for index, row in df.iterrows():
#     record = DulangTagList(
#         tag=row['Tag'],
#         description=row['Description'],
#         type=row['Type'],
#         area=row['Area'],
#         category=row['Category'],
#         sapmapping=row['SAP MAPPING '],
#         upperlimit=row['Upper Limit'],
#         lowlimit=row['Low Limit']
#     )
#     try:
#         session.add(record)
#         session.commit()
#     except IntegrityError:
#         session.rollback()  # Cancel the changes
#         print(f"Duplicate entry for tag: {row['Tag']}. Skipping insertion.")



# Load CSV data into a dataframe
df = pd.read_csv('app/data/CGCE7K-1920-0503-Pivot15M-updated.csv', parse_dates=True)
df = df.set_index('Time')

# Avoiding nan values
df = df.replace(np.nan, None)

last_inserted_time = session.query(func.max(CGCE7K.time)).scalar()
if last_inserted_time is None:
    last_inserted_time = pd.Timestamp.min 

df.index = pd.to_datetime(df.index)
df = df[df.index > last_inserted_time]
end_date = pd.to_datetime('2019-06-30 23:00:00')
df = df[df.index < end_date]

# Mapping each tag with its ID
tag_map = {tag.tag: tag.tagID for tag in session.query(DulangTagList).all()}

for time_val, row in df.iterrows():
    for tag, value_val in row.items():
        tagID_val = tag_map.get(tag)
        if tagID_val is not None:
            record = CGCE7K(
                tagID=tagID_val,
                value=value_val,
                time=time_val
            )
            session.add(record)
            session.commit()

session.close()