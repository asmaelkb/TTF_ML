import csv
import mysql.connector
import pandas as pd

# Connection to the MySQL Database
conn = mysql.connector.connect(
    host="localhost",
    user="intern_asmae",
    password="asmae123",
    database="mysql_server"
)
cursor = conn.cursor()

csv_filename = 'app/data/CGCE7K-1920-0503-Pivot15M-updated.csv'
table_name = 'cgce_7k'


df = pd.read_csv(csv_filename, parse_dates=['Time'])
df = df.set_index('Time')

# Divide in 3 parts
tags_part1 = df.iloc[:, :2500]
tags_part2 = df.iloc[:, 2500:5000]
tags_part3 = df.iloc[:, 5000:]

# Table creation
def create_table(table_name, df):
    headers = df.columns

    sql_query = f"CREATE TABLE IF NOT EXISTS {table_name} (id INT AUTO_INCREMENT PRIMARY KEY, timestamp DATETIME NOT NULL,"
    for header in headers:  
        
        newheader = header.replace(" ", "")
        if "PC.PM" in header:
            newheader = newheader.replace("PC.PM", "")
        
        sql_query += f"`{newheader}` FLOAT, "  

    sql_query = sql_query[:-2] + ", UNIQUE KEY unique_timestamp (timestamp))"  # Unicity constraint
    cursor.execute(sql_query)

# Function to insert datas into a table
def insert_data(table_name, df_part):
    columns = ', '.join(df_part.columns)
    values = ', '.join(['?' for _ in df_part.columns])
    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
    for _, row in df_part.iterrows():
        cursor.execute(insert_query, *row)
    conn.commit()

create_table('TagsPart1', tags_part1)
create_table('TagsPart2', tags_part2)
create_table('TagsPart3', tags_part3)

insert_data('TagsPart1', tags_part1)
insert_data('TagsPart2', tags_part2)
insert_data('TagsPart3', tags_part3)

# Validation of changes
conn.commit()

# Closing the connection
cursor.close()
conn.close()
