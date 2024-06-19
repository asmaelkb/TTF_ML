import pandas as pd
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer

def preprocess_data(tags):
    db_config = {
        'user': 'intern_asmae',
        'password': 'asmae123',
        'host': '172.20.0.2',
        'database': 'mysql_server'
    }

    engine = create_engine(f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")

    tags_condition = " OR ".join([f"DulangTagList.tag = '{tag}'" for tag in tags])

    query = "SELECT tag, value, time FROM CGCE7K NATURAL JOIN DulangTagList WHERE " + tags_condition
    df = pd.read_sql(query, engine)
    
    # Impute nan values by the mean of the tag
    df['avg_value'] = df.groupby('tag')['value'].transform('mean')
    df['value'] = df['value'].fillna(df['avg_value'])
    df.drop(columns=['avg_value', 'tag'], inplace=True)
    df = df.set_index('time')

    return df