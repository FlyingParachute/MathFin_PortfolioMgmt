import pandas as pd
from utils.excess_ret_comp import direct_substract
from config import DATA_PATH, MIN_MONTHS

def load_and_clean_data():
    df = pd.read_csv(DATA_PATH,low_memory=False)
    df = direct_substract(df)
    df = df.dropna(subset=['RETX'])
    df = df[~df['RETX'].isin(['B','C'])]
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby('PERMNO').filter(lambda x: len(x) >= MIN_MONTHS)
    df = df.sort_values(['PERMNO','date']).reset_index(drop=True)
    return df
