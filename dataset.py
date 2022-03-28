import pandas as pd
from sklearn.model_selection import train_test_split


def open_dataset(ticker):
    df = pd.read_csv("../mercury/"+ticker+".csv", header=None, index_col=None, delimiter=',')
    train_df, test_df = train_test_split(df, test_size=0.20)
    return train_df, test_df
