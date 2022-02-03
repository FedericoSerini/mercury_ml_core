import pandas as pd
from sklearn.model_selection import train_test_split


def open_dataset():
    df = pd.read_csv("./data/res.csv", header=None, index_col=None, delimiter=',')
    train_df, test_df = train_test_split(df, test_size=0.20)
    return train_df, test_df
