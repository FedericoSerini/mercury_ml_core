import pandas as pd
from sklearn.utils import shuffle

def normalize_data(train_df, test_df):
    # drop nan values
    train_df = train_df.dropna(axis=0)
    test_df = test_df.dropna(axis=0)

    # drop first 15 row
    train_df = train_df.iloc[15:, :]
    test_df = test_df.iloc[15:, :]

    label_hold_size = train_df.loc[train_df[0] == 0].shape[0]
    label_buy_size = train_df.loc[train_df[0] == 1].shape[0]
    label_sell_size = train_df.loc[train_df[0] == 2].shape[0]

    l0_l1_ratio = (label_hold_size // label_buy_size)
    l0_l2_ratio = (label_hold_size // label_sell_size)
    print("Before")
    print("l0_size:", label_hold_size, "l1_size:", label_buy_size, "l2_size:", label_sell_size)
    print("l0_l1_ratio:", l0_l1_ratio, "l0_l2_ratio:", l0_l2_ratio)

    l1_new = pd.DataFrame()
    l2_new = pd.DataFrame()
    for idx, row in train_df.iterrows():
        if row[0] == 1:
            for i in range(l0_l1_ratio):
                l1_new = l1_new.append(row)
        if row[0] == 2:
            for i in range(l0_l2_ratio):
                l2_new = l2_new.append(row)

    train_df = train_df.append(l1_new)
    train_df = train_df.append(l2_new)

    # shuffle
    train_df = shuffle(train_df)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    return train_df, test_df
