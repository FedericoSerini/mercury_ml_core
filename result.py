import numpy as np
import pandas as pd


def print_results(predictions, test_labels, test_prices):

    result_df = pd.DataFrame({"prediction": np.argmax(predictions, axis=1),
                          "test_label": np.argmax(test_labels, axis=1),
                         "test_price": test_prices})

    result_df.to_csv("./data/cnn_result.csv", sep=';', index=None)