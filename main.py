from model import start_train
from normalizer import normalize_data
from dataset import open_dataset
from result import print_results

parameters = {"input_w": 15, "input_h": 15, "num_classes": 3, "batch_size": 1, "epochs": 5}

train_df, test_df = open_dataset()
train_df, test_df = normalize_data(train_df, test_df)
predictions, test_labels, test_prices = start_train(train_df, test_df, parameters)
print_results(predictions, test_labels, test_prices)