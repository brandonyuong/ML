import csv
import sklearn.model_selection as ms

from rand_opt.helpers import *


def data_creation():

    # Load Data Set
    x, y = load_trunc_data('purchase_intent.csv', 1000)
    scaled_x = scale_features(x)
    x_train, x_test, y_train, y_test = ms.train_test_split(
        scaled_x, y, train_size=0.80, random_state=0)

    # Export to csv
    x_train.to_csv('x_train.csv', index=False, header=False)
    x_test.to_csv('x_test.csv', index=False, header=False)
    y_train.to_csv('y_train.csv', index=False, header=False)
    y_test.to_csv('y_test.csv', index=False, header=False)


if __name__ == '__main__':
    data_creation()
