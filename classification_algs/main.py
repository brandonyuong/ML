import pandas as pd
import sklearn.model_selection as ms
from classification_algs.DecisionTreeAnalysis import DecisionTreeAnalysis


def load_data(csv_file, **kwargs):

    df = pd.read_csv(csv_file)
    x = attr_from_csv(csv_file)  # attribute columns
    y = df.iloc[:, -1]  # results column
    x_train, x_test, y_train, y_test = ms.train_test_split(x, y, **kwargs)
    return x_train, x_test, y_train, y_test


def attr_from_csv(path):
    df = pd.read_csv(path, nrows=1)  # read just first line for columns
    columns = df.columns.tolist()  # get the columns
    cols_to_use = columns[:len(columns) - 1]  # drop the last one
    df = pd.read_csv(path, usecols=cols_to_use)
    return df


def main():
    x_train, x_test, y_train, y_test = load_data('csv_result-PhishingData.csv',
                                                 test_size=0.80, random_state=0)
    DecisionTreeAnalysis(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
