import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import StandardScaler


def load_data(csv_file):
    """
    Load data from CSV file.  The results vector must be in the last column!

    :param csv_file: String name of csv file
    :return: Data frames to be input into learning algorithms
    """
    df = pd.read_csv(csv_file)
    col_index = list(df.columns.values)
    result_label = col_index[-1]  # get label of the last column
    x = df.drop(columns=result_label, axis=1)
    y = df.iloc[:, -1]
    return x, y


def load_trunc_data(csv_file, max_samples):
    """
    Load data from CSV file and randomly trim.
    The results vector must be in the last column!

    :param csv_file: String name of csv file
    :param max_samples: (int) max amount of samples/rows
    :return: Data frames to be input into learning algorithms
    """

    df = pd.read_csv(csv_file)
    trunc_df = df.sample(max_samples)
    col_index = list(df.columns.values)
    result_label = col_index[-1]  # get label of the last column
    x = trunc_df.drop(columns=result_label, axis=1)
    y = trunc_df.iloc[:, -1]
    return x, y


# Plot onto an open plot figure
def plot_rand_opt(X, y, plot_label, custom_color=None, marker="o"):
    if custom_color is None:
        custom_color = random_color()
    plt.grid()
    plt.plot(X, y, 'o-', color=custom_color, label=plot_label, marker=marker)
    plt.legend(loc="best")


def scale_features(input_df):
    scaler = StandardScaler()
    scaler.fit(input_df)
    scaled_features = scaler.transform(input_df)
    return pd.DataFrame(scaled_features)


def random_color():
    return "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])


def avg_list(input_list):
    return sum(input_list) / float(len(input_list))
