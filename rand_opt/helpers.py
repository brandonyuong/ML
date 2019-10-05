import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from time import clock
from collections import defaultdict

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as ms


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


"""
def load_data(csv_file, dep_vars):

    #Load data from CSV file.  For multiple dependent vars

    #:param csv_file: (str) name of csv file
    #:param dep_vars: (int) number of dependent variables at end of file
    #:return: Data frames to be input into learning algorithms

    df = pd.read_csv(csv_file)
    col_index = list(df.columns.values)
    x = df.copy()
    for n in range(1, dep_vars + 1):
        dep_var_label = col_index[-1 * n]  # get label of the last column
        x.drop(columns=dep_var_label, axis=1, inplace=True)
    #len_df = len(df.columns)
    #y = df.iloc[:, len_df - dep_vars:len_df]
    y = df.iloc[:, -1]

    return x, y
"""


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=None, train_sizes=np.linspace(.1, 1., 10)):
    """
    Function retrieved from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    #sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#f92672")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#007fff")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="#f92672",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="#007fff",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(title + ".png")
    plt.close()


def plot_fit_times(estimator, title, x, y):
    out = defaultdict(dict)
    length_x = len(x)
    split_floats = np.linspace(.1, .9, 9)
    for split_float in split_floats:
        x_train, x_test, y_train, y_test = ms.train_test_split(x, y,
                                                               train_size=split_float)
        start_time = clock()
        clf = estimator
        clf.fit(x_train, y_train)
        out['train'][split_float] = clock() - start_time
        start_time = clock()
        clf.predict(x_test)
        out['test'][split_float] = clock() - start_time
    out = pd.DataFrame(out)
    print(out)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Time (s)")
    plt.grid()
    plt.plot(split_floats * length_x, out['test'], 'o-', color="#c1ffc1",
             label="Test set")
    plt.plot(split_floats * length_x, out['train'], 'o-', color="#50d3dc",
             label="Train set")
    plt.legend(loc="best")
    plt.savefig(title + ".png")
    plt.close()


def scale_features(input_df):
    scaler = StandardScaler()
    scaler.fit(input_df)
    scaled_features = scaler.transform(input_df)
    return pd.DataFrame(scaled_features)


def plot_nn_solver_fit_times(title, x, y, list_of_hps):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Time (s)")
    plt.grid()

    out = defaultdict(dict)
    length_x = len(x)
    split_floats = np.linspace(.1, .9, 9)
    for hps in list_of_hps:
        for split_float in split_floats:
            x_train, x_test, y_train, y_test = ms.train_test_split(x, y,
                                                                   train_size=split_float)
            start_time = clock()
            clf = MLPClassifier(solver=hps)
            clf.fit(x_train, y_train)
            out['Train'][split_float] = clock() - start_time
        out = pd.DataFrame(out)
        print(out)
        plt.plot(split_floats * length_x, out['Train'], 'o-', color=random_color(),
                 label=hps)

    plt.legend(loc="best")
    plt.savefig(title + ".png")
    plt.close()


def plot_nn_lr_fit_times(title, x, y, list_of_hps):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Time (s)")
    plt.grid()

    out = defaultdict(dict)
    length_x = len(x)
    split_floats = np.linspace(.1, .9, 9)
    for hps in list_of_hps:
        for split_float in split_floats:
            x_train, x_test, y_train, y_test = ms.train_test_split(x, y,
                                                                   train_size=split_float)
            start_time = clock()
            clf = MLPClassifier(learning_rate=hps)
            clf.fit(x_train, y_train)
            out['Train'][split_float] = clock() - start_time
        out = pd.DataFrame(out)
        print(out)
        plt.plot(split_floats * length_x, out['Train'], 'o-', color=random_color(),
                 label=hps)

    plt.legend(loc="best")
    plt.savefig(title + ".png")
    plt.close()


def random_color():
    return "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
