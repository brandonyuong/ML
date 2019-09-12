import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

import sklearn.model_selection as ms
from sklearn.model_selection import learning_curve

from sklearn.tree import DecisionTreeClassifier
from classification_algs.DT_Analysis import DT_Analysis
from classification_algs.KNN_Analysis import KNN_Analysis


def load_split_data(csv_file, **kwargs):
    """
    Load data from CSV file.  The results vector must be in the last column!

    :param csv_file: String name of csv file
    :param kwargs: Optional arguments for train_test_split()
    :return: Data frames to be input into learning algorithms
    """
    df = pd.read_csv(csv_file)
    col_index = list(df.columns.values)
    result_label = col_index[-1]  # get label of the last column
    x = df.drop(columns=result_label, axis=1)
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = ms.train_test_split(x, y, **kwargs)
    return x, y, x_train, x_test, y_train, y_test


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.15, 0.95, 9)):
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
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(title + ".png")

    return plt


def main():
    x, y, x_train, x_test, y_train, y_test = load_split_data('PhishingData.csv',
                                                             test_size=0.80,
                                                             random_state=0)

    """
    *** History ***
    
    DT_Analysis(x_train, x_test, y_train, y_test)
    plot_learning_curve(DecisionTreeClassifier(), "Phishing Data DT", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_depth=3),
                        "Phishing Data DT: Max Depth 3", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_depth=6),
                        "Phishing Data DT: Max Depth 6", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_depth=9),
                        "Phishing Data DT: Max Depth 9", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_features='sqrt'),
                        "Phishing Data DT: (sqrt n) Max Features", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_features='log2'),
                        "Phishing Data DT: (log2 n) Max Features", x, y)
    
    KNN_Analysis(x_train, x_test, y_train, y_test)
    
    """


if __name__ == '__main__':
    main()
