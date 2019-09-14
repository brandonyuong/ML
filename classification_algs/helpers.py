import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler


def load_data(csv_file):
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
    return x, y


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
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


def scale_features(input_df):
    scaler = StandardScaler()
    scaler.fit(input_df)
    scaled_features = scaler.transform(input_df)
    return pd.DataFrame(scaled_features)
