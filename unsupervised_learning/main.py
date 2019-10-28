from unsupervised_learning.helpers import *

import sklearn.model_selection as ms
from sklearn.neural_network import MLPClassifier
from classification_algs.MLPAnalysis import MLPAnalysis


def main():

    # Load X + Y Data Sets
    x1, y1 = load_data('banknote_auth.csv')
    x2, y2 = load_data('Skin_NonSkin.csv')

    scaled_x1 = scale_features(x1)
    scaled_x2 = scale_features(x2)

    # plot_learning_curve(MLPClassifier(), "Bank Auth Data NN", scaled_x1, y1)
    # plot_fit_times(MLPClassifier(), "Bank Auth Data Fit Time", scaled_x1, y1)

    x1_train, x1_test, y1_train, y1_test = ms.train_test_split(
        scaled_x1, y1, train_size=40, random_state=2)

    print("*** Bank Auth Data ***")
    MLPAnalysis(x1_train, x1_test, y1_train, y1_test)

    x2_train, x2_test, y2_train, y2_test = ms.train_test_split(
        scaled_x2, y2, train_size=40, random_state=2)

    print("*** Skin Data ***")
    MLPAnalysis(x2_train, x2_test, y2_train, y2_test)


if __name__ == '__main__':
    main()
