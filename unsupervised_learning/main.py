from unsupervised_learning.helpers import *

import sklearn.model_selection as ms
from sklearn.neural_network import MLPClassifier
from classification_algs.MLPAnalysis import MLPAnalysis
from unsupervised_learning.KMeansPlots import KMeansPlots
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score


def main():

    # Load X + Y Data Sets
    x1, y1 = load_data('seeds_dataset.csv')
    x2, y2 = load_data('drivPoints.csv')

    scaled_x1 = scale_features(x1)
    scaled_x2 = scale_features(x2)

    #plot_learning_curve(MLPClassifier(), "Seeds Data NN", scaled_x1, y1)
    #plot_fit_times(MLPClassifier(), "Seeds Data NN Fit Time", scaled_x1, y1)

    print("*** Seeds Data ***")
    x1_train, x1_test, y1_train, y1_test = ms.train_test_split(
        scaled_x1, y1, train_size=0.8, random_state=2)
    #MLPAnalysis(x1_train, x1_test, y1_train, y1_test)

    print("*** Driver Face Data ***")
    x2_train, x2_test, y2_train, y2_test = ms.train_test_split(
        scaled_x2, y2, train_size=300)
    #MLPAnalysis(x2_train, x2_test, y2_train, y2_test)

    print("*** KMeans ***")

    #KMeansPlots(x1_train, y1_train, "Seeds", 6)
    #KMeansPlots(x2_train, y2_train, "Driver Face", 10)


if __name__ == '__main__':
    main()
