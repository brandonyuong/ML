from unsupervised_learning.helpers import *

import sklearn.model_selection as ms
from sklearn.neural_network import MLPClassifier
from classification_algs.MLPAnalysis import MLPAnalysis
from unsupervised_learning.KMeansPlots import KMeansPlots
from unsupervised_learning.EMPlots import EMPlots
from unsupervised_learning.PCAPlots import PCAPlots
from unsupervised_learning.ICAPlots import ICAPlots
from unsupervised_learning.RPPlots import RPPlots
from unsupervised_learning.LDAPlots import LDAPlots
from unsupervised_learning.NNPlots import NNPlots
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score


def main():

    # Load X + Y Data Sets
    x1, y1 = load_data('seeds_dataset.csv')
    x2, y2 = load_data('drivPoints.csv')

    scaled_x1 = scale_features(x1)
    scaled_x2 = scale_features(x2)
    """
    plot_learning_curve(MLPClassifier(), "Seeds Data NN", scaled_x1, y1)
    plot_fit_times(MLPClassifier(), "Seeds Data NN Fit Time", scaled_x1, y1)
    plot_learning_curve(MLPClassifier(), "Driver Data NN", scaled_x2, y2)
    plot_fit_times(MLPClassifier(), "Driver Data NN Fit Time", scaled_x2, y2)
    """
    print("*** Seeds Data ***")
    x1_train, x1_test, y1_train, y1_test = ms.train_test_split(
        scaled_x1, y1, train_size=120, random_state=2)
    #MLPAnalysis(x1_train, x1_test, y1_train, y1_test, random_state=2)

    print("*** Driver Face Data ***")
    x2_train, x2_test, y2_train, y2_test = ms.train_test_split(
        scaled_x2, y2, train_size=400, random_state=2)
    #MLPAnalysis(x2_train, x2_test, y2_train, y2_test, random_state=2)
    """
    print("*** KMeans ***")
    seeds_km = KMeansPlots(x1_train, y1_train, "Seeds", 6)
    driver_km = KMeansPlots(x2_train, y2_train, "Driver", 16)
    seeds_km.plot_silhouette()
    driver_km.plot_silhouette()

    print("*** EM ***")
    EMPlots(x1_train, y1_train, "Seeds", 6)
    EMPlots(x2_train, y2_train, "Driver", 16)
"""
    print("*** PCA ***")
    seeds_pca = PCAPlots(x1_train, "Seeds", 6)
    driver_pca = PCAPlots(x2_train, "Driver", 16)
    transformed = seeds_pca.get_transformed_features(3)
    KMeansPlots(transformed, y1_train, "Seeds with PCA", 6)
    EMPlots(transformed, y1_train, "Seeds with PCA", 6)
    transformed = driver_pca.get_transformed_features(6)
    KMeansPlots(transformed, y2_train, "Driver with PCA", 9)
    EMPlots(transformed, y2_train, "Driver with PCA", 9)
    """
    print("*** ICA ***")
    seeds_ica = ICAPlots(x1_train, "Seeds", 6)
    driver_ica = ICAPlots(x2_train, "Driver", 16)
    transformed = seeds_ica.get_transformed_features(6)
    KMeansPlots(transformed, y1_train, "Seeds with ICA", 9)
    EMPlots(transformed, y1_train, "Seeds with ICA", 9)
    transformed = driver_ica.get_transformed_features(10)
    KMeansPlots(transformed, y2_train, "Driver with ICA", 13)
    EMPlots(transformed, y2_train, "Driver with ICA", 13)
    """
    print("*** RP ***")
    seeds_rp = RPPlots(x1_train, y1_train, x1_test, y1_test, "Seeds", 6)
    driver_rp = RPPlots(x2_train, y2_train, x2_test, y2_test, "Driver", 16)
    transformed = seeds_rp.get_transformed_features(4)
    KMeansPlots(transformed, y1_train, "Seeds with RP", 7)
    EMPlots(transformed, y1_train, "Seeds with RP", 7)
    transformed = driver_rp.get_transformed_features(6)
    KMeansPlots(transformed, y2_train, "Driver with RP", 9)
    EMPlots(transformed, y2_train, "Driver with RP", 9)
    """
    print("*** LDA ***")
    seeds_lda = LDAPlots(x1_train, y1_train, x1_test, y1_test, "Seeds", 6)
    driver_lda = LDAPlots(x2_train, y2_train, x2_test, y2_test, "Driver", 16)
    
    transformed = seeds_lda.get_transformed_features(2)
    KMeansPlots(transformed, y1_train, "Seeds with LDA", 5)
    EMPlots(transformed, y1_train, "Seeds with LDA", 5)
    transformed = driver_lda.get_transformed_features(3)
    KMeansPlots(transformed, y2_train, "Driver with LDA", 6)
    EMPlots(transformed, y2_train, "Driver with LDA", 6)

    print("*** NN ***")
    NNPlots(x1_train, y1_train, x1_test, y1_test, "Seeds", [3, 6, 4, 2], [4, 3])
    NNPlots(x2_train, y2_train, x2_test, y2_test, "Driver", [6, 10, 6, 3], [4, 8])
"""

if __name__ == '__main__':
    main()
