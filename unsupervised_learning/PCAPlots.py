import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from unsupervised_learning.KMeansPlots import KMeansPlots


class PCAPlots(object):
    def __init__(self, x_train, data_name, n=4):
        self.x_train = x_train
        self.data_name = data_name
        self.n = n
        self.plot_explained_var_ratio()

    def plot_explained_var_ratio(self):
        X = self.x_train
        n = self.n
        range_n_components = np.arange(1, n + 1)
        pca = PCA(n_components=n, random_state=2)
        pca.fit(X)
        explained_var = pca.explained_variance_ratio_
        plt.figure()
        title = self.data_name + " PCA Explained Variance Ratio"
        plt.title(title)
        plt.xlabel("n-components")
        plt.ylabel("Explained Variance Ratio")
        plt.grid()

        plt.plot(range_n_components, explained_var, 'o-', color="#f92672")

        plt.savefig("plots/" + title + ".png")
        plt.close()

    def get_transformed_features(self, best_n):
        X = self.x_train
        pca = PCA(n_components=best_n, random_state=2)
        return pca.fit_transform(X)
