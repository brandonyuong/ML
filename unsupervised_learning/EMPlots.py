import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


class EMPlots(object):
    def __init__(self, x_train, data_name, n=4):
        self.X = x_train
        self.title = data_name
        self.plot_bic(self.X, n)
        self.plot_aic(self.X, n)

    def plot_bic(self, X, n):
        np_bic = np.zeros(n)
        range_n_components = np.arange(1, n + 1)
        for i, j in enumerate(range_n_components):
            gmm = GaussianMixture(n_components=j, random_state=2)
            gmm.fit(X)
            bic = gmm.bic(X)
            print("n=" + str(j) + " BIC: " + str(bic))
            np_bic[i] = bic
        plt.figure()
        title = self.title + " Bayesian Info Criterion"
        plt.title(title)
        plt.xlabel("n-components")
        plt.ylabel("BIC")
        plt.grid()

        plt.plot(range_n_components, np_bic, 'o-', color="#f92672")

        plt.savefig("plots/" + title + ".png")
        plt.close()

    def plot_aic(self, X, n):
        np_aic = np.zeros(n)
        range_n_components = np.arange(1, n + 1)

        for i, j in enumerate(range_n_components):
            gmm = GaussianMixture(n_components=j, random_state=2)
            gmm.fit(X)
            aic = gmm.aic(X)
            print("n=" + str(j) + " AIC: " + str(aic))
            np_aic[i] = aic

        plt.figure()
        title = self.title + " AIC"
        plt.title(title)
        plt.xlabel("n-components")
        plt.ylabel("AIC")
        plt.grid()

        plt.plot(range_n_components, np_aic, 'o-', color="#007fff")

        plt.savefig("plots/" + title + ".png")
        plt.close()
