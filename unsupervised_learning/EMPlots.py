import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, v_measure_score


class EMPlots(object):
    def __init__(self, x_train, y_train, data_name, n=4):
        self.x_train = x_train
        self.y_train = y_train
        self.title = data_name
        self.n = n
        self.plot_bic()
        self.plot_mutual_info()

    def plot_mutual_info(self):
        X = self.x_train
        y = self.y_train
        n = self.n

        mi_list = []
        range_n_clusters = range(2, n + 1)
        for n in range_n_clusters:
            clusterer = GaussianMixture(n_components=n, random_state=2)
            cluster_labels = clusterer.fit_predict(X)
            mi = normalized_mutual_info_score(y, cluster_labels)
            print("k=" + str(n) + " EM Normalized Mutual Info Score: " + str(mi))
            mi_list.append(mi)
        plt.figure()
        title = self.title + " EM Normalized Mutual Info"
        plt.title(title)
        plt.xlabel("n-clusters")
        plt.ylabel("Mutual Info Score")
        plt.grid()

        plt.plot(range_n_clusters, mi_list, 'o-', color="#f92672")

        plt.savefig("plots/" + title + ".png")
        plt.close()

    def plot_v(self):
        X = self.x_train
        y = self.y_train
        n = self.n

        v_list = []
        range_n_clusters = range(2, n + 1)
        for n in range_n_clusters:
            clusterer = GaussianMixture(n_components=n, random_state=2)
            cluster_labels = clusterer.fit_predict(X)
            v_score = v_measure_score(y, cluster_labels)
            print("k=" + str(n) + " EM V Score: " + str(v_score))
            v_list.append(v_score)
        plt.figure()
        title = self.title + " EM V Score"
        plt.title(title)
        plt.xlabel("n-clusters")
        plt.ylabel("V Score")
        plt.grid()

        plt.plot(range_n_clusters, v_list, 'o-', color="#f92672")

        plt.savefig("plots/" + title + ".png")
        plt.close()

    def plot_bic(self):
        X = self.x_train
        n = self.n
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

    def plot_aic(self):
        X = self.x_train
        n = self.n
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
