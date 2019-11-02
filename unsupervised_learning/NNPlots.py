import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import clock
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture


# the original sklearn implementation does not have a transform function
class BY_GMM(GaussianMixture):
    def transform(self, X):
        return self.predict_proba(X)


class NNPlots(object):
    def __init__(self, x_train, y_train, x_test, y_test, data_name, best_dr_n,
                 best_cluster_n):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.data_name = data_name
        self.dr_n = best_dr_n  # list of best n's for PCA, ICA, RP, LDA
        self.cluster_n = best_cluster_n

        self.plot_nn_with_dr()
        self.plot_nn_dr_fit_times()
        self.plot_nn_clusters()
        self.plot_nn_clusters_fit_times()

    def plot_nn_with_dr(self):
        n = self.dr_n
        acc = np.zeros(len(n) + 1)
        best_dr_n = self.dr_n
        bar_names = ['PCA', 'ICA', 'RP', 'LDA', 'Base NN']
        dim_reducers = [PCA(random_state=2), FastICA(random_state=2),
                        SparseRandomProjection(random_state=2),
                        LinearDiscriminantAnalysis()]

        model = MLPClassifier(random_state=2)
        model.fit(self.x_train, self.y_train)
        baseline = accuracy_score(model.predict(self.x_test), self.y_test)
        acc[len(n)] = baseline

        for i, num in enumerate(best_dr_n):
            dim_reducer = dim_reducers[i]
            dim_reducer.set_params(n_components=num)
            if i == 3:  # LDA
                transformed_x = dim_reducer.fit_transform(self.x_train, self.y_train)
            else:
                transformed_x = dim_reducer.fit_transform(self.x_train)
            model.fit(transformed_x, self.y_train)
            test = dim_reducer.transform(self.x_test)
            nn_acc = accuracy_score(model.predict(test), self.y_test)
            acc[i] = nn_acc

        x = np.arange(len(n) + 1)
        title = self.data_name + " NN with Dim Reduction"
        bar_colors = ['#7770ff', '#05acbf', '#f92672', '#007fff', '#95a3b0']

        plt.subplots()
        plt.title(title)
        plt.xlabel("Dim. Reduction Algorithms")
        plt.ylabel("Accuracy")
        plt.ylim(0.5, 1.)
        plt.bar(x, acc, color=bar_colors)
        plt.xticks(x, bar_names)
        plt.savefig("plots/" + title + ".png")
        plt.close()

    def plot_nn_dr_fit_times(self):
        n = self.dr_n
        fit_times = np.zeros(len(n) + 1)
        best_dr_n = self.dr_n
        bar_names = ['PCA', 'ICA', 'RP', 'LDA', 'Base NN']
        dim_reducers = [PCA(random_state=2), FastICA(random_state=2),
                        SparseRandomProjection(random_state=2),
                        LinearDiscriminantAnalysis()]

        model = MLPClassifier(random_state=2)
        start_time = clock()
        model.fit(self.x_train, self.y_train)
        fit_times[len(n)] = clock() - start_time

        for i, num in enumerate(best_dr_n):
            dim_reducer = dim_reducers[i]
            dim_reducer.set_params(n_components=num)
            if i == 3:  # LDA
                transformed_x = dim_reducer.fit_transform(self.x_train, self.y_train)
            else:
                transformed_x = dim_reducer.fit_transform(self.x_train)
            start_time = clock()
            model.fit(transformed_x, self.y_train)
            fit_times[i] = clock() - start_time

        x = np.arange(len(n) + 1)
        title = self.data_name + " NN Fit Times with Dim Reduction"
        bar_colors = ['#7770ff', '#05acbf', '#f92672', '#007fff', '#95a3b0']

        plt.subplots()
        plt.title(title)
        plt.xlabel("Dim. Reduction Algorithms")
        plt.ylabel("Fit Time (seconds)")
        plt.bar(x, fit_times, color=bar_colors)
        plt.xticks(x, bar_names)
        plt.savefig("plots/" + title + ".png")
        plt.close()

    def plot_nn_clusters(self):
        n = self.cluster_n
        acc = np.zeros(len(n) + 1)
        best_n = self.cluster_n
        bar_names = ['KM', 'EM', 'Base NN']
        clusterers = [KMeans(random_state=2), BY_GMM(random_state=2)]

        model = MLPClassifier(random_state=2)
        model.fit(self.x_train, self.y_train)
        baseline = accuracy_score(model.predict(self.x_test), self.y_test)
        acc[len(n)] = baseline

        for i, num in enumerate(best_n):
            clusterer = clusterers[i]
            if i == 0:  # KMeans
                clusterer.set_params(n_clusters=num)
            else:
                clusterer.set_params(n_components=num)
            clusterer.fit(self.x_train)
            transformed_x = clusterer.transform(self.x_train)
            model.fit(transformed_x, self.y_train)
            test = clusterer.transform(self.x_test)
            nn_acc = accuracy_score(model.predict(test), self.y_test)
            acc[i] = nn_acc

        x = np.arange(len(n) + 1)
        title = self.data_name + " NN with Clustering"
        bar_colors = ['#f92672', '#007fff', '#95a3b0']

        plt.subplots()
        plt.title(title)
        plt.xlabel("Clustering Algorithms")
        plt.ylabel("Accuracy")
        plt.ylim(0.5, 1.)
        plt.bar(x, acc, color=bar_colors)
        plt.xticks(x, bar_names)
        plt.savefig("plots/" + title + ".png")
        plt.close()

    def plot_nn_clusters_fit_times(self):
        n = self.cluster_n
        fit_times = np.zeros(len(n) + 1)
        best_n = self.cluster_n
        bar_names = ['KM', 'EM', 'Base NN']
        clusterers = [KMeans(random_state=2), BY_GMM(random_state=2)]

        model = MLPClassifier(random_state=2)
        start_time = clock()
        model.fit(self.x_train, self.y_train)
        fit_times[len(n)] = clock() - start_time

        for i, num in enumerate(best_n):
            clusterer = clusterers[i]
            if i == 0:  # KMeans
                clusterer.set_params(n_clusters=num)
            else:
                clusterer.set_params(n_components=num)
            clusterer.fit(self.x_train)
            transformed_x = clusterer.transform(self.x_train)
            start_time = clock()
            model.fit(transformed_x, self.y_train)
            fit_times[i] = clock() - start_time

        x = np.arange(len(n) + 1)
        title = self.data_name + " NN Fit Times with Clustering"
        bar_colors = ['#f92672', '#007fff', '#95a3b0']

        plt.subplots()
        plt.title(title)
        plt.xlabel("Clustering Algorithms")
        plt.ylabel("Fit Time (seconds)")
        plt.bar(x, fit_times, color=bar_colors)
        plt.xticks(x, bar_names)
        plt.savefig("plots/" + title + ".png")
        plt.close()
