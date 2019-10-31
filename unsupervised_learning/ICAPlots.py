import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA


class ICAPlots(object):
    def __init__(self, x_train, data_name, n=4):
        self.x_train = x_train
        self.data_name = data_name
        self.n = n
        self.plot_kurtosis()

    def plot_kurtosis(self):
        X = self.x_train
        n = self.n
        kurt = np.zeros(n)
        range_n_components = np.arange(1, n + 1)
        for i, j in enumerate(range_n_components):
            ica = FastICA(n_components=j, random_state=2)
            transformed_x = ica.fit_transform(X)
            transformed_x = pd.DataFrame(transformed_x)
            transformed_x = transformed_x.kurt(axis=0)
            kurt[i] = transformed_x.abs().mean()

        plt.figure()
        title = self.data_name + " ICA Average Kurtosis"
        plt.title(title)
        plt.xlabel("n-components")
        plt.ylabel("Average Kurtosis")
        plt.grid()

        plt.plot(range_n_components, kurt, 'o-', color="#f92672")

        plt.savefig("plots/" + title + ".png")
        plt.close()

    def get_transformed_features(self, best_n):
        X = self.x_train
        ica = FastICA(n_components=best_n, random_state=2)
        return ica.fit_transform(X)

