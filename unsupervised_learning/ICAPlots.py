import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class ICAPlots(object):
    def __init__(self, x_train, y_train, x_test, y_test, data_name, n=4):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.data_name = data_name
        self.n = n
        self.plot_kurtosis()
        self.plot_rf_accuracy()

    def plot_rf_accuracy(self):
        n = self.n
        acc = np.zeros(n)
        range_n_components = np.arange(1, n + 1)
        for i, j in enumerate(range_n_components):
            dim_reducer = FastICA(n_components=j)
            transformed_x = dim_reducer.fit_transform(self.x_train, self.y_train)
            model = RandomForestClassifier()
            model.fit(transformed_x, self.y_train)
            test = dim_reducer.transform(self.x_test)
            rf_acc = accuracy_score(model.predict(test), self.y_test)
            acc[i] = rf_acc

        model = RandomForestClassifier(n_estimators=10)
        model.fit(self.x_train, self.y_train)
        baseline = accuracy_score(model.predict(self.x_test), self.y_test)

        plt.figure()
        title = self.data_name + " ICA Random Forest Performance"
        plt.title(title)
        plt.xlabel("n-components")
        plt.ylabel("Accuracy")
        plt.grid()

        plt.plot(range_n_components, acc, 'o-', color="#f92672",
                 label='ICA')
        plt.plot(range_n_components, [baseline] * len(acc), color="#007fff",
                 label='Baseline Model')

        plt.savefig("plots/" + title + ".png")
        plt.close()

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

