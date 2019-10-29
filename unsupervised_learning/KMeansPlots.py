import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import normalized_mutual_info_score


class KMeansPlots(object):

    def __init__(self, x_train, y_train, data_name, n=4):
        self.title = data_name
        self.X = x_train
        self.y = y_train
        self.plot_silhouette(self.X, n)
        self.plot_mutual_info(self.X, self.y, n)

    def plot_mutual_info(self, X, y, n):
        mi_list = []
        range_n_clusters = range(2, n + 1)
        for n in range_n_clusters:
            clusterer = KMeans(n_clusters=n, random_state=2)
            cluster_labels = clusterer.fit_predict(X)
            mi = normalized_mutual_info_score(y, cluster_labels)
            print("k=" + str(n) + " Normalized Mutual Info Score: " + str(mi))
            mi_list.append(mi)
        plt.figure()
        title = self.title + " Normalized Mutual Info"
        plt.title(title)
        plt.xlabel("n-clusters")
        plt.ylabel("Mutual Info Score")
        plt.grid()

        plt.plot(range_n_clusters, mi_list, 'o-', color="#f92672")

        plt.savefig("plots/" + title + ".png")
        plt.close()

    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    def plot_silhouette(self, X, n):
        range_n_clusters = range(2, n + 1)
        mean_silhouette = []

        for n_clusters in range_n_clusters:
            fig, ax1 = plt.subplots()

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=2)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            mean_silhouette.append(silhouette_avg)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_xlabel("Silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            title = self.title + " Silhouette Analysis k=" + str(n_clusters)
            plt.title(title, fontsize=13)
            plt.savefig("plots/" + title + ".png")
            plt.close()

        plt.figure()
        title = self.title + " Silhouette Mean"
        plt.title(title)
        plt.xlabel("n-clusters")
        plt.ylabel("Mean of Silhouette Value")
        plt.grid()
        plt.plot(range_n_clusters, mean_silhouette, 'o-', color="#007fff")
        plt.savefig("plots/" + title + ".png")
        plt.close()
