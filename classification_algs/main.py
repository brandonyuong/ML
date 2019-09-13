from classification_algs.helpers import *
from sklearn.tree import DecisionTreeClassifier
from classification_algs.DTAnalysis import DTAnalysis
from sklearn.neighbors import KNeighborsClassifier
from classification_algs.KNNAnalysis import KNNAnalysis
from sklearn.neural_network import MLPClassifier
from classification_algs.MLPAnalysis import MLPAnalysis
from sklearn.linear_model import SGDClassifier
from classification_algs.SVMAnalysis import SVMAnalysis
from sklearn.ensemble import AdaBoostClassifier
from classification_algs.BoostAnalysis import BoostAnalysis


def main():
    x_train, x_test, y_train, y_test = load_split_data('PhishingData.csv',
                                                       test_size=0.80,
                                                       random_state=0)
    x, y = load_data('PhishingData.csv')
    scaled_x = scale_features(x)

    """
    *** History ***
    
    DTAnalysis(x_train, x_test, y_train, y_test)
    plot_learning_curve(DecisionTreeClassifier(), "Phishing Data DT", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_depth=3),
                        "Phishing Data DT: Max Depth 3", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_depth=6),
                        "Phishing Data DT: Max Depth 6", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_depth=9),
                        "Phishing Data DT: Max Depth 9", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_features='sqrt'),
                        "Phishing Data DT: (sqrt n) Max Features", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_features='log2'),
                        "Phishing Data DT: (log2 n) Max Features", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_leaf_nodes=2),
                        "Phishing Data DT: Max Leaf Nodes = 2", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_leaf_nodes=5),
                        "Phishing Data DT: Max Leaf Nodes = 5", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_leaf_nodes=8),
                        "Phishing Data DT: Max Leaf Nodes = 8", x, y)
    
    KNNAnalysis(x_train, x_test, y_train, y_test)
    plot_learning_curve(KNeighborsClassifier(n_neighbors=2),
                        "Phishing Data KNN: k = 2", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(n_neighbors=5),
                        "Phishing Data KNN: k = 5", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(n_neighbors=8),
                        "Phishing Data KNN: k = 8", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(weights='distance'),
                        "Phishing Data KNN: weights='distance'", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(p=1),
                        "Phishing Data KNN: p=1 Manhattan Distance", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(weights='distance', p=1),
                        "Phishing Data KNN: weights='distance', p=1", scaled_x, y)
    
    MLPAnalysis(x_train, x_test, y_train, y_test)
    plot_learning_curve(MLPClassifier(), "Phishing Data NN (unscaled)", x, y)
    plot_learning_curve(MLPClassifier(), "Phishing Data NN", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='identity'),
                        "Phishing Data NN: activation='identity'", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='logistic'),
                        "Phishing Data NN: activation='logistic'", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='tanh'),
                        "Phishing Data NN: activation='tanh'", scaled_x, y)
    
    
    SVMAnalysis(x_train, x_test, y_train, y_test)
    plot_learning_curve(SGDClassifier(), "Phishing Data SVM (unscaled)", x, y)
    plot_learning_curve(SGDClassifier(), "Phishing Data SVM", scaled_x, y)
    
    BoostAnalysis(x_train, x_test, y_train, y_test)
    plot_learning_curve(AdaBoostClassifier(), "Phishing Data Boosting (unscaled)", x, y)
    plot_learning_curve(AdaBoostClassifier(), "Phishing Data Boosting", scaled_x, y)
    """
    plot_learning_curve(MLPClassifier(activation='relu', hidden_layer_sizes=10,
                                      solver='lbfgs'),
                        "Phishing Data NN: Test'", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='logistic', hidden_layer_sizes=12,
                                      solver='lbfgs'),
                        "Phishing Data NN: Test 2'", scaled_x, y)


if __name__ == '__main__':
    main()
