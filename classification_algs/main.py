import sklearn.model_selection as ms

from classification_algs.helpers import *

from sklearn.tree import DecisionTreeClassifier
from classification_algs.DTAnalysis import DTAnalysis
from sklearn.neighbors import KNeighborsClassifier
from classification_algs.KNNAnalysis import KNNAnalysis
from sklearn.neural_network import MLPClassifier
from classification_algs.MLPAnalysis import MLPAnalysis
from sklearn.svm import SVC
from classification_algs.SVCAnalysis import SVCAnalysis
from sklearn.ensemble import AdaBoostClassifier
from classification_algs.BoostAnalysis import BoostAnalysis
from sklearn.feature_extraction import DictVectorizer


def main():

    #x, y_unraveled = load_data('PhishingData.csv')
    #y = np.ravel(y_unraveled)
    #x_train, x_test, y_train, y_test = \
    #    ms.train_test_split(x, np.ravel(y), test_size=0.80, random_state=0)

    #x, y = load_data('PhishingData.csv')
    x, y = load_trunc_data('purchase_intent.csv', 10000)
    #x_train, x_test, y_train, y_test = \
    #    ms.train_test_split(x, y, test_size=0.80, random_state=0)

    scaled_x = scale_features(x)

    # INITIAL TESTING
    #DTAnalysis(x_train, x_test, y_train, y_test)
    #KNNAnalysis(x_train, x_test, y_train, y_test)
    #MLPAnalysis(x_train, x_test, y_train, y_test)
    #BoostAnalysis(x_train, x_test, y_train, y_test)
    #SVCAnalysis(x_train, x_test, y_train, y_test)

    """
    *** Phishing Data History ***
    
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
    
    plot_learning_curve(KNeighborsClassifier(n_neighbors=2),
                        "Phishing Data KNN: k = 2", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(n_neighbors=5),
                        "Phishing Data KNN: k = 5", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(n_neighbors=8),
                        "Phishing Data KNN: k = 8", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(weights='distance'),
                        "Phishing Data KNN: weights='distance'", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(weights='uniform'),
                        "Phishing Data KNN: weights='uniform'", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(p=1),
                        "Phishing Data KNN: p=1 Manhattan Distance", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(p=2),
                        "Phishing Data KNN: p=2 Euclidean Distance", scaled_x, y)
    
    plot_learning_curve(MLPClassifier(activation='relu', hidden_layer_sizes=10),
                        "Phishing Data NN (unscaled)", x, y)
    plot_learning_curve(MLPClassifier(activation='relu', hidden_layer_sizes=10),
                        "Phishing Data NN", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='relu', hidden_layer_sizes=10,
                                      solver='lbfgs'),
                        "Phishing Data NN: solver='lbfgs'", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='relu', hidden_layer_sizes=10,
                                      solver='sgd'),
                        "Phishing Data NN: solver='sgd'", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='relu', hidden_layer_sizes=10,
                                      solver='adam'),
                        "Phishing Data NN: solver='adam'", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='relu', hidden_layer_sizes=10,
                                      learning_rate='constant'),
                        "Phishing Data NN: learning_rate='constant'", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='relu', hidden_layer_sizes=10,
                                      learning_rate='invscaling'),
                        "Phishing Data NN: learning_rate='invscaling'", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='relu', hidden_layer_sizes=10,
                                      learning_rate='adaptive'),
                        "Phishing Data NN: learning_rate='adaptive'", scaled_x, y)
    
    plot_learning_curve(AdaBoostClassifier(), "Phishing Data Boosting (unscaled)", x, y)
    plot_learning_curve(AdaBoostClassifier(), "Phishing Data Boosting", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=2)),
                        "Phishing Data: Boosting DT Depth 2", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=5)),
                        "Phishing Data: Boosting DT Depth 5", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=8)),
                        "Phishing Data: Boosting DT Depth 8", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=5),
                                           learning_rate=.25),
                        "Phishing Data: Boosting, learning_rate=0.25", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=5),
                                           learning_rate=1.),
                        "Phishing Data: Boosting, learning_rate=1.0", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=5),
                                           learning_rate=2.),
                        "Phishing Data: Boosting, learning_rate=2.0", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=5),
                                           n_estimators=25),
                        "Phishing Data: Boosting, n_estimators=25", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=5),
                                           n_estimators=150),
                        "Phishing Data: Boosting, n_estimators=150", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=5),
                                           n_estimators=300),
                        "Phishing Data: Boosting, n_estimators=300", scaled_x, y)
    
    plot_learning_curve(SVC(), "Phishing Data: SVC (unscaled)", x, y)
    plot_learning_curve(SVC(), "Phishing Data: SVC", scaled_x, y)
    plot_learning_curve(SVC(kernel='poly'), 
                        "Phishing Data: SVC kernel='poly'", scaled_x, y)
    plot_learning_curve(SVC(kernel='rbf'),
                        "Phishing Data: SVC kernel='rbf'", scaled_x, y)
    plot_learning_curve(SVC(gamma='auto'),
                        "Phishing Data: SVC gamma='auto'", scaled_x, y)
    plot_learning_curve(SVC(gamma='scale'),
                        "Phishing Data: SVC gamma='scale'", scaled_x, y)
    plot_learning_curve(SVC(kernel='poly', degree=2),
                        "Phishing Data: SVC kernel='poly', deg 2", scaled_x, y)
    plot_learning_curve(SVC(kernel='poly', degree=3),
                        "Phishing Data: SVC kernel='poly', deg 3", scaled_x, y)
    plot_learning_curve(SVC(kernel='poly', degree=4),
                        "Phishing Data: SVC kernel='poly', deg 4", scaled_x, y)
    plot_learning_curve(SVC(kernel='poly', coef0=0.0),
                        "Phishing Data: SVC kernel='poly', coef0=0.0", scaled_x, y)
    plot_learning_curve(SVC(kernel='poly', coef0=1.0),
                        "Phishing Data: SVC kernel='poly', coef0=1.0", scaled_x, y)
    plot_learning_curve(SVC(kernel='poly', coef0=2.0),
                        "Phishing Data: SVC kernel='poly', coef0=2.0", scaled_x, y)
    plot_learning_curve(SVC(kernel='poly', coef0=3.0),
                        "Phishing Data: SVC kernel='poly', coef0=3.0", scaled_x, y)
    """


    """
    TEST Phishing Data
    
    plot_learning_curve(MLPClassifier(activation='relu', hidden_layer_sizes=10,
                                      solver='lbfgs'),
                        "Phishing Data NN: Test'", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='logistic', hidden_layer_sizes=12,
                                      solver='lbfgs'),
                        "Phishing Data NN: Test 2'", scaled_x, y)
                        
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=5),
                                           learning_rate=1., n_estimators=300),
                        "Phishing Data: Boosting TEST", scaled_x, y)
    
    plot_learning_curve(SVC(kernel='poly', gamma='auto', coef0=0.0, degree=3),
                        "Phishing Data: SVC TEST 1", scaled_x, y)
    plot_learning_curve(SVC(kernel='rbf', gamma='auto', coef0=0.0),
                        "Phishing Data: SVC TEST 2", scaled_x, y)
    
    """

    """
        *** Purchase Intent Data History ***
    """
    plot_learning_curve(DecisionTreeClassifier(), "Purchase Intent Data DT", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_depth=15),
                        "Purchase Intent Data DT: Max Depth 15", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_depth=19),
                        "Purchase Intent Data DT: Max Depth 19", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_depth=23),
                        "Purchase Intent Data DT: Max Depth 23", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_features='sqrt'),
                        "Purchase Intent Data DT: (sqrt n) Max Features", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_features='log2'),
                        "Purchase Intent Data DT: (log2 n) Max Features", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_leaf_nodes=2),
                        "Purchase Intent Data DT: Max Leaf Nodes = 2", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_leaf_nodes=5),
                        "Purchase Intent Data DT: Max Leaf Nodes = 5", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_leaf_nodes=8),
                        "Purchase Intent Data DT: Max Leaf Nodes = 8", x, y)

    plot_learning_curve(KNeighborsClassifier(n_neighbors=4),
                        "Purchase Intent Data KNN: k = 2", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(n_neighbors=6),
                        "Purchase Intent Data KNN: k = 5", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(n_neighbors=8),
                        "Purchase Intent Data KNN: k = 8", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(weights='distance'),
                        "Purchase Intent Data KNN: weights='distance'", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(weights='uniform'),
                        "Purchase Intent Data KNN: weights='uniform'", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(p=1),
                        "Purchase Intent Data KNN: p=1 Manhattan Distance", scaled_x, y)
    plot_learning_curve(KNeighborsClassifier(p=2),
                        "Purchase Intent Data KNN: p=2 Euclidean Distance", scaled_x, y)

    plot_learning_curve(MLPClassifier(activation='tanh', hidden_layer_sizes=12),
                        "Purchase Intent Data NN", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='tanh', hidden_layer_sizes=12,
                                      solver='lbfgs'),
                        "Purchase Intent Data NN: solver='lbfgs'", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='tanh', hidden_layer_sizes=12,
                                      solver='sgd'),
                        "Purchase Intent Data NN: solver='sgd'", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='tanh', hidden_layer_sizes=12,
                                      solver='adam'),
                        "Purchase Intent Data NN: solver='adam'", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='tanh', hidden_layer_sizes=12,
                                      learning_rate='constant'),
                        "Purchase Intent Data NN: learning_rate='constant'", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='tanh', hidden_layer_sizes=12,
                                      learning_rate='invscaling'),
                        "Purchase Intent Data NN: learning_rate='invscaling'", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='tanh', hidden_layer_sizes=12,
                                      learning_rate='adaptive'),
                        "Purchase Intent Data NN: learning_rate='adaptive'", scaled_x, y)

    plot_learning_curve(AdaBoostClassifier(), "Purchase Intent Data Boosting", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=15)),
                        "Purchase Intent Data: Boosting DT Depth 15", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=19)),
                        "Purchase Intent Data: Boosting DT Depth 19", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=23)),
                        "Purchase Intent Data: Boosting DT Depth 23", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=19),
                                           learning_rate=-1.),
                        "Purchase Intent Data: Boosting, learning_rate=-1.0", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=19),
                                           learning_rate=-.1),
                        "Purchase Intent Data: Boosting, learning_rate=-0.1", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=19),
                                           learning_rate=1.),
                        "Purchase Intent Data: Boosting, learning_rate=1.0", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=19),
                                           n_estimators=100),
                        "Purchase Intent Data: Boosting, n_estimators=100", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=19),
                                           n_estimators=300),
                        "Purchase Intent Data: Boosting, n_estimators=300", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=19),
                                           n_estimators=500),
                        "Purchase Intent Data: Boosting, n_estimators=500", scaled_x, y)

    plot_learning_curve(SVC(), "Purchase Intent Data: SVC", scaled_x, y)
    plot_learning_curve(SVC(kernel='poly'),
                        "Purchase Intent Data: SVC kernel='poly'", scaled_x, y)
    plot_learning_curve(SVC(kernel='rbf'),
                        "Purchase Intent Data: SVC kernel='rbf'", scaled_x, y)
    plot_learning_curve(SVC(kernel='poly', degree=3),
                        "Purchase Intent Data: SVC kernel='poly', deg 3", scaled_x, y)
    plot_learning_curve(SVC(kernel='poly', degree=5),
                        "Purchase Intent Data: SVC kernel='poly', deg 5", scaled_x, y)
    plot_learning_curve(SVC(kernel='poly', degree=7),
                        "Purchase Intent Data: SVC kernel='poly', deg 7", scaled_x, y)
    plot_learning_curve(SVC(kernel='rbf', coef0=-1.0),
                        "Purchase Intent Data: SVC kernel='poly', coef0=-1.0", scaled_x, y)
    plot_learning_curve(SVC(kernel='rbf', coef0=0.0),
                        "Purchase Intent Data: SVC kernel='poly', coef0=0.0", scaled_x, y)
    plot_learning_curve(SVC(kernel='rbf', coef0=1.0),
                        "Purchase Intent Data: SVC kernel='poly', coef0=1.0", scaled_x, y)


    """
    TEST Purchase Intent
    
    plot_learning_curve(DecisionTreeClassifier(max_depth=4, max_features=None),
                        "Purchase Intent Data DT TEST 1", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_depth=18, max_features=None,
                                               max_leaf_nodes=9),
                        "Purchase Intent Data DT TEST 2", x, y)
    plot_learning_curve(DecisionTreeClassifier(max_depth=7, max_features=None,
                                               max_leaf_nodes=23),
                        "Purchase Intent Data DT TEST 3", x, y)
    
    plot_learning_curve(KNeighborsClassifier(n_neighbors=18, p=2, weights='uniform'),
                        "Purchase Intent Data KNN TEST 1", scaled_x, y)
    plot_learning_curve(MLPClassifier(activation='tanh', hidden_layer_sizes=12, solver='adam'),
                        "Purchase Intent Data NN TEST 1", scaled_x, y)
    
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=4)),
                        "Purchase Intent Data: Boosting DT Depth 4", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=7)),
                        "Purchase Intent Data: Boosting DT Depth 7", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=19)),
                        "Purchase Intent Data: Boosting DT Depth 19", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(base_estimator=
                                           DecisionTreeClassifier(max_depth=29)),
                        "Purchase Intent Data: Boosting DT Depth 29", scaled_x, y)
    
    plot_learning_curve(AdaBoostClassifier(learning_rate=1., n_estimators=300),
                        "Purchase Intent Data Boost TEST 1", scaled_x, y)
    plot_learning_curve(AdaBoostClassifier(learning_rate=.1, n_estimators=100),
                        "Purchase Intent Data Boost TEST 2", scaled_x, y)
    
    plot_learning_curve(SVC(kernel='rbf', gamma='auto'),
                        "Purchase Intent Data SVC TEST 1", scaled_x, y)
    """




if __name__ == '__main__':
    main()
