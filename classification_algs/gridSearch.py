from classification_algs.helpers import *
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

dt_param_grid = [
    {
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22],
        'max_features': ['auto', 'sqrt', 'log2', None],
        #'max_leaf_nodes': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    }
]

knn_param_grid = [
    {
        'n_neighbors': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'p': [1, 2],
        'weights': ['uniform', 'distance']
    }
]

mlp_param_grid = [
    {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'hidden_layer_sizes': [
            (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,)
        ]
    }
]

boost_param_grid = [
    {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [.1, .25, .5, .75, 1., 1.25, 1.5]
    }
]

sgdc_param_grid = [
    {
        'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron',
                 'squared_loss'],
        'penalty': ['none', 'l2', 'l1', 'elasticnet'],
        'alpha': [0.00001, 0.0001, 0.001, 0.01, .1, 1., 2.],
        'eta0':[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1., 2.],
        'fit_intercept': [True, False],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
    }
]

svc_param_grid = [
    {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [3, 4, 5, 6, 7],
        'gamma': ['auto', 'scale'],
        'coef0': [-1., -.5, 0.0, .5, .75]
    }
]


def grid_search_svc(x, y):
    clf = GridSearchCV(SVC(), svc_param_grid, cv=5, scoring='accuracy')
    clf.fit(x, y)
    print("Best parameters set found for SVClassifier():")
    print(clf.best_params_)


def grid_search_sgdc(x, y):
    clf = GridSearchCV(SGDClassifier(), sgdc_param_grid, cv=5, scoring='accuracy')
    clf.fit(x, y)
    print("Best parameters set found for SGDClassifier():")
    print(clf.best_params_)


def grid_search_boost(x, y, **kwargs):
    clf = GridSearchCV(AdaBoostClassifier(base_estimator=
                                          DecisionTreeClassifier(**kwargs)),
                       boost_param_grid, cv=5, scoring='accuracy')
    clf.fit(x, y)
    print("Best parameters set found for AdaBoostClassifier():")
    print(clf.best_params_)


def grid_search_mlp(x, y):
    clf = GridSearchCV(MLPClassifier(), mlp_param_grid, cv=5, scoring='accuracy')
    clf.fit(x, y)
    print("Best parameters set found for MLPClassifier():")
    print(clf.best_params_)


def grid_search_knn(x, y):
    clf = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring='accuracy')
    clf.fit(x, y)
    print("Best parameters set found for KNeighborsClassifier():")
    print(clf.best_params_)


def grid_search_dt(x, y):
    clf = GridSearchCV(DecisionTreeClassifier(), dt_param_grid, cv=5, scoring='accuracy')
    clf.fit(x, y)
    print("Best parameters set found for DecisionTreeClassifier():")
    print(clf.best_params_)


def main():
    x, y = load_data('purchase_intent.csv')
    scaled_x = scale_features(x)

    #grid_search_dt(scaled_x, y)
    #grid_search_knn(scaled_x, y)
    #grid_search_mlp(scaled_x, y)
    #grid_search_boost(scaled_x, y, max_depth=4)
    grid_search_svc(scaled_x, y)

    #grid_search_sgdc(scaled_x, y)


if __name__ == '__main__':
    main()
