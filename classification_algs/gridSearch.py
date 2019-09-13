from classification_algs.helpers import *
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

mlp_param_grid = [
    {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'hidden_layer_sizes': [
            (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,),
            (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,), (21,)
        ]
    }
]


def grid_search_mlp(x, y):
    clf = GridSearchCV(MLPClassifier(max_iter=1000), mlp_param_grid, cv=5, scoring='accuracy')
    clf.fit(x, y)
    print("Best parameters set found for MLPClassifier():")
    print(clf.best_params_)


def main():
    x, y = load_data('PhishingData.csv')
    scaled_x = scale_features(x)

    grid_search_mlp(scaled_x, y)


if __name__ == '__main__':
    main()
