import mlrose

import sklearn.model_selection as ms
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from rand_opt.fastmimic import mlrose as fastmimic
from rand_opt.helpers import *


def main():

    # Load Data Set
    x_train = pd.read_csv('x_train.csv')
    x_test = pd.read_csv('x_test.csv')
    y_train = pd.read_csv('y_train.csv')
    y_test = pd.read_csv('y_test.csv')

    # Initialize neural network object and fit object
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu',
                                     algorithm='random_hill_climb', max_iters=1000,
                                     bias=True, is_classifier=True, learning_rate=0.01,
                                     early_stopping=True, clip_max=50, max_attempts=200,
                                     random_state=0)

    nn_model1.fit(x_train, y_train)

    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)

    print('Training accuracy: ', y_train_accuracy)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(x_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)

    print('Test accuracy: ', y_test_accuracy)


if __name__ == '__main__':
    main()
