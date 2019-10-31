from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from classification_algs.helpers import scale_features


class MLPAnalysis(object):
    """
    This class is for printing a MLP/neural network analysis of a data set.
    """

    def __init__(self, x_train, x_test, y_train, y_test, **kwargs):
        mlp = MLPClassifier(**kwargs)
        mlp.fit(x_train, y_train)
        predictions = mlp.predict(x_test)
        report = classification_report(y_test, predictions)
        report_dict = classification_report(y_test, predictions, output_dict=True)
        self.accuracy = str(report_dict['accuracy'])

        print("MLP Analysis:\n")
        print(report)
        print("Test Set Accuracy: " + self.accuracy)
