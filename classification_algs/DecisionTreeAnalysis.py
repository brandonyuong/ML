import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn.model_selection as ms
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


class DecisionTreeAnalysis(object):
    """
    This class is for printing a decision tree classification analysis of a data set.
    """

    def __init__(self, x_train, x_test, y_train, y_test, **kwargs):
        dtree = DecisionTreeClassifier(**kwargs)
        dtree.fit(x_train, y_train)
        predictions = dtree.predict(x_test)
        report = classification_report(y_test, predictions)
        report_dict = classification_report(y_test, predictions, output_dict=True)

        print("Decision Tree Analysis:\n")
        print(report)
        print("Test Set Accuracy: " + str(report_dict['accuracy']))
