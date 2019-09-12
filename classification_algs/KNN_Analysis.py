import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class KNN_Analysis(object):
    """
    This class is for printing a decision tree classification analysis of a data set.
    """

    def __init__(self, x_train, x_test, y_train, y_test, **kwargs):
        scaled_x_train = self.scale_features(x_train)
        scaled_x_test = self.scale_features(x_test)

        knn = KNeighborsClassifier(**kwargs)
        knn.fit(scaled_x_train, y_train)
        predictions = knn.predict(scaled_x_test)
        report = classification_report(y_test, predictions)
        report_dict = classification_report(y_test, predictions, output_dict=True)
        
        print("KNN Analysis:\n")
        print(report)
        print("Test Set Accuracy: " + str(report_dict['accuracy']))

    @staticmethod
    def scale_features(input_df):
        scaler = StandardScaler()
        scaler.fit(input_df)
        scaled_features = scaler.transform(input_df)
        return pd.DataFrame(scaled_features)



