#!/usr/bin/env python
"""
    File name: model_predict_obj.py
    Author: skconan
    Date created: 2019/07/27
    Python Version: 2.7
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import *
from sklearn.externals import joblib

def main():
    csv_file = "/home/skconan/underwater_semantic_segmentation/data.csv"
    X_train, y_train, X_test, y_test = get_dataset(csv_file)
    forest = RandomForestClassifier(n_estimators=5000)
    forest = forest.fit(X_train, y_train)
    forest_output = forest.predict(X_test)

    print(accuracy_score(y_test, forest_output))

    filename = "./forest_model.sav"
    joblib.dump(forest, filename)


if __name__ == "__main__":
    main()
