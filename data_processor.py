from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
import Utilities


class DataProcessor:
    def data_load(self, file_name):
        data = pd.read_csv(file_name, header=0, na_values='?')
        self.df = pd.DataFrame(data)
        # x = self.process_data(df)
        x = self.df.iloc[:, :-1]
        y_data = self.df.iloc[:, -1].values
        encoder = preprocessing.LabelEncoder()
        y = encoder.fit_transform(y_data)
        return x, y

    # Fix the NAN data with median strategy
    def process_data(self):
        X_copy = self.value_columns.copy()
        imputer = SimpleImputer(strategy="median")
        new_X = imputer.fit_transform(X_copy)
        # new_X_df = pd.DataFrame(new_X, columns=X_copy.columns, index=X_copy.index)
        return new_X

    def feature_selection_kBest(self):
        self.value_columns = SelectKBest(k=Utilities.SELECTKBEST).fit_transform(self.value_columns, self.y)

    def __init__(self, file_name):
        self.x, self.y = self.data_load("data/" + file_name)
        # The label columns are the columns that are not numeric features, thus cannot be used for training. So we need
        # to separate them, the value columns are the numeric features that we can use for training.
        # self.x stores the concatenation of label columns and value columns, it cannot be directly used in training or
        # feature selection.
        self.label_columns = self.x[["IDType", "project", "package", "complextype"]]
        self.value_columns = self.x.drop(columns=["IDType", "project", "package", "complextype"])
        self.value_columns = self.process_data()
        self.feature_selection_kBest()
        # Update x with the processed and feature selected value columns
        self.x = pd.concat([pd.DataFrame(self.label_columns), pd.DataFrame(self.value_columns)], axis=1).values

