from scipy.io import arff
import pandas as pd
import csv
from sklearn import preprocessing
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
import Utilities


class DataProcessor:
    def data_load(self, file_name):
        data = pd.read_csv(file_name, header=0, na_values='?', quotechar="'", skipinitialspace=True)
        self.df = pd.DataFrame(data)
        # x = self.process_data(df)
        x = self.df.iloc[:, :-1]
        y_data = self.df.iloc[:, -1].values
        encoder = preprocessing.LabelEncoder()
        y = encoder.fit_transform(y_data)
        return x, y

    # Fix the NAN data with median strategy
    def process_data(self):
        self.value_columns[:] = SimpleImputer(strategy="median").fit_transform(self.value_columns)
        # new_X_df = pd.DataFrame(new_X, columns=X_copy.columns, index=X_copy.index)
        return self.value_columns.values

    def update_x(self):
        self.x = pd.concat([pd.DataFrame(self.label_columns), pd.DataFrame(self.value_columns)], axis=1).values

    def feature_selection_kBest(self):
        model = SelectKBest(k=Utilities.SELECTKBEST)
        self.value_columns = model.fit_transform(self.value_columns, self.y)
        self.attribute_names = self.attribute_names[np.array(model.get_support(indices=True))]
        # print(model.get_support(indices=True))

    def update_value_label_columns_index(self):
        if self.class_level:
            self.label_columns = self.x[Utilities.CLASSLEVELLABEL_INDEX]
            self.value_columns = np.delete(self.x, Utilities.CLASSLEVELLABEL_INDEX, axis=1)
        else:
            self.label_columns = self.x[Utilities.METHODLEVELLABEL_INDEX]
            self.value_columns = np.delete(self.x, Utilities.METHODLEVELLABEL_INDEX, axis=1)

    # If class_level is true, then the data is class level data, otherwise it is method level data
    def __init__(self, file_name, class_level, feature_selection):
        self.feature_selection = feature_selection
        self.x, self.y = self.data_load("data/" + file_name)
        # print(self.x)
        self.class_level = class_level
        # The label columns are the columns that are not numeric features, thus cannot be used for training. So we need
        # to separate them, the value columns are the numeric features that we can use for training.
        # self.x stores the concatenation of label columns and value columns, it cannot be directly used in training or
        # feature selection.
        if class_level:
            self.label_columns = self.x[Utilities.CLASSLEVELLABEL]
            self.value_columns = self.x.drop(columns=Utilities.CLASSLEVELLABEL)
        else:
            self.label_columns = self.x[Utilities.METHODLEVELLABEL]
            self.value_columns = self.x.drop(columns=Utilities.METHODLEVELLABEL)
        self.attribute_names = np.array(self.value_columns.columns.tolist())
        self.value_columns = self.process_data()
        if self.feature_selection:
            self.feature_selection_kBest()
        # print(self.value_columns)
        # Update x with the processed and feature selected value columns
        self.update_x()

