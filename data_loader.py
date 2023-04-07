import csv

import pandas as pd
class DataLoader:
    def load_data(self, file_path):
        with open(file_path, newline='') as csvfile:
            df = pd.read_csv(file_path)
            num_cols = len(df.columns)
            X = df.iloc[:, 4:num_cols - 1].replace("?", "0").astype(float)
            y = df.iloc[:, num_cols - 1].astype(int)
            return X, y
