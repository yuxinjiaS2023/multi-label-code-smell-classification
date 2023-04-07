# This is a sample Python script.
import self as self

from data.feature_selector import FeatureSelector
from data_loader import DataLoader


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

class Main:
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_selector = FeatureSelector()
    def start(self):
        X, y = self.data_loader.load_data("data/god-class.csv")
        self.feature_selector.select_feature(X, y)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main = Main()
    main.start()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
