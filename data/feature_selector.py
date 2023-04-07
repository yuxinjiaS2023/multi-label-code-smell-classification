from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split


class FeatureSelector:
    def select_feature(self, X, y):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        print("X_train", X_train.shape)
        #print("y_train", y_train)
        # Select the top k features using the ANOVA F-value method
        k = 10
        selector = SelectKBest(f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        print(X_train_selected.shape)
