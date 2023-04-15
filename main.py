# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import data_processor
import numpy as np
import copy
import Utilities
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Passes in two label arrays and returns the indices of the rows that are common to both arrays
def extract_common_rows(arr1, arr2):
    col_idx = 0
    # Extract the values in the first column of each array
    col1_arr1 = arr1[:, 0]
    col1_arr2 = arr2[:, 0]
    # Find the common values and their corresponding row indices
    common_values = np.intersect1d(col1_arr1, col1_arr2)
    common_rows_arr1 = np.where(np.isin(col1_arr1, common_values))[0]
    common_rows_arr2 = np.where(np.isin(col1_arr2, common_values))[0]
    return common_rows_arr1, common_rows_arr2


# Passes in target of both smells and combine them to a single multi-labelled target array
def combine_common_rows(y1, y2, common_rows_arr1, common_rows_arr2):
    #  Return the indices of the elements in arr1 that are also in arr2
    new_y1 = y1[common_rows_arr1]
    new_y2 = y2[common_rows_arr2]
    result = [int(f"{a}{b}", 2) for a, b in zip(new_y1, new_y2)]
    #  Return the elements in arr1 that are also in arr2
    return result


def common_instances_combine(dp1, dp2):
    new_dp = copy.deepcopy(dp1)
    # print(dp1.x.shape)
    common_rows_arr1, common_rows_arr2 = extract_common_rows(dp1.x, dp2.x)
    new_dp.y = np.array(combine_common_rows(dp1.y, dp2.y, common_rows_arr1, common_rows_arr2))
    new_dp.x = new_dp.x[common_rows_arr1]
    new_dp.value_columns = new_dp.value_columns[common_rows_arr1]
    new_dp.update_value_label_columns_index()
    return new_dp

def common_instances_chain(dp1, dp2):
    new_dp = copy.deepcopy(dp1)
    common_rows_arr1, common_rows_arr2 = extract_common_rows(dp1.x, dp2.x)
    new_dp.y = dp2.y[common_rows_arr2]
    #print("dp1", dp1.y[common_rows_arr1])
    new_dp.x = np.concatenate((new_dp.x[common_rows_arr1], dp1.y[common_rows_arr1].reshape(-1, 1)), axis=1)
    # new_dp.value_columns = new_dp.value_columns[common_rows_arr1]
    new_dp.value_columns = new_dp.value_columns[common_rows_arr1]
    new_dp.update_value_label_columns_index()
    return new_dp

# This picks up on
def uncommon_instances_combine(dp1, dp2, model_name):
    clf1 = Utilities.get_model(model_name)
    common_rows_arr1, common_rows_arr2 = extract_common_rows(dp1.x, dp2.x)
    # Create a boolean array indicating which rows to select
    select_x1 = np.zeros(dp1.value_columns.shape[0], dtype=bool)
    select_x1[common_rows_arr1] = True
    uncommon_values1 = dp1.value_columns[~select_x1]
    # print(uncommon_values1.shape)
    clf1.fit(dp1.value_columns, dp1.y)
    # the second dataset
    clf2 = Utilities.get_model(model_name)
    clf2.fit(dp2.value_columns, dp2.y)
    select_x2 = np.zeros(dp2.value_columns.shape[0], dtype=bool)
    select_x2[common_rows_arr2] = True
    uncommon_values2 = dp2.value_columns[~select_x2]
    predicted_uncommon_y1 = clf1.predict(uncommon_values2)
    uncommon_y2 = dp2.y[~select_x2]
    new_y2 = [int(f"{a}{b}", 2) for a, b in zip(predicted_uncommon_y1, uncommon_y2)]
    predicted_uncommon_y2 = clf2.predict(uncommon_values1)
    uncommon_y1 = dp1.y[~select_x1]
    new_y1 = [int(f"{a}{b}", 2) for a, b in zip(uncommon_y1, predicted_uncommon_y2)]
    return uncommon_values1, uncommon_values2, new_y1, new_y2

def label_combination(dp1, dp2, model_name):
    # Common instances:
    new_dp = common_instances_combine(dp1, dp2)
    # Uncommon instances:
    uncommon_values1, uncommon_values2, uncommon_y1, uncommon_y2 = uncommon_instances_combine(dp1, dp2, model_name)
    x = np.concatenate((new_dp.value_columns, uncommon_values1, uncommon_values2), axis=0)
    y = np.concatenate((new_dp.y, uncommon_y1, uncommon_y2), axis=0)
    return x, y

def label_chain(dp1, dp2):
    new_dp = common_instances_chain(dp1, dp2)
    return new_dp.value_columns, new_dp.y


def train(model_name,x,y,feature_selection=False):
    clf = Utilities.get_model(model_name)
    model = clf
    print(x)
    print(y)
    #   split the set into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15, random_state=42)
    if feature_selection:
        model = RFE(estimator=clf, step=1)
    model.fit(x_train, y_train)
    print("================================================")
    print("\nThe testing set results are: ")
    print("DT accuracy " + str(accuracy_score(y_test, model.predict(x_test))))
    print("DT f1_score " + str(f1_score(y_test, model.predict(x_test),average='weighted')))
    print("DT precision " + str(precision_score(y_test, model.predict(x_test),average='weighted')))
    print("DT recall " + str(recall_score(y_test, model.predict(x_test),average='weighted')))
    '''
    #   =====
    #   CROSS VALIDATION
    #   =====
    # want to use cross-validation to check different models using the same small data pool (grid-search)
    #
    depths = np.arange(10, 21)  # something between 10 and 21, exclusive on 21
    num_leafs = [1, 5, 10, 20, 50, 100]  # number of leaves
    #  we want to tune based on criteria, max_depth, and min_samples_leaf
    param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': depths, 'min_samples_leaf': num_leafs}
    # create another DT
    new_tree_clf = DecisionTreeClassifier()
    # do a grid search where you find all the possible combinations of the GS based on their grid search parameters
    grid_search = GridSearchCV(new_tree_clf, param_grid, cv=10, scoring="accuracy", return_train_score=True)
    grid_search.fit(x_train, y_train)
    # this is the best estimator for the grid
    best_model = grid_search.best_estimator_
    '''
    
    

def simple_processor_example(method):
    # give the appropriate file name for input data
    dp_gc = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=False)
    dp_dc = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=False)
    dp_lm = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=True)
    dp_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=True)
    # dp.x stores the processed and feature selected data
    # dp.value_columns vs dp.label_columns
    # dp.y stores the target
    if method == "Classifier Chain":
        method_mld_cc_x, method_mld_cc_y = label_chain(dp_lm, dp_fe)
        class_mld_cc_x, class_mld_cc_y = label_chain(dp_gc, dp_dc)
        print("x", method_mld_cc_x)
        print("y", method_mld_cc_y)
    elif method == "Label Combination":
        method_mld_lc_x, method_mld_lc_y = label_combination(dp_lm, dp_fe, "CART")
        class_mld_lc_x, class_mld_lc_y = label_combination(dp_gc, dp_dc, "CART")
        return method_mld_lc_x, method_mld_lc_y, class_mld_lc_x, class_mld_lc_y

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    simple_processor_example("Classifier Chain")
    method_mld_lc_x, method_mld_lc_y, class_mld_lc_x, class_mld_lc_y = simple_processor_example("Label Combination")
    train("RF", class_mld_lc_x, class_mld_lc_y, False)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
