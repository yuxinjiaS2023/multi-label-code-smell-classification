# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import data_processor
import numpy as np
import copy
import Utilities


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
    # print(new_dp.y.shape)
    # print(new_dp.x.shape)
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
    # print(uncommon_values2.shape)
    return uncommon_values1, uncommon_values2, new_y1, new_y2


def label_combination(dp1, dp2, model_name):
    # Common instances:
    new_dp = common_instances_combine(dp1, dp2)
    # Uncommon instances:
    uncommon_values1, uncommon_values2, uncommon_y1, uncommon_y2 = uncommon_instances_combine(dp1, dp2, model_name)
    # print(new_dp.y)
    # print(new_dp.value_columns.shape)
    x = np.concatenate((new_dp.value_columns, uncommon_values1, uncommon_values2), axis=0)
    y = np.concatenate((new_dp.y, uncommon_y1, uncommon_y2), axis=0)
    return x, y


def simple_processor_example():
    # give the appropriate file name for input data
    dp_gc = data_processor.DataProcessor("god-class.csv", class_level=True)
    dp_dc = data_processor.DataProcessor("data-class.csv", class_level=True)
    dp_lm = data_processor.DataProcessor("long-method.csv", class_level=False)
    dp_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False)
    # dp.x stores the processed and feature selected data
    # dp.value_columns vs dp.label_columns
    # dp.y stores the target
    method_mld_lc_x, method_mld_lc_y = label_combination(dp_lm, dp_fe, "SVM")
    class_mld_lc_x, class_mld_lc_y = label_combination(dp_gc, dp_dc, "SVM")
    print(class_mld_lc_y)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    simple_processor_example()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
