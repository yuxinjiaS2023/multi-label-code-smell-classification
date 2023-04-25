# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import data_processor
import numpy as np
import copy
import arff
import Utilities
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score, cross_validate
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer, \
    hamming_loss, jaccard_score
from sklearn.neural_network import MLPClassifier


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
    # print("dp1", dp1.y[common_rows_arr1])
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

def hyperparameter_tuning(X, Y, clf, model_name):
    if (model_name == "DT"):
        param_grid = {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_depth": [None, 10, 50, 100, 200],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            
            
        }
    elif (model_name == "RF"):
        param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [3,5,7,10],
            "min_samples_leaf": [1, 5, 10, 20, 50, 100],
            "max_features": [ "sqrt", "log2"],
            "n_jobs": [-1]
            
        }
    X_train, X_test, y_train, y_test = train_test_split(X,

                                                        Y,

                                                        test_size=0.1,

                                                        random_state=42)
    clf.fit(X_train, y_train)
    grid_search = GridSearchCV(
        clf, param_grid, cv=10, scoring="accuracy", return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model

def train(model_name,x,y,ht=False, feature_selection=False):
    clf = Utilities.get_model(model_name)
    scoring_dict =  {'accuracy' : make_scorer(accuracy_score), 
       'precision' : make_scorer(precision_score, average = 'weighted'),
       'recall' : make_scorer(recall_score, average = 'weighted'), 
       'f1_score' : make_scorer(f1_score, average = 'weighted'),
       'hamming_loss': make_scorer(hamming_loss),
       'jaccard_score': make_scorer(jaccard_score, average = 'weighted'),
       }
    if(ht and (model_name == "DT" or model_name == "RF") ):
        clf =  hyperparameter_tuning(x,y,clf,model_name)
    #   I belive this will do feature selection in the Cross_validate?
    if feature_selection:
        clf = RFE(estimator=clf)
    #   clf.fit(x_train, y_train)
    k_folds = KFold(n_splits = 10, shuffle=True, random_state=42)
    scores = cross_validate(clf, x, y, cv = k_folds, scoring=scoring_dict)
    
    print("================================================")
    print("\nThe testing set results are: ")
    print("Accuracy " + str(scores["test_accuracy"].mean()))
    print("F1_score " + str(scores["test_f1_score"].mean()))
    print("Precision " + str(scores["test_precision"].mean()))
    print("Recall " + str(scores["test_recall"].mean()))
    print("Hamming Loss " + str(scores["test_hamming_loss"].mean()))
    print("Jaccard Score " + str(scores["test_jaccard_score"].mean()))
    print("================================================")

    '''
    do not need this ???
    '''
    #   split the set into training and testing
    # x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15, random_state=42)
    # if feature_selection:
    #    clf = RFE(estimator=clf, step=1)
    # clf.fit(x_train, y_train)

    '''
    Not exactly  sure about showing the results for each label because of 
    what is needed as parameters for classification_report or precision_recall_fscore_support.
    
    K_fold does not have x_text or y_test so not sure how to use Classification Report on it
    '''
    # print("Classification report")
    # pred = clf.predict(x_test)
    # print("Y_TEST:", y_test)
    # print("PRED: ", pred)
    # print(classification_report(y_test,pred))


def dump_arff_file(x, y, file_name):
    # Dump the dataset to an ARFF file
    y_int = y.astype(int)
    # print(x.shape[1])
    attributes = []
    for i in range(x.shape[1]):
        attributes.append(("attrs" + str(i), 'NUMERIC'))
    attributes.append(("class", 'NOMINAL'))
    print(attributes)
    y_str = []
    for i in range(y.shape[0]):
        if y_int[i] == 3:
            y_str.append("Y3")
        elif y_int[i] == 2:
            y_str.append("Y2")
        elif y_int[i] == 1:
            y_str.append("Y1")
        else:
            y_str.append("Y0")
    y_str = np.array(y_str)
    data = np.concatenate((x, y_str.reshape(-1, 1)), axis=1)
    data = data.tolist()
    for row in data:
        for i in range(len(row)):
            try:
                row[i] = float(row[i])
            except ValueError:
                pass
    arff.dump(file_name, data, relation='my_relation')


def simple_processor_example(method, dump=False):
    # give the appropriate file name for input data
    dp_gc_no_fe = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=False)
    dp_gc_fe = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=True)
    dp_dc_no_fe = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=False)
    dp_dc_fe = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=True)
    dp_lm_fe = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=True)
    dp_lm_no_fe = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=False)
    dp_fe_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=True)
    dp_fe_no_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=False)
    if dump:
        dump_arff_file(dp_gc_no_fe.value_columns, dp_gc_no_fe.y, "god_class_no_fe.arff")
        dump_arff_file(dp_gc_fe.value_columns, dp_gc_fe.y, "god_class_fe.arff")
        dump_arff_file(dp_dc_no_fe.value_columns, dp_dc_no_fe.y, "data_class_no_fe.arff")
        dump_arff_file(dp_dc_fe.value_columns, dp_dc_fe.y, "data_class_fe.arff")
        dump_arff_file(dp_lm_no_fe.value_columns, dp_lm_no_fe.y, "long_method_no_fe.arff")
        dump_arff_file(dp_lm_fe.value_columns, dp_lm_fe.y, "long_method_fe.arff")
        dump_arff_file(dp_fe_no_fe.value_columns, dp_fe_no_fe.y, "feature_envy_no_fe.arff")
        dump_arff_file(dp_fe_fe.value_columns, dp_fe_fe.y, "feature_envy_fe.arff")

    # dp.x stores the processed and feature selected data
    # dp.value_columns vs dp.label_columns
    # dp.y stores the target
    if method == "Classifier Chain":
        method_mld_cc_fe_x, method_mld_cc_fe_y = label_chain(dp_lm_fe, dp_fe_fe)
        method_mld_cc_no_fe_x, method_mld_cc_no_fe_y = label_chain(dp_lm_no_fe, dp_fe_no_fe)
        class_mld_cc_no_fe_x, class_mld_cc_no_fe_y = label_chain(dp_gc_no_fe, dp_dc_no_fe)
        class_mld_cc_fe_x, class_mld_cc_fe_y = label_chain(dp_gc_fe, dp_dc_fe)
        if dump:
            dump_arff_file(method_mld_cc_fe_x, method_mld_cc_fe_y, "method_mld_fe_cc.arff")
            dump_arff_file(class_mld_cc_no_fe_x, class_mld_cc_no_fe_y, "class_mld__no_fe_cc.arff")
            dump_arff_file(class_mld_cc_fe_x, class_mld_cc_fe_y, "class_mld_fe_cc.arff")
            dump_arff_file(method_mld_cc_no_fe_x, method_mld_cc_no_fe_y, "method_mld_no_fe_cc.arff")
    elif method == "Label Combination":
        method_mld_lc_fe_x, method_mld_lc_fe_y = label_combination(dp_lm_fe, dp_fe_fe, "CART")
        method_mld_lc_no_fe_x, method_mld_lc_no_fe_y = label_combination(dp_lm_no_fe, dp_fe_no_fe, "CART")
        class_mld_lc_no_fe_x, class_mld_lc_no_fe_y = label_combination(dp_gc_no_fe, dp_dc_no_fe, "CART")
        class_mld_lc_fe_x, class_mld_lc_fe_y = label_combination(dp_gc_fe, dp_dc_fe, "CART")
        if dump:
            dump_arff_file(method_mld_lc_fe_x, method_mld_lc_fe_y, "method_mld_fe_lc.arff")
            dump_arff_file(method_mld_lc_no_fe_x, method_mld_lc_no_fe_y, "method_mld_no_fe_lc.arff")
            dump_arff_file(class_mld_lc_no_fe_x, class_mld_lc_no_fe_y, "class_mld_no_fe_lc.arff")
            dump_arff_file(class_mld_lc_fe_x, class_mld_lc_fe_y, "class_mld_fe_lc.arff")
        return method_mld_lc_fe_x, method_mld_lc_fe_y, class_mld_lc_fe_x, class_mld_lc_fe_y


def dt_rf_runner():
    #   regular
    dp_gc_no_fe = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=False)
    dp_gc_fe = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=True)
    dp_dc_no_fe = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=False)
    dp_dc_fe = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=True)
    dp_lm_fe = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=True)
    dp_lm_no_fe = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=False)
    dp_fe_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=True)
    dp_fe_no_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=False)
    #   CC
    method_mld_cc_fe_x, method_mld_cc_fe_y = label_chain(dp_lm_fe, dp_fe_fe)
    method_mld_cc_no_fe_x, method_mld_cc_no_fe_y = label_chain(dp_lm_no_fe, dp_fe_no_fe)
    class_mld_cc_no_fe_x, class_mld_cc_no_fe_y = label_chain(dp_gc_no_fe, dp_dc_no_fe)
    class_mld_cc_fe_x, class_mld_cc_fe_y = label_chain(dp_gc_fe, dp_dc_fe)
    #   LC
    method_mld_lc_fe_x, method_mld_lc_fe_y = label_combination(dp_lm_fe, dp_fe_fe, "CART")
    method_mld_lc_no_fe_x, method_mld_lc_no_fe_y = label_combination(dp_lm_no_fe, dp_fe_no_fe, "CART")
    class_mld_lc_no_fe_x, class_mld_lc_no_fe_y = label_combination(dp_gc_no_fe, dp_dc_no_fe, "CART")
    class_mld_lc_fe_x, class_mld_lc_fe_y = label_combination(dp_gc_fe, dp_dc_fe, "CART")

    #   RUN THE DT 
    '''
    print("=============== STARTING DT for BASE ===============")
    train("DT", dp_gc_no_fe.value_columns, dp_gc_no_fe.y)
    train("DT", dp_dc_no_fe.value_columns, dp_gc_no_fe.y)
    train("DT", dp_lm_no_fe.value_columns, dp_gc_no_fe.y)
    train("DT", dp_fe_no_fe.value_columns, dp_gc_no_fe.y)
    print("=============== ENDING DT for BASE ===============")
    
    
    print("=============== STARTING RF for BASE ===============")
    train("RF", dp_gc_no_fe.value_columns, dp_gc_no_fe.y)
    train("RF", dp_dc_no_fe.value_columns, dp_gc_no_fe.y)
    train("RF", dp_lm_no_fe.value_columns, dp_gc_no_fe.y)
    train("RF", dp_fe_no_fe.value_columns, dp_gc_no_fe.y)
    print("=============== ENDING RF for BASE ===============")
    #   print(dp_gc_no_fe.label_columns, dp_gc_no_fe.y)

    print("=============== STARTING DT FOR COMBINED===============")
    print("DT 1")
    train("DT", method_mld_cc_no_fe_x, method_mld_cc_no_fe_y)
    print("DT 2")
    train("DT", method_mld_lc_no_fe_x, method_mld_lc_no_fe_y)
    print("DT 3")
    train("DT", class_mld_cc_no_fe_x, class_mld_cc_no_fe_y)
    print("DT 4")
    train("DT", class_mld_lc_no_fe_x, class_mld_lc_no_fe_y)
    print("DT 5")
    train("DT", class_mld_lc_fe_x, class_mld_lc_fe_y, True)
    print("=============== ENDING DT FOR COMBINED===============")

    print("=============== STARTING RF FOR COMBINED===============")
    print("RF 1")
    train("RF", method_mld_cc_no_fe_x, method_mld_cc_no_fe_y)
    print("RF 2")
    train("RF", method_mld_lc_no_fe_x, method_mld_lc_no_fe_y)
    print("RF 3")
    train("RF", class_mld_cc_no_fe_x, class_mld_cc_no_fe_y)
    print("RF 4")
    train("RF", class_mld_lc_no_fe_x, class_mld_lc_no_fe_y)
    print("RF 5")
    '''
    train("RF", class_mld_lc_fe_x, class_mld_lc_fe_y, True)
    print("=============== ENDING RF FOR COMBINED ===============")

def svm():
    dp_gc_no_fe = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=False)
    dp_gc_fe = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=True)
    dp_dc_no_fe = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=False)
    dp_dc_fe = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=True)
    dp_lm_fe = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=True)
    dp_lm_no_fe = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=False)
    dp_fe_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=True)
    dp_fe_no_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=False)

    method_mld_cc_fe_x, method_mld_cc_fe_y = label_chain(dp_lm_fe, dp_fe_fe)
    method_mld_cc_no_fe_x, method_mld_cc_no_fe_y = label_chain(dp_lm_no_fe, dp_fe_no_fe)
    class_mld_cc_no_fe_x, class_mld_cc_no_fe_y = label_chain(dp_gc_no_fe, dp_dc_no_fe)
    class_mld_cc_fe_x, class_mld_cc_fe_y = label_chain(dp_gc_fe, dp_dc_fe)
    #   LC
    method_mld_lc_fe_x, method_mld_lc_fe_y = label_combination(dp_lm_fe, dp_fe_fe, "CART")
    method_mld_lc_no_fe_x, method_mld_lc_no_fe_y = label_combination(dp_lm_no_fe, dp_fe_no_fe, "CART")
    class_mld_lc_no_fe_x, class_mld_lc_no_fe_y = label_combination(dp_gc_no_fe, dp_dc_no_fe, "CART")
    class_mld_lc_fe_x, class_mld_lc_fe_y = label_combination(dp_gc_fe, dp_dc_fe, "CART")

    print("=============== STARTING SVM for BASE ===============")
    print("SVM god class no FE")
    train("SVM", dp_gc_no_fe.value_columns, dp_gc_no_fe.y)
    print("SVM data class no FE")
    train("SVM", dp_dc_no_fe.value_columns, dp_dc_no_fe.y)
    print("SVM long method no FE")
    train("SVM", dp_lm_no_fe.value_columns, dp_lm_no_fe.y)
    print("SVM feature envy no FE")
    train("SVM", dp_fe_no_fe.value_columns, dp_fe_no_fe.y)
    print("=============== ENDING SVM for BASE ===============")

    print("=============== STARTING SVM FOR COMBINED===============")
    print("SVM method CC no FE")
    train("SVM", method_mld_cc_no_fe_x, method_mld_cc_no_fe_y)
    print("SVM method LC no FE")
    train("SVM", method_mld_lc_no_fe_x, method_mld_lc_no_fe_y)
    print("SVM class CC no FE")
    train("SVM", class_mld_cc_no_fe_x, class_mld_cc_no_fe_y)
    print("SVM class LC no FE")
    train("SVM", class_mld_lc_no_fe_x, class_mld_lc_no_fe_y)
    print("SVM class LC with FE")
    train("SVM", class_mld_lc_fe_x, class_mld_lc_fe_y)
    print("=============== ENDING SVM FOR COMBINED===============")
def svm_ova():
    dp_gc_no_fe = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=False)
    dp_gc_fe = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=True)
    dp_dc_no_fe = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=False)
    dp_dc_fe = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=True)
    dp_lm_fe = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=True)
    dp_lm_no_fe = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=False)
    dp_fe_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=True)
    dp_fe_no_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=False)

    method_mld_cc_fe_x, method_mld_cc_fe_y = label_chain(dp_lm_fe, dp_fe_fe)
    method_mld_cc_no_fe_x, method_mld_cc_no_fe_y = label_chain(dp_lm_no_fe, dp_fe_no_fe)
    class_mld_cc_no_fe_x, class_mld_cc_no_fe_y = label_chain(dp_gc_no_fe, dp_dc_no_fe)
    class_mld_cc_fe_x, class_mld_cc_fe_y = label_chain(dp_gc_fe, dp_dc_fe)
    #   LC
    method_mld_lc_fe_x, method_mld_lc_fe_y = label_combination(dp_lm_fe, dp_fe_fe, "CART")
    method_mld_lc_no_fe_x, method_mld_lc_no_fe_y = label_combination(dp_lm_no_fe, dp_fe_no_fe, "CART")
    class_mld_lc_no_fe_x, class_mld_lc_no_fe_y = label_combination(dp_gc_no_fe, dp_dc_no_fe, "CART")
    class_mld_lc_fe_x, class_mld_lc_fe_y = label_combination(dp_gc_fe, dp_dc_fe, "CART")

    print("=============== STARTING SVM_OVA for BASE ===============")
    print("SVM god class no FE")
    train("SVM_OVA", dp_gc_no_fe.value_columns, dp_gc_no_fe.y)
    print("SVM data class no FE")
    train("SVM_OVA", dp_dc_no_fe.value_columns, dp_dc_no_fe.y)
    print("SVM long method no FE")
    train("SVM_OVA", dp_lm_no_fe.value_columns, dp_lm_no_fe.y)
    print("SVM feature envy no FE")
    train("SVM_OVA", dp_fe_no_fe.value_columns, dp_fe_no_fe.y)
    print("=============== ENDING SVM_OVA for BASE ===============")

    print("=============== STARTING SVM_OVA FOR COMBINED===============")
    print("SVM method CC no FE")
    train("SVM_OVA", method_mld_cc_no_fe_x, method_mld_cc_no_fe_y)
    print("SVM method LC no FE")
    train("SVM_OVA", method_mld_lc_no_fe_x, method_mld_lc_no_fe_y)
    print("SVM class CC no FE")
    train("SVM_OVA", class_mld_cc_no_fe_x, class_mld_cc_no_fe_y)
    print("SVM class LC no FE")
    train("SVM_OVA", class_mld_lc_no_fe_x, class_mld_lc_no_fe_y)
    print("SVM class LC with FE")
    train("SVM_OVA", class_mld_lc_fe_x, class_mld_lc_fe_y)
    print("=============== ENDING SVM_OVA FOR COMBINED===============")

def NB():
    dp_gc_no_fe = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=False)
    dp_gc_fe = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=True)
    dp_dc_no_fe = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=False)
    dp_dc_fe = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=True)
    dp_lm_fe = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=True)
    dp_lm_no_fe = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=False)
    dp_fe_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=True)
    dp_fe_no_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=False)

    method_mld_cc_fe_x, method_mld_cc_fe_y = label_chain(dp_lm_fe, dp_fe_fe)
    method_mld_cc_no_fe_x, method_mld_cc_no_fe_y = label_chain(dp_lm_no_fe, dp_fe_no_fe)
    class_mld_cc_no_fe_x, class_mld_cc_no_fe_y = label_chain(dp_gc_no_fe, dp_dc_no_fe)
    class_mld_cc_fe_x, class_mld_cc_fe_y = label_chain(dp_gc_fe, dp_dc_fe)
    #   LC
    method_mld_lc_fe_x, method_mld_lc_fe_y = label_combination(dp_lm_fe, dp_fe_fe, "CART")
    method_mld_lc_no_fe_x, method_mld_lc_no_fe_y = label_combination(dp_lm_no_fe, dp_fe_no_fe, "CART")
    class_mld_lc_no_fe_x, class_mld_lc_no_fe_y = label_combination(dp_gc_no_fe, dp_dc_no_fe, "CART")
    class_mld_lc_fe_x, class_mld_lc_fe_y = label_combination(dp_gc_fe, dp_dc_fe, "CART")

    print("=============== STARTING NB for BASE ===============")
    print("NB god class no FE")
    train("NB", dp_gc_no_fe.value_columns, dp_gc_no_fe.y)
    print("NB data class no FE")
    train("NB", dp_dc_no_fe.value_columns, dp_dc_no_fe.y)
    print("NB long method no FE")
    train("NB", dp_lm_no_fe.value_columns, dp_lm_no_fe.y)
    print("NB feature envy no FE")
    train("NB", dp_fe_no_fe.value_columns, dp_fe_no_fe.y)
    print("=============== ENDING NB for BASE ===============")

    print("=============== STARTING NB FOR COMBINED===============")
    print("NB method CC no FE")
    train("NB", method_mld_cc_no_fe_x, method_mld_cc_no_fe_y)
    print("NB method LC no FE")
    train("NB", method_mld_lc_no_fe_x, method_mld_lc_no_fe_y)
    print("NB class CC no FE")
    train("NB", class_mld_cc_no_fe_x, class_mld_cc_no_fe_y)
    print("NB class LC no FE")
    train("NB", class_mld_lc_no_fe_x, class_mld_lc_no_fe_y)
    print("NB class LC with FE")
    train("NB", class_mld_lc_fe_x, class_mld_lc_fe_y)
    print("=============== ENDING NB FOR COMBINED===============")
def nn_runner():
    #   regular
    dp_gc_no_fe = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=False)
    dp_gc_fe = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=True)
    dp_dc_no_fe = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=False)
    dp_dc_fe = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=True)
    dp_lm_fe = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=True)
    dp_lm_no_fe = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=False)
    dp_fe_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=True)
    dp_fe_no_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=False)
    #   CC
    method_mld_cc_fe_x, method_mld_cc_fe_y = label_chain(dp_lm_fe, dp_fe_fe)
    method_mld_cc_no_fe_x, method_mld_cc_no_fe_y = label_chain(dp_lm_no_fe, dp_fe_no_fe)
    class_mld_cc_no_fe_x, class_mld_cc_no_fe_y = label_chain(dp_gc_no_fe, dp_dc_no_fe)
    class_mld_cc_fe_x, class_mld_cc_fe_y = label_chain(dp_gc_fe, dp_dc_fe)
    #   LC
    method_mld_lc_fe_x, method_mld_lc_fe_y = label_combination(dp_lm_fe, dp_fe_fe, "CART")
    method_mld_lc_no_fe_x, method_mld_lc_no_fe_y = label_combination(dp_lm_no_fe, dp_fe_no_fe, "CART")
    class_mld_lc_no_fe_x, class_mld_lc_no_fe_y = label_combination(dp_gc_no_fe, dp_dc_no_fe, "CART")
    class_mld_lc_fe_x, class_mld_lc_fe_y = label_combination(dp_gc_fe, dp_dc_fe, "CART")

    #   RUN THE DT
    print("=============== STARTING DT for BASE ===============")
    train("NN", dp_gc_no_fe.value_columns, dp_gc_no_fe.y)
    train("NN", dp_dc_no_fe.value_columns, dp_gc_no_fe.y)
    train("NN", dp_lm_no_fe.value_columns, dp_gc_no_fe.y)
    train("NN", dp_fe_no_fe.value_columns, dp_gc_no_fe.y)
    print("=============== ENDING DT for BASE ===============")

    print("=============== STARTING NN FOR COMBINED===============")
    print("NN 1")
    train("NN", method_mld_cc_no_fe_x, method_mld_cc_no_fe_y)
    print("NN 2")
    train("NN", method_mld_lc_no_fe_x, method_mld_lc_no_fe_y)
    print("NN 3")
    train("NN", class_mld_cc_no_fe_x, class_mld_cc_no_fe_y)
    print("NN 4")
    train("NN", class_mld_lc_no_fe_x, class_mld_lc_no_fe_y)
    print("NN 5")
    train("NN", class_mld_lc_fe_x, class_mld_lc_fe_y, True)
    print("=============== ENDING DT FOR COMBINED===============")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    #   simple_processor_example("Classifier Chain", dump=True)
    # method_mld_lc_x, method_mld_lc_y, class_mld_lc_x, class_mld_lc_y = simple_processor_example("Label Combination", dump=True)
    # print(class_mld_lc_y)
    # train("RF", class_mld_lc_x, class_mld_lc_y, False)
    # nn_runner()
    # svm()
    NB()
    # svm_ova()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
