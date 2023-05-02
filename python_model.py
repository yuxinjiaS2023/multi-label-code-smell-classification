import data_processor
import copy
import numpy as np
import Utilities
import arff
import Utilities
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score, cross_validate
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer, \
    hamming_loss, jaccard_score

class Py_Model:
    def __init__(self):
        self.SPACER = "="*50
        self.name = 'Py_model'
        self.version = '1.0.0'
        
    def hyperparameter_tuning(self, X, Y, clf, model_name):
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
                "max_depth": [3, 5, 7, 10],
                "min_samples_leaf": [1, 5, 10, 20, 50, 100],
                "max_features": ["sqrt", "log2"],
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
    
    def train(self, model_name, x, y, ht=False, feature_selection=False):
        clf = Utilities.get_model(model_name)
        scoring_dict = {'accuracy': make_scorer(accuracy_score),
                        'precision': make_scorer(precision_score, average='weighted'),
                        'recall': make_scorer(recall_score, average='weighted'),
                        'f1_score': make_scorer(f1_score, average='weighted'),
                        'hamming_loss': make_scorer(hamming_loss),
                        'jaccard_score': make_scorer(jaccard_score, average='weighted'),
                        }
        if (ht and (model_name == "DT" or model_name == "RF")):
            clf = self.hyperparameter_tuning(x, y, clf, model_name)
        #   I belive this will do feature selection in the Cross_validate?
        if feature_selection:
            clf = RFE(estimator=clf)
        #   clf.fit(x_train, y_train)
        k_folds = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_validate(clf, x, y, cv=k_folds, scoring=scoring_dict)

        print(self.SPACER)
        print("The testing set results are: ")
        print("Accuracy " + str(scores["test_accuracy"].mean()))
        print("F1_score " + str(scores["test_f1_score"].mean()))
        print("Precision " + str(scores["test_precision"].mean()))
        print("Recall " + str(scores["test_recall"].mean()))
        print("Hamming Loss " + str(scores["test_hamming_loss"].mean()))
        print("Jaccard Score " + str(scores["test_jaccard_score"].mean()))
        print(self.SPACER+"\n")

    def combine_common_rows(self,y1, y2, common_rows_arr1, common_rows_arr2):
        #  Return the indices of the elements in arr1 that are also in arr2
        new_y1 = y1[common_rows_arr1]
        new_y2 = y2[common_rows_arr2]
        result = [int(f"{a}{b}", 2) for a, b in zip(new_y1, new_y2)]
        #  Return the elements in arr1 that are also in arr2
        return result

    def extract_common_rows(self, arr1, arr2):
        col_idx = 0
        # Extract the values in the first column of each array
        col1_arr1 = arr1[:, 0]
        col1_arr2 = arr2[:, 0]
        # Find the common values and their corresponding row indices
        common_values = np.intersect1d(col1_arr1, col1_arr2)
        common_rows_arr1 = np.where(np.isin(col1_arr1, common_values))[0]
        common_rows_arr2 = np.where(np.isin(col1_arr2, common_values))[0]
        return common_rows_arr1, common_rows_arr2
    def common_instances_combine(self,dp1, dp2):
        new_dp = copy.deepcopy(dp1)
        # print(dp1.x.shape)
        common_rows_arr1, common_rows_arr2 = self.extract_common_rows(dp1.x, dp2.x)
        new_dp.y = np.array(self.combine_common_rows(dp1.y, dp2.y, common_rows_arr1, common_rows_arr2))
        new_dp.x = new_dp.x[common_rows_arr1]
        new_dp.value_columns = new_dp.value_columns[common_rows_arr1]
        new_dp.update_value_label_columns_index()
        return new_dp
    
    # This picks up on
    def uncommon_instances_combine(this,dp1, dp2, model_name):
        clf1 = Utilities.get_model(model_name)
        common_rows_arr1, common_rows_arr2 = this.extract_common_rows(dp1.x, dp2.x)
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
    
   
    def common_instances_chain(self,dp1, dp2):
        new_dp = copy.deepcopy(dp1)
        common_rows_arr1, common_rows_arr2 = self.extract_common_rows(dp1.x, dp2.x)
        new_dp.y = dp2.y[common_rows_arr2]
        # print("dp1", dp1.y[common_rows_arr1])
        new_dp.x = np.concatenate((new_dp.x[common_rows_arr1], dp1.y[common_rows_arr1].reshape(-1, 1)), axis=1)
        # new_dp.value_columns = new_dp.value_columns[common_rows_arr1]
        new_dp.value_columns = new_dp.value_columns[common_rows_arr1]
        new_dp.update_value_label_columns_index()
        return new_dp
    
    def label_combination(self,dp1, dp2, model_name):
        # Common instances:
        new_dp = self.common_instances_combine(dp1, dp2)
        # Uncommon instances:
        uncommon_values1, uncommon_values2, uncommon_y1, uncommon_y2 = self.uncommon_instances_combine(dp1, dp2, model_name)
        x = np.concatenate((new_dp.value_columns, uncommon_values1, uncommon_values2), axis=0)
        y = np.concatenate((new_dp.y, uncommon_y1, uncommon_y2), axis=0)
        return x, y   
    
    def label_chain(self,dp1, dp2):
        new_dp = self.common_instances_chain(dp1, dp2)
        return new_dp.value_columns, new_dp.y
    def run_helper_no_fe(self, model_type, dp_no_fe_list):
        dp_dic = [
            "god class no FE",
            "data class no FE",
            "long method no FE",
            "feature envy no FE"
        ]
        print("=============== STARTING " + model_type + " for BASE ===============")
        i=0
        for dp in dp_no_fe_list:
            print(model_type + " " + dp_dic[i])
            self.train(model_type, dp.value_columns, dp.y)
            i+=1
        print("=============== ENDING " + model_type + " for BASE ===============")

    def run_helper_combine(self,model_type, data_list):
        print(f"=============== STARTING {model_type} FOR COMBINED===============")
        for description, x_data, y_data in data_list:
            print(f"{model_type} {description}")
            self.train(model_type, x_data, y_data)
        print(f"=============== ENDING {model_type} FOR COMBINED===============")
        
    def create_data_list(self):
        dp_gc_no_fe = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=False)
        dp_gc_fe = data_processor.DataProcessor("god-class.csv", class_level=True, feature_selection=True)
        dp_dc_no_fe = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=False)
        dp_dc_fe = data_processor.DataProcessor("data-class.csv", class_level=True, feature_selection=True)
        dp_lm_fe = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=True)
        dp_lm_no_fe = data_processor.DataProcessor("long-method.csv", class_level=False, feature_selection=False)
        dp_fe_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=True)
        dp_fe_no_fe = data_processor.DataProcessor("feature-envy.csv", class_level=False, feature_selection=False)

        #   CC
        method_mld_cc_no_fe_x, method_mld_cc_no_fe_y = self.label_chain(dp_lm_no_fe, dp_fe_no_fe)
        class_mld_cc_no_fe_x, class_mld_cc_no_fe_y = self.label_chain(dp_gc_no_fe, dp_dc_no_fe)
        method_mld_cc_fe_x, method_mld_cc_fe_y = self.label_chain(dp_lm_fe, dp_fe_fe)
        class_mld_cc_fe_x, class_mld_cc_fe_y = self.label_chain(dp_gc_fe, dp_dc_fe)

        #   LC
        method_mld_lc_no_fe_x, method_mld_lc_no_fe_y = self.label_combination(dp_lm_no_fe, dp_fe_no_fe, "CART")
        class_mld_lc_no_fe_x, class_mld_lc_no_fe_y = self.label_combination(dp_gc_no_fe, dp_dc_no_fe, "CART")
        method_mld_lc_fe_x, method_mld_lc_fe_y = self.label_combination(dp_lm_fe, dp_fe_fe, "CART")
        class_mld_lc_fe_x, class_mld_lc_fe_y = self.label_combination(dp_gc_fe, dp_dc_fe, "CART")

        data_list = [
            ("method CC no FE", method_mld_cc_no_fe_x, method_mld_cc_no_fe_y),
            ("method LC no FE", method_mld_lc_no_fe_x, method_mld_lc_no_fe_y),
            ("class CC no FE", class_mld_cc_no_fe_x, class_mld_cc_no_fe_y),
            ("class LC no FE", class_mld_lc_no_fe_x, class_mld_lc_no_fe_y),
            ("class LC with FE", class_mld_lc_fe_x, class_mld_lc_fe_y),
        ]

        dp_no_fe_list=[dp_gc_no_fe,dp_dc_no_fe,dp_lm_no_fe,dp_fe_no_fe]
        return dp_no_fe_list, data_list

    def run_model(self,model_type):
        if model_type not in ["DT", "RF","SVM", "SVM_OVA", "NB","NN"]:
            print("Invalid model type. Choose from: 'DT_RF', 'SVM', 'SVM_OVA', 'NB'.")
            return
        dp_no_fe_list, data_list = self.create_data_list()

        self.run_helper_no_fe(model_type, dp_no_fe_list)
        self.run_helper_combine(model_type, data_list)