import arff
import numpy as np
import data_processor
from python_model import Py_Model

class File_Processor:

    def __init__(self) -> None:
        pass

    def dump_arff_file(self, x, y, file_name):
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


    def simple_processor_example(self,method, dump=False):
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
            self.dump_arff_file(dp_gc_no_fe.value_columns, dp_gc_no_fe.y, "god_class_no_fe.arff")
            self.dump_arff_file(dp_gc_fe.value_columns, dp_gc_fe.y, "god_class_fe.arff")
            self.dump_arff_file(dp_dc_no_fe.value_columns, dp_dc_no_fe.y, "data_class_no_fe.arff")
            self.dump_arff_file(dp_dc_fe.value_columns, dp_dc_fe.y, "data_class_fe.arff")
            self.dump_arff_file(dp_lm_no_fe.value_columns, dp_lm_no_fe.y, "long_method_no_fe.arff")
            self.dump_arff_file(dp_lm_fe.value_columns, dp_lm_fe.y, "long_method_fe.arff")
            self.dump_arff_file(dp_fe_no_fe.value_columns, dp_fe_no_fe.y, "feature_envy_no_fe.arff")
            self.dump_arff_file(dp_fe_fe.value_columns, dp_fe_fe.y, "feature_envy_fe.arff")

        # dp.x stores the processed and feature selected data
        # dp.value_columns vs dp.label_columns
        # dp.y stores the target
        if method == "Classifier Chain":
            method_mld_cc_fe_x, method_mld_cc_fe_y = Py_Model.label_chain(dp_lm_fe, dp_fe_fe)
            method_mld_cc_no_fe_x, method_mld_cc_no_fe_y = Py_Model.label_chain(dp_lm_no_fe, dp_fe_no_fe)
            class_mld_cc_no_fe_x, class_mld_cc_no_fe_y = Py_Model.label_chain(dp_gc_no_fe, dp_dc_no_fe)
            class_mld_cc_fe_x, class_mld_cc_fe_y = Py_Model.label_chain(dp_gc_fe, dp_dc_fe)
            if dump:
                self.dump_arff_file(method_mld_cc_fe_x, method_mld_cc_fe_y, "method_mld_fe_cc.arff")
                self.dump_arff_file(class_mld_cc_no_fe_x, class_mld_cc_no_fe_y, "class_mld__no_fe_cc.arff")
                self.self.dump_arff_file(class_mld_cc_fe_x, class_mld_cc_fe_y, "class_mld_fe_cc.arff")
                self.dump_arff_file(method_mld_cc_no_fe_x, method_mld_cc_no_fe_y, "method_mld_no_fe_cc.arff")
        elif method == "Label Combination":
            method_mld_lc_fe_x, method_mld_lc_fe_y = Py_Model.label_combination(dp_lm_fe, dp_fe_fe, "CART")
            method_mld_lc_no_fe_x, method_mld_lc_no_fe_y = Py_Model.label_combination(dp_lm_no_fe, dp_fe_no_fe, "CART")
            class_mld_lc_no_fe_x, class_mld_lc_no_fe_y = Py_Model.label_combination(dp_gc_no_fe, dp_dc_no_fe, "CART")
            class_mld_lc_fe_x, class_mld_lc_fe_y = Py_Model.label_combination(dp_gc_fe, dp_dc_fe, "CART")
            if dump:
                self.dump_arff_file(method_mld_lc_fe_x, method_mld_lc_fe_y, "method_mld_fe_lc.arff")
                self.dump_arff_file(method_mld_lc_no_fe_x, method_mld_lc_no_fe_y, "method_mld_no_fe_lc.arff")
                self.dump_arff_file(class_mld_lc_no_fe_x, class_mld_lc_no_fe_y, "class_mld_no_fe_lc.arff")
                self.dump_arff_file(class_mld_lc_fe_x, class_mld_lc_fe_y, "class_mld_fe_lc.arff")
            return method_mld_lc_fe_x, method_mld_lc_fe_y, class_mld_lc_fe_x, class_mld_lc_fe_y