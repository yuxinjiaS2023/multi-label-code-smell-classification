from python_model import Py_Model 
from file_processor import File_Processor

def runner(model_name):
    py_model=Py_Model()
    py_model.run_model(model_name)

if __name__ == '__main__':
    while(1):
        print("Please enter the model name you want to run (SVM, SVM_OVA ,RF, DT, CART, NB, NN):")
        model_name = input()
        if model_name not in ['SVM', 'SVM_OVA' ,'RF', 'DT', 'CART', 'NB', 'NN']:
            print("Please enter the correct model name!")
            continue
        runner(model_name)
        exit_flag = input("Do you want to exit? (y/n)")
        if exit_flag == 'y':
            break
        else:
            continue
