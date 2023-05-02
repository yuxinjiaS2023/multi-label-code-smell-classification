from python_model import Py_Model 
from file_processor import File_Processor

def runner(model_name):
    py_model=Py_Model()
    py_model.run_model(model_name)

if __name__ == '__main__':
    print("Please enter the model name you want to run (SVM, SVM_OVA ,RF, DT, CART, NB, NN):")
    model_name = input()
    runner(model_name)
