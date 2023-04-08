# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import data_processor


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def simple_processer_example():
    # give the appropriate file name for input data
    new_dp = data_processor.DataProcessor("long-method.csv", class_level=False)
    # dp.x stores the processed and feature selected data
    # dp.value_columns vs dp.label_columns
    # dp.y stores the target
    print(new_dp.label_columns)
    print(new_dp.value_columns)
    print(new_dp.x) # X= label_columns + value_columns
    print(new_dp.y)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    simple_processer_example()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
