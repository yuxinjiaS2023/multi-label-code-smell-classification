# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import data_processor


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def simple_processer_example():
    # give the appropriate file name for input data
    new_dp = data_processor.DataProcessor("data-class.csv", True)
    print(new_dp.x)
    print(new_dp.y)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    simple_processer_example()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
