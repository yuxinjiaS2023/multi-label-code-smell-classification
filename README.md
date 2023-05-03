
# Before Running Python models
`conda install requirements.txt`

Or

`pip install requirements.txt`
# How to RUN Python models?
We recommend using [JetBrains PyCharm ](https://www.jetbrains.com/pycharm/)  
In pycharm, run with green button on `main.py`. Input the model name you want to use, and press `enter`. Model would then be trained and print the results.

# Version Log
May 1, 2023
- Refactory code structure, add simple interface to use.

Apr 25, 2023
- Add SVM and NB

Apr 24, 2023
- Fix arff and add Jaccard index in J48
- Add NN

Apr 23, 2023
- Add DT and RF

Apr 15, 2023
- Add J48 Model implemented in Java

Apr 7, 2023
Currently has a basic preprocessing tool located in data_processor.py, check the simple_processer_example() in main for more details.
The data_processor uses median to fill NaN values and uses selectKBest to feature select.
