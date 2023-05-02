# multi-label-code-smell-classification
Currently has a basic preprocessing tool located in data_processor.py, check the simple_processer_example() in main for more details.
The data_processor uses median to fill NaN values and uses selectKBest to feature select.

# Before Running Python models
`conda install requirement.txt`
# How to RUN Python models?
We recommend using [JetBrains PyCharm ](https://www.jetbrains.com/pycharm/)  
In pycharm, run with green button on `main.py`. Input the model name you want to use, and press `enter`. Model would then be trained and print the results.