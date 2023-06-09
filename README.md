
# Before Running Python models
`conda install requirements.txt`

Or

`pip install requirements.txt`
# How to RUN Python models?
We recommend using [JetBrains PyCharm ](https://www.jetbrains.com/pycharm/)  
In pycharm, run with the green button on `main.py`. Input the model name you want to use, and then press `enter`. Model would then be trained and print the results.

# Running Java code
Java code is located in /J48-Classifier/src folder. Run the file J48Classifier.java, it should give you relevant result of a specific arff file. You can change the file name passed into function evaluateFile() inside the main function, to experiment on different arff files located in /data/exported/ folder.

# Version Log
May 1, 2023
- Refactory code structure, add simple interface to use.

Apr 25, 2023
- Add SVM model using both One-Vs-One and One-vs-All strategies.
- Add NB model.

Apr 24, 2023
- Fix arff and add Jaccard index in J48
- Add NN

Apr 23, 2023
- Added code that will run the process of training and outputting every results (BR, CC, LC) for both Decision Tree and Random Forest within one function

Apr 17, 2023
- Added cross validation to training code

Apr 15, 2023
- Add J48 Model implemented in Java

Apr 14, 202
- Add training code that works with Decision Tree and Random Forest (although this code should be able to train any model given a string name of the model)

Apr 7, 2023
- Currently has a basic preprocessing tool located in data_processor.py, check the simple_processer_example() in main for more details.
- The data_processor uses median to fill NaN values and uses selectKBest to feature select.
