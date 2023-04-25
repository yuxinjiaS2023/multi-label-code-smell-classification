from sklearn.feature_selection import mutual_info_classif
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
# from weka.core.converters import Loader
# from weka.core.dataset import Instances
# from sklweka.classifiers import WekaEstimator
# from sklearn.ensemble import RandomForestClassifier
# from weka.classifiers import Classifier

SELECTKBEST = 10

CLASSLEVELLABEL = ["IDType", "project", "package", "complextype"]
METHODLEVELLABEL = ["IDMethod", "project", "package", "complextype", "method"]
CLASSLEVELLABEL_INDEX = [0,1,2,3]
METHODLEVELLABEL_INDEX = [0,1,2,3,4]


def get_model(model):
    global K_NEIGHBORS
    if model is None:
        print("no model specified")
        exit()
    if model == "B-J48":
        # This might not exist!
        return DecisionTreeClassifier(criterion='entropy')
    elif model == "BP":
        # Create a sequential model
        model = Sequential()
        # Add input layer
        model.add(Dense(units=4, input_dim=2, activation='relu'))
        # Add hidden layer
        model.add(Dense(units=4, activation='relu'))
        # Add output layer
        model.add(Dense(units=1, activation='sigmoid'))
        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    elif model == "J48":
        # Create a sequential model
        return DecisionTreeClassifier(criterion='entropy')
    elif model == "SVM":
        return make_pipeline(StandardScaler(), OneVsOneClassifier(SVC(kernel='linear')))
    elif model == "SVM_OVA":
        return make_pipeline(StandardScaler(), OneVsRestClassifier(SVC(kernel='linear')))
    elif model == "CART":
        return DecisionTreeClassifier(random_state=42)
    elif model == "RF":
        return RandomForestClassifier(random_state=42)
    elif model == "DT":
        return DecisionTreeClassifier(random_state=42)
    elif model == "NB":
        return GaussianNB()
    elif model == "NN":
        return MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500)
    else:
        print("such model does not exist")
        exit()

