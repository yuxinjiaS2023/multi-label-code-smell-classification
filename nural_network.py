from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_dnd(x,y,feature_selection=False):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    if feature_selection:
        model = RFE(estimator=clf, step=1)
    clf.fit(x_train, y_train)
    print("================================================")
    print("\nThe testing set results are: ")
    print("Deep Nural Network accuracy " + str(accuracy_score(y_test, model.predict(x_test))))
    print("Deep Nural Network f1_score " + str(f1_score(y_test, model.predict(x_test),average='weighted')))
    print("Deep Nural Network precision " + str(precision_score(y_test, model.predict(x_test),average='weighted')))
    print("Deep Nural Network recall " + str(recall_score(y_test, model.predict(x_test),average='weighted')))
