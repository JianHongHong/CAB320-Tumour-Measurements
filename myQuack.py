
'''
Scaffolding code for the Machine Learning assignment.
You should complete the provided functions and add more functions and classes as necessary.

You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.
You are welcome to use the pandas library if you know it.
'''
import numpy as np
import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
    return [ (9790136, 'JianHong', 'Lee'),
             (9863320, 'Myeonghan', 'Ryu'),
             (9757449, 'PinYun', 'Wang') ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def prepare_dataset(dataset_path):
    '''
    Read a comma separated text file where
	- the first field is a ID number
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued
    Return two numpy arrays X and y where
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'
    @param dataset_path: full path of the dataset text file
    @return
	X,y
    '''
    dataset = np.genfromtxt(dataset_path, delimiter=',', dtype=None)
    X = [list(row)[2:] for row in dataset] # 2d for X
    y = [1 if row[1] == b'M' else 0 for row in dataset] # 1d for y
    return np.array(X), np.array(y) # output array for X and y


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def build_DecisionTree_classifier(X_training, y_training):
    '''
    Build a Decision Tree classifier based on the training set X_training, y_training.
    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    @return
	clf : the classifier built in this function
    '''
    para = {'max_depth':list(range(1,100,1))} # Parameter list
    clf = DecisionTreeClassifier() # Initiliase clf
    tune_clf = GridSearchCV(clf, para, scoring='accuracy',cv=5) # Cross-validation
    tune_clf.fit(X_training, y_training)
    return tune_clf



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    '''
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.
    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    @return
	clf : the classifier built in this function
    '''

    para = {'n_neighbors':list(range(1,100,1))} # Parameter list
    clf = KNeighborsClassifier()
    tune_clf = GridSearchCV(clf, para, scoring='accuracy',cv=5) # Cross-validation
    tune_clf.fit(X_training,y_training)
    return tune_clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''
    Build a Support Vector Machine classifier based on the training set X_training, y_training.
    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    @return
	clf : the classifier built in this function
    '''
    # clf= LinearSVC(C=1.0)
    # clf.fit(X_training, y_training)

    para = {'C':list(range(1,100,1))}
    clf = SVC(kernel='linear')
    tune_clf = GridSearchCV(clf, para, scoring='accuracy',cv=5)
    tune_clf.fit(X_training,y_training)
    return tune_clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    '''
    Build a Neural Network with two dense hidden layers classifier
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library
    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    @return
	clf : the classifier built in this function
    '''
    para = {'hidden_layer_sizes': (100,),} # Parameter list
    iterations=1000   # define the iterations for training over the dataset
    clf = MLPClassifier(max_iter=iterations) # hidden layer will default to 100
    tune_clf = GridSearchCV(clf, para, scoring='accuracy',cv=5) # Cross-validation
    tune_clf.fit(X_training,y_training)
    return tune_clf


def get_validation_accuracy(classfier):
    '''
    Gets the validation accuracy of the classfier
    '''
    mean_list = classfier.cv_results_['mean_test_score']
    validation_accuracy = mean_list.mean()
    return validation_accuracy

def get_training_accuracy(classifier):
    '''
    Gets the training accuracy of the classfier
    '''
    training_accuracy = classifier.best_score_
    return training_accuracy

def get_testing_accuracy(classfier,X_test,y_test):
    '''
    Get testing accuracy of the classifer
    '''
    y_pred = classfier.predict(X_test)
    testing_accuracy = accuracy_score(y_test, y_pred)
    return testing_accuracy

def draw_graph():
    return NotImplemented



def create_table(DecisionTree,NearrestNeighbours,SupportVectorMachine,NeuralNetwork,X_input,y_input,accuracy_type):
    '''
    Table to list out the accuracy of each classifers
    '''
    if (accuracy_type == 'test'):
        # DecisionTree
        accuracy_DecisionTree = get_testing_accuracy(DecisionTree,X_input,y_input)

        # NearrestNeighbours
        accuracy_NearestNeighbours = get_testing_accuracy(NearrestNeighbours,X_input,y_input)

        # SupportVectorMachine
        accuracy_SVM = get_testing_accuracy(SupportVectorMachine,X_input,y_input)

        # NeuralNetwork
        accuracy_NeuralNetwork = get_testing_accuracy(NeuralNetwork,X_input,y_input)

    if (accuracy_type == 'train'):
        # DecisionTree
        accuracy_DecisionTree = get_training_accuracy(DecisionTree)

        # NearrestNeighbours
        accuracy_NearestNeighbours = get_training_accuracy(NearrestNeighbours)

        # SupportVectorMachine
        accuracy_SVM = get_training_accuracy(SupportVectorMachine)

        # NeuralNetwork
        accuracy_NeuralNetwork = get_training_accuracy(NeuralNetwork)
    
    if (accuracy_type == 'validation'):
        # DecisionTree
        accuracy_DecisionTree = get_validation_accuracy(DecisionTree)

        # NearrestNeighbours
        accuracy_NearestNeighbours = get_validation_accuracy(NearrestNeighbours)

        # SupportVectorMachine
        accuracy_SVM = get_validation_accuracy(SupportVectorMachine)

        # NeuralNetwork
        accuracy_NeuralNetwork = get_validation_accuracy(NeuralNetwork)

    data = {'Classifer':['DecisionTree', 'NearrestNeighbours','SupportVectorMachine','NeuralNetwork'],
            'Accuracy':[accuracy_DecisionTree, accuracy_NearestNeighbours, accuracy_SVM, accuracy_NeuralNetwork]}
    df = pd.DataFrame(data)
    print(df[['Classifer', 'Accuracy']].to_string(index=False))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":

    # Team
    print(my_team())

    # Input file to prepare dataset
    X,y = prepare_dataset('medical_records.data')

    # Splitting into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=50)

    # Scaling of data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Building classifiers (add int the classifers here)
    DecisionTree = build_DecisionTree_classifier(X_train, y_train)
    NearrestNeighbour = build_NearrestNeighbours_classifier(X_train, y_train)
    NeuralNetwork = build_NeuralNetwork_classifier(X_train,y_train)
    SupportVectorMachine = build_SupportVectorMachine_classifier(X_train, y_train)

    # Printout results
    print("------------------------------------")
    print("Prediction Accuracy for Training set")
    print("------------------------------------")
    create_table(DecisionTree,NearrestNeighbour,NeuralNetwork, SupportVectorMachine, X_test,y_test,accuracy_type='train')
    print()

    print("--------------------------------------")
    print("Prediction Accuracy for Validation set")
    print("--------------------------------------")
    create_table(DecisionTree,NearrestNeighbour,NeuralNetwork,SupportVectorMachine, X_test,y_test,accuracy_type='validation')
    print()

    print("-----------------------------------")
    print("Prediction Accuracy for Testing set")
    print("-----------------------------------")
    create_table(DecisionTree,NearrestNeighbour,NeuralNetwork, SupportVectorMachine, X_test,y_test,accuracy_type='test')
    print()


########################################################################################
