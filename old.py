
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
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt


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

# JianHong
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
# JianHong
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
    final_clf = DecisionTreeClassifier(max_depth=tune_clf.best_params_['max_depth']) # Building clf using best parameters
    final_clf.fit(X_training,y_training) # Train model using best parameter added
    print(tune_clf.best_params_)
    means = tune_clf.cv_results_['mean_test_score']
    stds = tune_clf.cv_results_['std_test_score']
    print(list(range(1,100,1)))
    print(means)
    plt.plot(list(range(1,100,1)),means, 'b-')
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Test Score')
                          # plt.axis([0, 6, 0, 20])
    plt.show()
    for mean, std, params in zip(means, stds, tune_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print()

    return final_clf



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

    list_ = list(range(100))

    para = {'n_neighbors':list_[1::2]} # Parameter list
    clf = KNeighborsClassifier()
    tune_clf = GridSearchCV(clf, para, scoring='accuracy',cv=5) # Cross-validation
    tune_clf.fit(X_training,y_training)
    print(tune_clf.best_params_)
    means = tune_clf.cv_results_['mean_test_score']
    stds = tune_clf.cv_results_['std_test_score']
    plt.plot(list_[1::2],means, 'b-')
    plt.xlabel('Number of Neighbours')
    plt.ylabel('Mean Test Score')
                              # plt.axis([0, 6, 0, 20])
    plt.show()
    for mean, std, params in zip(means, stds, tune_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                 % (mean, std * 2, params))
        print()
    exit()
    final_clf = KNeighborsClassifier(n_neighbors=tune_clf.best_params_) # Building clf using best parameters
    final_clf.fit(X_training,y_training) # Train model using best parameter added


    return final_clf

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
    clf= LinearSVC(C=1.0)
    clf.fit(X_training, y_training)

    return clf

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
    para = {'hidden_layer_sizes':list(range(1,300,1))} # Parameter list
    clf = MLPClassifier()
    tune_clf = GridSearchCV(clf, para, scoring='accuracy',cv=5) # Cross-validation
    tune_clf.fit(X_training,y_training)
    print(tune_clf.best_params_)
    means = tune_clf.cv_results_['mean_test_score']
    stds = tune_clf.cv_results_['std_test_score']
    plt.plot(list(range(1,300,1)),means, 'b-')
    plt.xlabel('Hidden Layer Sizes')
    plt.ylabel('Mean Test Score')
                              # plt.axis([0, 6, 0, 20])
    plt.show()
    for mean, std, params in zip(means, stds, tune_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                    % (mean, std * 2, params))
        print()
    # mlp.fit(X_training,y_training)
    final_clf = MLPClassifier(hidden_layer_sizes=tune_clf.best_params_['hidden_layer_sizes']) # Building clf using best parameters
    final_clf.fit(X_training,y_training) # Train model using best parameter added

    return final_clf




def create_table(DecisionTree,NearrestNeighbours,NeuralNetwork, SupportVectorMachine, X_input,y_input,):
    '''
    Table to list out the accuracy of each classifers
    I've place the other classfier as None at the moment until classifers are completed
    '''
    # DecisionTree
    y_pred_DecisionTree = DecisionTree.predict(X_input)
    accuracy_DecisionTree = accuracy_score(y_input, y_pred_DecisionTree)

    # NearrestNeighbours
    y_pred_NearrestNeighbours = NearrestNeighbour.predict(X_input)
    accuracy_NearestNeighbours = accuracy_score(y_input, y_pred_NearrestNeighbours)

    # SupportVectorMachine
    y_pred_SVM = SupportVectorMachine.predict(X_input)
    accuracy_SVM = accuracy_score(y_input, y_pred_SVM)

    # NeuralNetwork
    y_pred_NeuralNetwork = NeuralNetwork.predict(X_input)
    accuracy_NeuralNetwork = accuracy_score(y_input, y_pred_NeuralNetwork)

    data = {'Classifer':['DecisionTree', 'NearrestNeighbours','NeuralNetwork', 'SupportVectorMachine'],
            'Accuracy':[accuracy_DecisionTree, accuracy_NearestNeighbours, accuracy_NeuralNetwork, accuracy_SVM]}
    df = pd.DataFrame(data)
    print(df[['Classifer', 'Accuracy']].to_string(index=False))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":

    # Team
    print(my_team())
    # plt.plot([1,2,3,4], [1,4,9,16], 'ro')
    # plt.axis([0, 6, 0, 20])
    # plt.show()
    # exit()
    # Input file to prepare dataset
    X,y = prepare_dataset('medical_records.data')

    # Transform X into 0 to 1 to be normalised to make processing of data faster
    X_scaled = scale(X)
    X_scaled = minmax_scale(X_scaled)

    # Splitting into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size=0.20, random_state=50)

    # Splitting into training and validation sets
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=50)

    # Building classifiers (add int the classifers here)
    DecisionTree = build_DecisionTree_classifier(X_train, y_train)
    NearrestNeighbour = build_NearrestNeighbours_classifier(X_train, y_train)
    NeuralNetwork = build_NeuralNetwork_classifier(X_train,y_train)
    SupportVectorMachine = build_SupportVectorMachine_classifier(X_train, y_train)

    # Printout results
    print("------------------------------------")
    print("Prediction Accuracy for Training set")
    print("------------------------------------")
    create_table(DecisionTree,NearrestNeighbour,NeuralNetwork, SupportVectorMachine, X_train, y_train)
    print()

    print("-----------------------------------")
    print("Prediction Accuracy for Testing set")
    print("-----------------------------------")
    create_table(DecisionTree,NearrestNeighbour,NeuralNetwork, SupportVectorMachine, X_test,y_test)
    print()

    print("--------------------------------------")
    print("Prediction Accuracy for Validation set")
    print("--------------------------------------")
    create_table(DecisionTree,NearrestNeighbour,NeuralNetwork,SupportVectorMachine, X_val,y_val)
    print()


########################################################################################
'''
    ## Nearrest Neighbours
    kNN = build_NearrestNeighbours_classifier(X_train, y_train)
    kNN_expect = y_test
    kNN_pred = kNN.predict(X_test)
    print(kNN_pred)
    test_report = metrics.classification_report(kNN_expect, kNN_pred)
    print("Nearrest Neighbours Test Report:")
    print(test_report)
    accuracy = accuracy_score(kNN_expect, kNN_pred)
    print("Nearrest Neighbours Test Accuracy score:")
    print(accuracy)
    mse = mean_squared_error(kNN_expect, kNN_pred)
    print("The mean-squared error of Nearrest Neighbours predictor on training data:")
    print(mse)

    ## Support Vector Machine

    ## Neural Network
    NN = build_NeuralNetwork_classifier(X_train, y_train)
    NN_expect = y_test
    NN_pred = NN.predict(X_test)
    test_report = metrics.classification_report(NN_expect, NN_pred)
    print("Neural Network Test Report:")
    print(test_report)
    accuracy = accuracy_score(NN_expect, NN_pred)
    print("Neural Network Test Accuracy score:")
    print(accuracy)
    mse = mean_squared_error(NN_expect, NN_pred)
    print("The mean-squared error of Neural Network predictor on training data:")
    print(mse)

'''
