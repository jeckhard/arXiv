"""
Module making definitions to train multiple different models and functions to plot the results

Functions:
----------
createArxivData : Create subset of training and cv data from the encoded arXiv data set
xgboost : Trains an XGBoost model for each label for the arXiv data set
denseNN : Trains a Keras dense NN for each label for the arXiv data set
svm : Trains a Keras dense NN for each label for the arXiv data set
forest : Trains a Keras dense NN for each label for the arXiv data set
resultsDifferentK : Trains a model for different values of k and different labels and evaluates it on train and test data.
plotDifferentK : Plot measures for a model, given a different amount of training data

"""

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, AUC

import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np



def createArxivData(labelColumns=['cs','physics','math'],k=10,testSize=.2):
    """
    Create subset of training and cv data from the encoded arXiv data set

    :param labelColumns: list of str elements
        sublist of ['cs','physics','math'] (default)
    :param k: int
        amount of data ~ 1/k
    :param testSize: float (0,1)
        percentage of cv data
    :return: tuple of data frames
        0 : training features
        1 : cv features
        2 : training labels
        3 : cv labels
    """

    featureColumns = []
    for i in range(512):
        featureColumns.append("encoding" + str(i))

    data = pd.read_csv("arxiv-encoded-full.csv", skiprows=lambda x: x != 0 and x % k != 0)

    features = data[featureColumns]
    labels = data[labelColumns]

    X_train, X_cv, y_train, y_cv = train_test_split(features, labels, test_size=testSize)

    return X_train, X_cv, y_train, y_cv


def xgboost(data, maxDepth = 6, eta = .3, objective = 'binary:logistic',metric = 'error',numRounds = 10,verbose=True):
    """
    Trains an XGBoost model for each label for the arXiv data set

    :param data: tuple of data frames
        0 : training features
        1 : cv features
        2 : training labels
        3 : cv labels
    :param maxDepth: int
        maximum tree depth, default=6
    :param eta: float
        learning rate, default=0.3
    :param objective: str
        splitting criterium for XGBoost, default=binary:logistic
    :param metric: str
        evaluating metric for XGBoost, default='error'
    :param numRounds: int
        number of boosting rounds, default=10
    :param verbose: bool
        True : evaluation is printed (default)
        False : nothing is printed
    :return: list of XGBoost models
    """
    X_train, X_cv, y_train, y_cv = data

    labels = y_train.columns

    models = []
    for label in labels:
        d_train = xgb.DMatrix(X_train, label=y_train[label])
        d_cv = xgb.DMatrix(X_cv, label=y_cv[label])

        evallist = [(d_cv, 'cv'), (d_train, 'train')]
        parameters = {'max_depth': maxDepth, 'eta': eta, 'objective': objective, 'eval_metric': metric}

        if verbose:
            print('label:',label)
        bst = xgb.train(parameters, d_train, numRounds, evallist,verbose_eval=1 if verbose else 0)
        models.append(bst)

    return models


def denseNN(data, denseLayers=1,denseNodes=128,denseActivation='relu',outputActivation='sigmoid',optimizer='adam',loss=binary_crossentropy,metric='accuracy',epochs=5,verbose=True):
    """
    Trains a Keras dense NN for each label for the arXiv data set

    :param data: tuple of data frames
        0 : training features
        1 : cv features
        2 : training labels
        3 : cv labels
    :param denseLayers: int
        number of dense layers, default=1
    :param denseNodes: int
        number of nodes per dense layer, default=128
    :param denseActivation: str or keras activation
        activation for the dense layers, default='relu'
    :param outputActivation: str or keras activation
        activation for the output layer, default='sigmoid'
    :param optimizer: str or keras optimizer
        optimizer, default='adam'
    :param loss: str or keras loss
        loss, default=binary_crossentropy
    :param metric: str
        metrics, one of 'accuracy'(default), 'precission', 'recall', 'auc'
    :param epochs: int
        number of epochs, default=5
    :param verbose: Bool
        True : evaluation is printed (default)
        False : nothing is printed
    :return: list of keras models
    """
    X_train, X_cv, y_train, y_cv = data

    labels = y_train.columns

    scoring = {'auc': AUC(),'precision': Precision(), 'recall': Recall(), 'accuracy': BinaryAccuracy()}

    models = []
    for label in labels:
        model = Sequential()
        for i in range(denseLayers):
            model.add(Dense(denseNodes,activation=denseActivation))
        model.add(Dense(1,activation=outputActivation))

        model.compile(optimizer=optimizer,loss=loss,metrics=[scoring[metric]])

        if verbose:
            print("label:",label)
        model.fit(X_train,y_train[label],epochs=epochs,verbose = 2 if verbose else 0,validation_data= (X_cv,y_cv[label]) if verbose else None)
        if verbose:
            print(model.evaluate(X_cv,y_cv[label]))
            print("-------------------")

        models.append(model)

    return models


def svm(data,regulariser=1.0,kernel='rbf',degree=3,metric='accuracy',verbose=True):
    """
    Trains a support vector machine for each label for the arXiv data set

    :param data: tuple of data frames
        0 : training features
        1 : cv features
        2 : training labels
        3 : cv labels
    :param regulariser: float
        regulariser C, default=1.0
    :param kernel: str
        svm kernel, one of 'rbf'(default), 'linear', 'poly'
    :param degree: int
        degree of a polynomial kernel, default=3
    :param metric: str
        metric, one of 'accuracy'(default), 'precision', 'recall'
    :param verbose: Bool
        True : evaluation is printed (default)
        False : nothing is printed
    :return: list of SVMs
    """
    X_train, X_cv, y_train, y_cv = data

    labels = y_train.columns

    scoring = {'precision': precision_score, 'recall': recall_score, 'accuracy': accuracy_score}

    models = []
    for label in labels:
        if verbose:
            print('label:', label)
        model = SVC(C=regulariser,kernel=kernel,degree=degree)
        model.fit(X_train,y_train[label])
        if verbose:
            print(scoring[metric](y_cv[label],model.predict(X_cv)))
            print("-------------------")

        models.append(model)

    return models


def forest(data,nEstimators=100,criterion='gini',maxDepth=6,metric='accuracy',verbose=True):
    """
        Trains a random forest for each label for the arXiv data set

        :param data: tuple of data frames
            0 : training features
            1 : cv features
            2 : training labels
            3 : cv labels
        :param nEstimators: int
            number of decision trees, default=100
        :param criterion: str
            splitting criterion, one of 'gini'(default), 'entropy'
        :param metric: int or None
            int : maximum depth of the decision trees, default=6
            None : no maximum depth
        :param metric: str
            metric, one of 'accuracy'(default), 'precision', 'recall'
        :param verbose: Bool
            True : evaluation is printed (default)
            False : nothing is printed
        :return: list of random forests
        """

    X_train, X_cv, y_train, y_cv = data

    labels = y_train.columns

    scoring = {'precision': precision_score, 'recall': recall_score, 'accuracy': accuracy_score}

    models = []
    for label in labels:
        if verbose:
            print('label:', label)
        model = RandomForestClassifier(n_estimators=nEstimators,criterion=criterion,max_depth=maxDepth)
        model.fit(X_train, y_train[label])
        if verbose:
            print(scoring[metric](y_cv[label], model.predict(X_cv)))
            print("-------------------")

        models.append(model)

    return models


def resultsDifferentK(kList,labels,modelType,parameters={'verbose' : False},train=True,test=True,testSize=.2):
    """
    Trains a model for different values of k and different labels and evaluates it on train and test data.

    :param kList: list of int elements
        list of values for k, amount of data ~1/k
    :param labels: list of str elements
        sublist of ['cs','physics','math'] (default)
    :param modelType: str
        type of model to be trained
        'denseNN' : dense neural network
        'xgboost' : XGBoost model
        'svm' : Support vector machine
        'forest' : Random forest
        raise ValueError for all other values
    :param parameters: dict
        optional parameters for model, see denseNN and xgboost, default={'verbose' : False}
        default eval metric : ROC-AUC
    :param train: bool
        evauluate training error, default=True
    :param test: bool
        evauluate test error, default=True
    :param testSize: float
        percentage of test data, default=.2
    :return: dict
        key-value pairs : str : list of float elements
        keys : str of the form label + train/test + metric
        values : measures for different k
    """

    if "verbose" not in parameters:
        parameters["verbose"] = False

    defaultMeasures = {'xgboost': 'error', 'denseNN': 'accuracy', 'svm': 'accuracy', 'forest': 'accuracy'}
    if modelType not in defaultMeasures:
        raise ValueError('Model type not known')

    if 'metric' not in parameters:
        parameters['metric'] = defaultMeasures[modelType]

    results = {}
    trainKeys=[]
    testKeys=[]
    for label in labels:
        if train:
            trainKey = label + " train "+parameters['metric'] + " " + modelType
            results[trainKey] = []
            trainKeys.append(trainKey)
        if test:
            testKey = label + " test " + parameters['metric'] + " " + modelType
            results[testKey] = []
            testKeys.append(testKey)
    for k in kList:
        data = createArxivData(labels, k=k,testSize=testSize)
        if modelType == 'denseNN':
            models = denseNN(data, **parameters)
            for i in range(len(labels)):
                if train:
                    results[trainKeys[i]].append(round(models[i].evaluate(data[0], data[2][labels[i]], verbose=0)[1],4))
                if test:
                    results[testKeys[i]].append(round(models[i].evaluate(data[1], data[3][labels[i]], verbose=0)[1],4))
        elif modelType == 'xgboost':
            models = xgboost(data, **parameters)
            for i in range(len(labels)):
                if train:
                    result_str = models[i].eval(xgb.DMatrix(data[0],label=data[2][labels[i]]))
                    results[trainKeys[i]].append(round(float(result_str[result_str.index(":")+1:]),4))
                if test:
                    result_str = models[i].eval(xgb.DMatrix(data[1], label=data[3][labels[i]]))
                    results[testKeys[i]].append(round(float(result_str[result_str.index(":") + 1:]),4))
        elif modelType == 'svm':
            models = svm(data, **parameters)
            svmScoring = {'precision': precision_score, 'recall': recall_score,'accuracy': accuracy_score}
            for i in range(len(labels)):
                if train:
                    results[trainKeys[i]].append(round(svmScoring[parameters['metric']](data[2][labels[i]],models[i].predict(data[0])),4))
                if test:
                    results[testKeys[i]].append(round(svmScoring[parameters['metric']](data[3][labels[i]],models[i].predict(data[1])),4))
        elif modelType == 'forest':
            models = forest(data, **parameters)
            forestScoring = {'precision': precision_score, 'recall': recall_score, 'accuracy': accuracy_score}
            for i in range(len(labels)):
                if train:
                    results[trainKeys[i]].append(round(forestScoring[parameters['metric']](data[2][labels[i]], models[i].predict(data[0])), 4))
                if test:
                    results[testKeys[i]].append(round(forestScoring[parameters['metric']](data[3][labels[i]], models[i].predict(data[1])), 4))

    return results


def plotDifferentK(kList,measures,yLabel=None,title=None,legend=True,show=False,save=None,kMode='lin'):
    """
    Plot measures for a model, given a different amount of training data

    :param kList: list of int elements of length n
        amount of training data ~1/k for each element
    :param measures: dict
        key-value pairs: str : list of floats of length n
        keys : measure names
        values : measures for different values of k
    :param yLabel: str or None(default)
    :param title: str or None(default)
    :param legend: bool, default=True
    :param show: bool
        True : plot is shown
        False : plot is not shown (default)
    :param save: str or None (default)
        str : plot is saved to str
        None : plot is not saved (default)
    :param kMode: str
        'lin' : linear plot (default)
        'inv' : inverse plot
        'log' : log plot
    :return: None
    """
    points = len(kList)
    for measure in measures:
        if len(measures[measure]) != points:
            raise ValueError("Wrong number of measurement points")

    if kMode == 'inv':
        kList = np.array([1/k for k in kList])

    fig, ax = plt.subplots()

    for measure in measures:
        ax.plot(kList,measures[measure],label= measure)

    if kMode == 'inv':
        ax.set_xlabel("1/k")
    else:
        ax.set_xlabel("k")

    if kMode == 'log':
        plt.xscale('log')

    if yLabel:
        ax.set_ylabel(yLabel)
    if title:
        ax.set_title(title)

    if legend:
        ax.legend()

    if show:
        plt.show()

    if save:
        plt.savefig(save)

