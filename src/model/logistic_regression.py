# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer
from model.network import Network

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm
    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int
    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.1, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        self.network = Network(learningRate)
        
    def train(self, verbose=True):
        """Train the Logistic Regression.
        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Here you have to implement training method "epochs" times
        # Please using LogisticLayer class
        self.epochs = 1
        for i in range(0,self.epochs):
            print i
            self.network.train(self.trainingSet.input, self.trainingSet.label)
        pass

    def classify(self, testInstance):
        """Classify a single instance.
        Parameters
        ----------
        testInstance : list of floats
        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """

        # Here you have to implement classification method given an
        # instance
        res = self.network.classify(testInstance)
        if res > 0.5:
            return 1
        else:
            return 0

    def evaluate(self, test=None):
        """Evaluate a whole dataset.
        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used
        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))
