# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)
pixels = 784


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and 0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/10

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # iterate over all images
        for x in range(0, 1000):
            #boolean
            fired = self.fire(self.trainingSet.input[x])
            firedInt = 0
            if(fired):
                firedInt = 1
            #int, 1 if it is a 7 otherwise 0
            expected = self.trainingSet.label[x]
            factor = expected - firedInt
            #not sure if decision is the right parameter here
            decision = self.decisionFunction(self.trainingSet.input[x])
            for weightEntry in range(0, pixels):
                weightChange = self.learningRate * self.trainingSet.input[x][weightEntry] * factor
                self.weight[weightEntry] += weightChange
        # Here you have to implement the Perceptron Learning Algorithm
        # to change the weights of the Perceptron
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
        

        # Here you have to implement the classification for one instance,
        # i.e., return True if the testInstance is recognized as a 7,
        # False otherwise

        # decisionValue = 0
        # for x in range(0, 783):
        #     decisionValue += self.weight[x]*testInstance[x]
        # print decisionValue
        return self.decisionFunction(testInstance)
        pass

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

        # Here is the map function of python - a functional programming concept
        # It applies the "classify" method to every element of "test"
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        # I already implemented it for you to see how you can work with numpy
        return Activation.sign(np.dot(np.array(input), self.weight))

    def decisionFunction(self, pixel):
        returnValue = 0
        for x in range(0, pixel.size):
            returnValue += pixel[x]*self.weight[x]
        return returnValue;
