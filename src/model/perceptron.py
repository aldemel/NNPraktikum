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
    def __init__(self, train, valid, test, learningRate=0.010, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and 0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/10
        self.w0 = 0.1

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # iterate over all images
        for epoc in range(0, self.epochs):
            # Variante mit update für jedes Bild
            for x in range(0, self.trainingSet.input.size/pixels -1):
                self.trainWeightsWithImage(self.trainingSet.input[x],
                                           self.trainingSet.label[x])
                
                # variante mit update nach sichtung aller Bilder
#            self.trainWeightsForAllImages(self.trainingSet)
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

        decision = self.decisionFunction(testInstance)
        if(decision > 0):
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

        return list(map(self.classify, test))

    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        # I already implemented it for you to see how you can work with numpy
        return Activation.sign(np.dot(np.array(input), self.weight))

    def decisionFunction(self, pixel):
        returnValue = self.w0
        returnValue += np.dot(np.array(pixel), self.weight)
        return returnValue;

    def trainWeightsWithImage(self, image, label):
        classifiedValue = self.classify(image)
        self.w0 += self.learningRate * (label - classifiedValue)
        delta = np.multiply(label - classifiedValue, image)
        delta = np.multiply(self.learningRate, delta)
        self.weight = np.add(self.weight, delta)

#    def trainWeightsForAllImages(trainingSet):
#        weightUpdate = (0..0)
#        for image in range(0, trainingSet.input.size):
#            classifiedValue = self.classify(trainingSet.input[image])
            
#            for(pixel in range(0, pixels):
#                weightUpdate[pixel] += self.learningRate * (label -
#                classifiedValue) * self.trainingSet.input[pixel]
    
