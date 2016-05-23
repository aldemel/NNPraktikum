#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
#    myStupidClassifier = StupidRecognizer(data.trainingSet,
#                                          data.validationSet,
#                                          data.testSet)
    # Uncomment this to make your Perceptron evaluated
    # myPerceptronClassifier = Perceptron(data.trainingSet,
    #                                     data.validationSet,
    #                                     data.testSet,
    #                                     learningRate=0.005,
    #                                     epochs=30)

    myNeuralNetwork = LogisticRegression(data.trainingSet,
                                         data.validationSet,
                                         data.testSet)
     # Train the classifiers
    print("=========================")
    print("Training..")

    # print("\nStupid Classifier has been training..")
    # myStupidClassifier.train()
    # print("Done..")

    print("\nNeural Network training..")
    #myPerceptronClassifier.train()
    myNeuralNetwork.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    # stupidPred = myStupidClassifier.evaluate()
    # Uncomment this to make your Perceptron evaluated
    # perceptronPred = myPerceptronClassifier.evaluate()
    neuralPred = myNeuralNetwork.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    # print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    # evaluator.printAccuracy(data.testSet, stupidPred)

    print("\nResult of the Perceptron recognizer:")
#    evaluator.printComparison(data.testSet, perceptronPred)
#    Uncomment this to make your Perceptron evaluated
    # evaluator.printAccuracy(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, neuralPred)

    # evaluator.printConfusionMatrix(data.testSet, perceptronPred)
    evaluator.printConfusionMatrix(data.testSet, neuralPred)
#    evaluator.printClassificationResult(data.testSet, perceptronPred, target_names)

if __name__ == '__main__':
    main()
