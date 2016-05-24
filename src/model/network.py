import numpy as np
from model.logistic_layer import LogisticLayer

class Network:

    """
    layertype
    nlayer
    narray int nneuronsPerLayer, at neuronsPL[0] it denotes the size of
    the input vector, so neuronsPL.size is one more than the
    number of layers
    """
    def __init__(self, learningRate, nlayer=1, neuronsPL=[785,1], layertype='logisticLayer'):
        self.learningRate = learningRate
        self.layertype = layertype
        self.nlayer = nlayer
        self.neuronsPL = neuronsPL
        self.layers = []
        if layertype is "logisticLayer":
            for layerIndex in range(1, len(neuronsPL)-1):
                self.layers[layerIndex] = LogisticLayer(neuronsPL[layerIndex-1],
                                                        neuronsPL[layerIndex])
            self.layers.append(LogisticLayer(neuronsPL[len(neuronsPL) - 2],
                                                            neuronsPL[len(neuronsPL)- 1],
                                                            is_classifier_layer=True))
        else:
            pass

    def train(self, inputImages, inputTargets):
        for image in inputImages:
            output = self.classify(image)
            print "out: " + str(output.shape)
            print str(output.shape)
            print output[0]
            print output[1]
#            print "targets: " + str(inputTargets.shape)
            error = np.subtract(inputTargets, output)
            for layer in range(len(self.layers)-1, -1, -1):
                self.layers[layer].updateWeights(self.learningRate, 0, 0, inputTargets)
        
    def classify(self, inputImage):
        inp = np.insert(inputImage, 0, 1)
        #move data through all layers
        print "image: " + str(inp.shape)
        for layer in self.layers:
            inp = layer.forward(inp)
        print "image: " + str(inp.shape)
        return inp
