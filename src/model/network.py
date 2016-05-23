class Network:

    
    """
    layertype
    nlayer
    narray int nneuronsPerLayer, at neuronsPL[0] it denotes the size of
    the input vector, so neuronsPL.size is one more than the
    number of layers
    """
    def __init__(self, learningRate, nlayer=1, neuronsPL=[785,1],
    layertype='logisticLayer'):
        self.learningRate = learningRate
        self.layertype = layertype
        self.nlayer = nlayer
        self.neuronsPL = neuronsPL
        self.layers = []
        if layertype is "LogisticLayer":
            for layerIndex in range(1, neuronsPL.size-1):
                self.layers[layerIndex] = LogisticLayer(neuronsPL[layerIndex-1],
                                                   neuronsPL[layerIndex])

            self.layers[neuronsPL.size - 1] = LogisticLayer(neuronsPL[neuronsPL.size - 2],
                                                       neuronsPL[neuronsPL.size - 1],
                                                       is_classifier_layer=True)
        else:
            pass

    def train(self, inputImages, inputTargets):
        for image in inputImages:
            evaluate(image)
            for layer in range(layers.size-1, -1, -1):
                updateWeights(self.learningRate, 0, 0, inputTargets)
        
        
    def evaluate(self, inputImage):
        inp = inputImage
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp
