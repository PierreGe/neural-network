import utils
import numpy as np

from NeuralNetwork import NeuralNetwork

def verifGradient():
    sigma = 1e-4

    neuralNetwork = NeuralNetwork(3,4,2)
    X = [0.4,0.7,0.3]
    y = 1 # imaginons que c'est un point de la classe 2
    neuralNetwork.fprop(X)
    neuralNetwork.bprop(X,y)
    calculategrad = neuralNetwork._gradw1
    lostInitial = -(np.log(neuralNetwork._os[y][0]))

    numgrad = []
    for i in range(len(neuralNetwork._w1)):
        for j in range(len(neuralNetwork._w1[0])):
            neuralNetwork._w1[i][j] += sigma
            neuralNetwork.fprop(X)
            neuralNetwork.bprop(X,y)
            lost2 = -(np.log(neuralNetwork._os[y][0]))
            neuralNetwork._w1[i][j] -= sigma
            numgrad.append((lost2 - lostInitial) / sigma)

    print("Verification gradient w1")
    print(utils.ratioGrad(list(calculategrad.ravel()), numgrad))

    neuralNetwork = NeuralNetwork(3,4,2)
    X = [0.4,0.7,0.3]
    y = 1 # imaginons que c'est un point de la classe 2
    neuralNetwork.fprop(X)
    neuralNetwork.bprop(X,y)
    calculategrad = neuralNetwork._gradb1
    lostInitial = -(np.log(neuralNetwork._os[y][0]))

    numgrad = []
    for i in range(len(neuralNetwork._b1)):
        for j in range(len(neuralNetwork._b1[0])):
            neuralNetwork._b1[i][j] += sigma
            neuralNetwork.fprop(X)
            neuralNetwork.bprop(X,y)
            lost2 = -(np.log(neuralNetwork._os[y][0]))
            neuralNetwork._b1[i][j] -= sigma
            numgrad.append((lost2 - lostInitial) * 1./ sigma)

    print("Verification gradient b1")
    print(utils.ratioGrad(list(calculategrad.ravel()), numgrad))

    neuralNetwork = NeuralNetwork(3,4,2)
    X = [0.6,0.7,0.3]
    y = 1 # imaginons que c'est un point de la classe 2
    neuralNetwork.fprop(X)
    neuralNetwork.bprop(X,y)
    calculategrad = neuralNetwork._gradw2
    lostInitial = -(np.log(neuralNetwork._os[y][0]))

    numgrad = []
    for i in range(len(neuralNetwork._w2)):
        for j in range(len(neuralNetwork._w2[0])):
            neuralNetwork._w2[i][j] += sigma
            neuralNetwork.fprop(X)
            neuralNetwork.bprop(X,y)
            lost2 = -(np.log(neuralNetwork._os[y][0]))
            neuralNetwork._w2[i][j] -= sigma
            numgrad.append((lost2 - lostInitial) * 1./ sigma)

    print("Verification gradient w2")
    print(utils.ratioGrad(list(calculategrad.ravel()), numgrad))


    neuralNetwork = NeuralNetwork(3,4,2)
    X = [0.4,0.7,0.3]
    y = 1 # imaginons que c'est un point de la classe 2
    neuralNetwork.fprop(X)
    neuralNetwork.bprop(X,y)
    calculategrad = neuralNetwork._gradb2
    lostInitial = -(np.log(neuralNetwork._os[y][0]))

    numgrad = []
    for i in range(len(neuralNetwork._b2)):
        for j in range(len(neuralNetwork._b2[0])):
            neuralNetwork._b2[i][j] += sigma
            neuralNetwork.fprop(X)
            neuralNetwork.bprop(X,y)
            lost2 = -(np.log(neuralNetwork._os[y][0]))
            neuralNetwork._b2[i][j] -= sigma
            numgrad.append((lost2 - lostInitial) * 1./ sigma)

    print("Verification gradient b2")
    print(utils.ratioGrad(list(calculategrad.ravel()), numgrad))


if __name__ == '__main__':

    verifGradient()