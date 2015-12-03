import utils
import numpy as np

from NeuralNetwork import NeuralNetwork


def verifW1(neuralNetwork,X,y,sigma):
    # perte sur ces donnees pour w1
    neuralNetwork.fprop(X)
    neuralNetwork.bprop(X,y)
    calculategrad = neuralNetwork._gradw1
    lostInitial = -(np.log(neuralNetwork._os[y][0]))

    # calcul numerique du gradient
    numgrad = []
    for i in range(len(neuralNetwork._w1)):
        for j in range(len(neuralNetwork._w1[0])):
            neuralNetwork._w1[i][j] += sigma
            neuralNetwork.fprop(X)
            neuralNetwork.bprop(X,y)
            lost2 = -(np.log(neuralNetwork._os[y][0]))
            neuralNetwork._w1[i][j] -= sigma
            numgrad.append((lost2 - lostInitial) / sigma)
    return list(calculategrad.ravel()), numgrad

def verifb1(neuralNetwork,X,y,sigma):
    # perte sur ces donnees pour b1
    neuralNetwork.fprop(X)
    neuralNetwork.bprop(X,y)
    calculategrad = neuralNetwork._gradb1
    lostInitial = -(np.log(neuralNetwork._os[y][0]))

    # calcul numerique du gradient
    numgrad = []
    for i in range(len(neuralNetwork._b1)):
        for j in range(len(neuralNetwork._b1[0])):
            neuralNetwork._b1[i][j] += sigma
            neuralNetwork.fprop(X)
            neuralNetwork.bprop(X,y)
            lost2 = -(np.log(neuralNetwork._os[y][0]))
            neuralNetwork._b1[i][j] -= sigma
            numgrad.append((lost2 - lostInitial) * 1./ sigma)
    return list(calculategrad.ravel()), numgrad

def verifW2(neuralNetwork,X,y,sigma):
    # perte sur ces donnees pour w2
    neuralNetwork.fprop(X)
    neuralNetwork.bprop(X,y)
    calculategrad = neuralNetwork._gradw2
    lostInitial = -(np.log(neuralNetwork._os[y][0]))

    # calcul numerique du gradient
    numgrad = []
    for i in range(len(neuralNetwork._w2)):
        for j in range(len(neuralNetwork._w2[0])):
            neuralNetwork._w2[i][j] += sigma
            neuralNetwork.fprop(X)
            neuralNetwork.bprop(X,y)
            lost2 = -(np.log(neuralNetwork._os[y][0]))
            neuralNetwork._w2[i][j] -= sigma
            numgrad.append((lost2 - lostInitial) * 1./ sigma)
    return list(calculategrad.ravel()), numgrad

def verifb2(neuralNetwork,X,y,sigma):
    # perte sur ces donnees pour b2
    neuralNetwork.fprop(X)
    neuralNetwork.bprop(X,y)
    calculategrad = neuralNetwork._gradb2
    lostInitial = -(np.log(neuralNetwork._os[y][0]))

    # calcul numerique du gradient
    numgrad = []
    for i in range(len(neuralNetwork._b2)):
        for j in range(len(neuralNetwork._b2[0])):
            neuralNetwork._b2[i][j] += sigma
            neuralNetwork.fprop(X)
            neuralNetwork.bprop(X,y)
            lost2 = -(np.log(neuralNetwork._os[y][0]))
            neuralNetwork._b2[i][j] -= sigma
            numgrad.append((lost2 - lostInitial) * 1./ sigma)
    return list(calculategrad.ravel()), numgrad

def verifGradient1d(neuralNetwork, X, y, sigma = 1e-4):
    w1nn, w1numerical = verifW1(neuralNetwork,X,y,sigma)
    ratioW1 = utils.ratioGrad(w1nn, w1numerical)

    b1nn, b1numerical = verifb1(neuralNetwork,X,y,sigma)
    ratiob1 = utils.ratioGrad(b1nn, b1numerical)

    w2nn, w2numerical = verifW2(neuralNetwork,X,y,sigma)
    ratiow2 = utils.ratioGrad(w2nn, w2numerical)

    b2nn, b2numerical = verifb2(neuralNetwork,X,y,sigma)
    ratiob2 = utils.ratioGrad(b2nn, b2numerical)

    return ratioW1, ratiob1, ratiow2, ratiob2


if __name__ == '__main__':

    verifGradient1d()