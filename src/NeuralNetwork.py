import utils
import numpy as np


class NeuralNetwork(object):
    """docstring for NeuralNetwork"""

    def __init__(self, d, h, m):
        self._d = d
        self._h = h
        self._m = m

        self._w1 = utils.randomArray(d, h, d)  # h x d
        self._w2 = utils.randomArray(h, m, h)  # m x h
        self._b1 = np.array([[0] for i in range(h)])  # h
        self._b2 = np.array([[0] for i in range(m)])  # m

    def fprop(self, X):
        X = np.array([[float(x)] for x in X])
        self._ha = np.dot(self._w1, X) + self._b1   # valeur des synapses entre x et hidden
        self._hs = utils.relu(self._ha)  # valeur hidden
        self._oa = np.dot(self._w2,self._hs) + self._b2  # valeur entre hidden et sortie
        self._os = utils.softmax(self._oa)  # valeur de sortie

    def bprop(self,X , y):
        X = np.array([[float(x)] for x in X])
        self._gradoa = np.array([np.exp(i[0]) for i in self._os]) - utils.onehot(self._m,y)
        self._gradb2 = np.array([[i] for i in self._gradoa])
        self._gradw2 = np.dot(np.array([[i] for i in self._gradoa]), np.transpose(self._hs))
        self._gradhs = np.dot(np.transpose(self._w2), np.array([[i] for i in self._gradoa]))
        self._gradha = (self._gradhs + np.abs(self._gradhs))/2
        self._gradb1 = np.array([[i] for i in self._gradha])
        self._gradw1 = np.dot(self._gradha, np.transpose(X))
        self._gradx = np.dot(np.transpose(self._w1), self._gradha)


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
            lost1 = -(np.log(neuralNetwork._os[y][0]))
            neuralNetwork._w1[i][j] -= sigma
            numgrad.append((lostInitial - lost1) / sigma)

    print("cal grad w1")
    print(list(calculategrad.ravel()))
    print("num grad w1")
    print(numgrad)


    neuralNetwork = NeuralNetwork(3,4,2)
    X = [0.4,0.7,0.3]
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

    print("cal grad w2")
    print(list(calculategrad.ravel()))
    print("num grad w2")
    print(numgrad)


    neuralNetwork.bprop(X,y)


if __name__ == '__main__':
    #neuralNetwork = NeuralNetwork(4,6,3)
    ##X = [30,20,45,50]
    #y = 1 # imaginons que c'est un point de la classe 3
    ##neuralNetwork.fprop(X)
    #neuralNetwork.bprop(X,y)

    verifGradient()