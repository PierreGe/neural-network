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
        self._x = np.array([[float(x)] for x in X])
        self._ha = np.dot(self._w1, self._x ) + self._b1   # valeur des synapses entre x et hidden
        self._hs = utils.relu(self._ha)  # valeur hidden
        self._oa = np.dot(self._w2,self._hs) + self._b2  # valeur entre hidden et sortie
        self._os = utils.softmax(self._oa)  # valeur de sortie

        #print(self._os)
        print("Cela devrait faire 1 :" + str(sum(self._os))) #TODO instabilite numerique ici

    def bprop(self, y):
        e_oa = np.array([np.exp(i[0]) for i in self._oa])
        self._gradoa = float(1/np.sum(e_oa)) * e_oa - utils.onehot(self._m,y)
        self._gradb2 = np.array([[i] for i in self._gradoa])
        self._gradw2 = np.dot(np.array([[i] for i in self._gradoa]), np.transpose(self._hs))
        self._gradhs = np.dot(np.transpose(self._w2), np.array([[i] for i in self._gradoa]))
        self._gradha = (self._gradhs + np.abs(self._gradhs))/2
        self._gradb1 = np.array([[i] for i in self._gradha])
        self._gradw1 = np.dot(np.array([[i] for i in self._gradha]), np.transpose(self._x))
        self._gradx = np.dot(np.transpose(self._w1), self._gradha)


if __name__ == '__main__':
    neuralNetwork = NeuralNetwork(4,6,3)
    neuralNetwork.fprop([30,20,45,50])

    neuralNetwork.bprop(3) # imaginons que c'est un point de la classe 3