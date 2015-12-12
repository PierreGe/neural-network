# -*- coding: utf-8 -*-
import utils
import numpy as np


class NeuralNetworkEfficient(object):
    """docstring for NeuralNetwork"""

    def __init__(self, d, h, m, K=50, wd=0):
        self._d = d
        self._h = h
        self._m = m
        self.wd = wd  # weight-decay

        self._w1 = utils.randomArray(d, h, d)  # h x d
        self._w2 = utils.randomArray(h, m, h)  # m x h
        self._b1 = np.array([[0.] for i in range(h)])  # h
        self._b2 = np.array([[0.] for i in range(m)])  # m

        self._K = K #

    def fprop(self, X):
        X = np.array([np.array([[float(x)] for x in j]) for j in X])
        self._ha = np.dot(self._w1, X) + self._b1  # valeur des synapses entre x et hidden
        self._hs = utils.relu(self._ha)  # valeur hidden
        self._oa = np.dot(self._w2, self._hs) + self._b2  # valeur entre hidden et sortie
        self._os = utils.softmax(self._oa)  # valeur de sortie

    def bprop(self, X, y):
        X = np.array([[float(x)] for x in X])
        self._gradoa = np.array([i[0] for i in self._os]) - utils.onehot(self._m,y)
        self._gradoa = np.array([[i] for i in self._gradoa]) # obtenir un vecteur colonne
        self._gradb2 = self._gradoa
        self._gradw2 = np.dot(self._gradoa, np.transpose(self._hs)) + 2 * self.wd * self._w2
        self._gradhs = np.dot(np.transpose(self._w2), self._gradoa)
        self._gradha = self._gradhs * np.where(self._ha > 0, 1, 0)
        self._gradb1 = np.array([i for i in
                                 self._gradha])
        self._gradw1 = np.dot(self._gradha,
                              np.transpose(X)) + 2 * self.wd * self._w1
        self._gradx = np.dot(np.transpose(self._w1), self._gradha)

    def predict(self, x):
        self.fprop(x)
        klass = -1
        maxVal = -1

        for i in range(len(self._os)):
            if self._os[i] > maxVal:
                maxVal = self._os[i]
                klass = i

        return klass

    def computePredictions(self, X):
        predictions = []

        for x in X:
            predictions.append(self.predict(x))

        return predictions

    def _nextBatchIndex(self, X, batchNbr):
        correctedBatchNbr = batchNbr % len(X)/self._K
        size = len(X)
        born1 = batchNbr * self._K
        born2 = (batchNbr+1) * self._K
        if born2 > size:
            born1 = 0
            born2 = self._K
        return born1, born2

    def train(self, X, y, maxIter, eta=0.01):
        """
        :param X: données d'entrainement
        :param y: classes réelles des données X
        :param wd: Weight-decay (lambda)
        :param h: d_h (nombre d'unités cachées)
        :param maxIter: condition d'arrêt prématurée (nombre d'époques d'entrainement)
        :param eta: Taille du pas
        :return:
        """
        batchNbr = 0
        for iter in range(maxIter):
            batchNbr+=1
            classificationErrorFound = False

            born1, born2 = self._nextBatchIndex(X, batchNbr)
            for elem in range(born1,born2):

                prediction = self.predict(X[elem])

                if prediction != y[elem]:
                    classificationErrorFound = True

                    self.fprop(X[elem])
                    self.bprop(X[elem], y[elem])

                    self._w1 -= eta * self._gradw1
                    self._w2 -= eta * self._gradw2
                    self._b1 -= eta * self._gradb1
                    self._b2 -= eta * self._gradb2

            if not classificationErrorFound:
                break


if __name__ == '__main__':
    self = NeuralNetwork(4, 6, 3)
    X = [30, 20, 45, 50]
    y = 1  # imaginons que c'est un point de la classe 3
    self.fprop(X)
    self.bprop(X, y)
