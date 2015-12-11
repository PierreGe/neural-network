# -*- coding: utf-8 -*-
import utils
import numpy as np


class NeuralNetwork(object):
    """docstring for NeuralNetwork"""

    def __init__(self, d, h, m, wd=0):
        self._d = d
        self._h = h
        self._m = m
        self.wd = wd  # weight-decay

        self._w1 = utils.randomArray(d, h, d)  # h x d
        self._w2 = utils.randomArray(h, m, h)  # m x h
        self._b1 = np.array([[0.] for i in range(h)])  # h
        self._b2 = np.array([[0.] for i in range(m)])  # m

        # self._K = 0.2 # hyperparametre K  ???

    def fprop(self, X):
        X = np.array([[float(x)] for x in X])
        self._ha = np.dot(self._w1, X) + self._b1  # valeur des synapses entre x et hidden
        self._hs = utils.relu(self._ha)  # valeur hidden
        self._oa = np.dot(self._w2, self._hs) + self._b2  # valeur entre hidden et sortie
        self._os = utils.softmax(self._oa)  # valeur de sortie

    def bprop(self, X, y):
        X = np.array([[float(x)] for x in X])
        self._gradoa = np.array([i[0] for i in self._os]) - utils.onehot(self._m,
                                                                         y)  # todo Vecteur ligne alors que tous les autres sont des vecteurs colonnes. Normal?
        self._gradb2 = np.array([[i] for i in self._gradoa])
        self._gradw2 = np.dot(np.array([[i] for i in self._gradoa]), np.transpose(
                self._hs)) + 2 * self.wd * self._w2  # todo verifer ajout terme regularisation
        self._gradhs = np.dot(np.transpose(self._w2), np.array([[i] for i in self._gradoa]))
        self._gradha = self._gradhs * np.where(self._ha > 0, 1, 0)
        self._gradb1 = np.array([i for i in
                                 self._gradha])  # todo Crochet retirés autour de i. Étaient-ils nécéssaires? on peut simplifier par gradb1 = gradha puisqu'ils sont 2 vecteurs colonnes
        self._gradw1 = np.dot(self._gradha,
                              np.transpose(X)) + 2 * self.wd * self._w1  # todo verifer ajout terme regularisation
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

        for iter in range(maxIter):
            classificationErrorFound = False

            for elem in range(len(X)):

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
