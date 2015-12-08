# -*- coding: utf-8 -*-
import numpy as np
from utils import getClassCount
from NeuralNetwork import NeuralNetwork

#X = données d'entrainement
#y = classes réelles des données X
#Hyper-paramètres:
#wd = Weight-decay (lambda)
#h = d_h (nombre d'unités cachées)
#maxIter = condition d'arrêt prématurée (nombre d'époques d'entrainement)
#eta = Taille du pas
def trainNetwork(X, y, wd, h, maxIter, eta = 0.01):
    d = len(X[0])
    m = getClassCount(y)

    neuralNetwork = NeuralNetwork(d, h, m, wd)

    for iter in range(maxIter):
        classificationErrorFound = False

        for elem in range(len(X)):

            prediction = neuralNetwork.predict(X[elem])

            if prediction != y[elem]:
                classificationErrorFound = True

                neuralNetwork.fprop(X[elem])
                neuralNetwork.bprop(X[elem], y[elem])

                neuralNetwork._w1 -= eta * neuralNetwork._gradw1
                neuralNetwork._w2 -= eta * neuralNetwork._gradw2
                neuralNetwork._b1 -= eta * neuralNetwork._gradb1
                neuralNetwork._b2 -= eta * neuralNetwork._gradb2

        if not classificationErrorFound:
            break

    return neuralNetwork