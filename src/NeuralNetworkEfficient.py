# -*- coding: utf-8 -*-
import utils
import numpy as np

import NeuralNetwork
# Mathieu Bouchard && Pierre Gerard

class NeuralNetworkEfficient(NeuralNetwork.NeuralNetwork):
    """Optimisation of NeuralNetwork class with method rewrite"""

    def __init__(self, d, h, m, K=50, wd=0):
        super(NeuralNetworkEfficient, self).__init__(d, h, m, K, wd)

    def fprop(self, X):
        X = np.array([np.array([float(x) for x in j]) for j in X])
        X = X.transpose()
        self._ha = np.dot(self._w1, X) + np.repeat(self._b1, len(X[0]), axis=1)  # valeur des synapses entre x et hidden
        self._hs = utils.relu(self._ha)  # valeur hidden
        self._oa = np.dot(self._w2, self._hs) + np.repeat(self._b2, len(X[0]), axis=1)  # valeur entre hidden et sortie
        self._os = utils.softmax(self._oa)  # valeur de sortie
        #print(self._ha)

    def bprop(self, X, y):
        # chaque colonne de X est une entrée
        X = np.array([np.array([float(x) for x in j]) for j in X])
        X = X.transpose()
        self._gradoa = self._os - utils.onehot(self._m,y)
        self._gradb2 = self._gradoa
        # gradw2 va être la somme des gradient pour chaque point individuelle
        self._gradw2 = np.dot(self._gradoa, np.transpose(self._hs)) + 2 * self.wd * self._w2
        self._gradhs = np.dot(np.transpose(self._w2), self._gradoa)
        self._gradha = self._gradhs * np.where(self._ha > 0, 1, 0)
        self._gradb1 = np.array(self._gradha)
        # gradw2 va être la somme des gradient pour chaque point individuelle
        self._gradw1 = np.dot(self._gradha, np.transpose(X)) + 2 * self.wd * self._w1
        self._gradx = np.dot(np.transpose(self._w1), self._gradha)


    def computePredictions(self, X):
        predictions = []
        self.fprop(X)
        for os in np.transpose(self._os):
            predictions.append(np.argmax(os))

        return predictions


    def train(self, X, y, maxIter, eta=0.05):
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

            xbatch, ybatch = self._nextBatchIndex(X,y, batchNbr)

            self.fprop(xbatch)
            self.bprop(xbatch, ybatch)

            norm = (1. / len(xbatch))
            self._w1 -= eta * (self._gradw1 * norm)
            self._w2 -= eta * (self._gradw2 * norm)
            ub1 = np.sum(self._gradb1, axis=1) * norm
            self._b1 -= eta * np.array([[i] for i in ub1])
            ub2 = np.sum(self._gradb2, axis=1) * norm
            self._b2 -= eta * np.array([[i] for i in ub2])
            #print(self._gradb1)
            #print(self._w1,self._w2,self._b1,self._b2)

             #Pour #9-10
            if iter % 10 == 0:
                self._calculateEfficiency()
                self._calculateAverageCosts()

                if len(self.epochData) > 0:
                    print self.epochData[len(self.epochData)-1]+" ("+str(iter)+")"

    def calculateLoss(self, Y):
        res = 0
        for i,y in enumerate(Y):
            res -=(np.log(self._os[y][i]))
        return res

    def _calculateAverageCosts(self):
        if self.Xtrain is None or self.Xvalid is None or self.Xtest is None:
            pass
        else:
            #Calcule coût moyen sur ensemble d'entrainement, de test et de validation
            sumLTrain = 0
            self.fprop(self.Xtrain)
            sumLTrain += self.calculateLoss(self.ytrain)
            self.trainSumL.append(sumLTrain/len(self.Xtrain))
            self.epochData[len(self.epochData)-1] += ";"+"{:.3f}".format(self.trainSumL[len(self.trainSumL)-1])

            sumLValid = 0
            self.fprop(self.Xvalid)
            sumLValid += self.calculateLoss(self.yvalid)
            self.validSumL.append(sumLValid/len(self.yvalid))
            self.epochData[len(self.epochData)-1] += ";"+"{:.3f}".format(self.validSumL[len(self.validSumL)-1])

            sumLTest = 0
            self.fprop(self.Xtest)
            sumLTest += self.calculateLoss(self.ytest)
            self.testSumL.append(sumLTest/len(self.ytest))
            self.epochData[len(self.epochData)-1] += ";"+"{:.3f}".format(self.testSumL[len(self.testSumL)-1])

if __name__ == '__main__':
    self = NeuralNetwork(4, 6, 3)
    X = [30, 20, 45, 50]
    y = 1  # imaginons que c'est un point de la classe 3
    self.fprop(X)
    self.bprop(X, y)
