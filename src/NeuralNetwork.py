# -*- coding: utf-8 -*-
import utils
import numpy as np


class NeuralNetwork(object):
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

        #Erreurs pour chaque epoque
        self.trainError = []
        self.validError = []
        self.testError = []

        #Somme du cout optimise total (somme des L encourus) pour chaque epoque
        self.trainSumL = []
        self.validSumL = []
        self.testSumL = []

        self.Xtrain = None
        self.Xvalid = None
        self.Xtest = None
        self.ytrain = None
        self.yvalid = None
        self.ytest = None
        self.epochData = []

    def setDataSets(self, Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest):
        self.Xtrain = Xtrain
        self.Xvalid = Xvalid
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.yvalid = yvalid
        self.ytest = ytest

    def fprop(self, X):
        X = np.array([[float(x)] for x in X])
        self._ha = np.add(np.dot(self._w1, X), self._b1) # valeur des synapses entre x et hidden
        self._hs = utils.relu(self._ha)  # valeur hidden
        self._oa = np.add(np.dot(self._w2, self._hs), self._b2)  # valeur entre hidden et sortie
        self._os = utils.softmax(self._oa)  # valeur de sortie

    def bprop(self, X, y):
        X = np.array([[float(x)] for x in X])
        self._gradoa = np.subtract(self._os, utils.onehot(self._m,y))
        self._gradb2 = self._gradoa
        self._gradw2 = np.add(np.dot(self._gradoa, np.transpose(self._hs)), 2 * self.wd * self._w2)
        self._gradhs = np.dot(np.transpose(self._w2), self._gradoa)
        self._gradha = self._gradhs * np.where(self._ha > 0, 1, 0)
        self._gradb1 = np.array(self._gradha)
        self._gradw1 = np.add(np.dot(self._gradha,np.transpose(X)), 2 * self.wd * self._w1)
        self._gradx = np.dot(np.transpose(self._w1), self._gradha)

    def calculateLoss(self, y):
        return -(np.log(self._os[y][0]))

    def predict(self, x):
        self.fprop(x)

        klass = np.argmax(np.transpose(self._os)[0])

        return klass

    def computePredictions(self, X):
        predictions = []

        for x in X:
            self.fprop(x)
            predictions.append(np.argmax(np.transpose(self._os)[0]))

        return predictions

    def _nextBatchIndex(self, X,y, batchNbr):
        correctedBatchNbr = batchNbr % int(float(len(X))/self._K)
        size = len(X)
        born1 = int(correctedBatchNbr * self._K + 0.001)
        born2 = int((correctedBatchNbr+1) * self._K + 0.001)
        if born2 > size:
            born1 = 0
            born2 = self._K
        return X[born1:born2],y[born1:born2]

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

            xbatch, ybatch = self._nextBatchIndex(X,y, batchNbr)

            nbrAverage = 0
            w1update = 0
            w2update = 0
            b1update = 0
            b2update = 0
            for elem in range(len(xbatch)):

                self.fprop(xbatch[elem])
                self.bprop(xbatch[elem], ybatch[elem])

                nbrAverage += 1
                w1update = np.add(w1update, self._gradw1)
                w2update = np.add(w2update, self._gradw2)
                b1update = np.add(b1update, self._gradb1)
                b2update = np.add(b2update, self._gradb2)

            if nbrAverage > 0:
                self._w1 = np.add(self._w1, - np.multiply(eta, np.multiply(w1update, 1./nbrAverage)))
                self._w2 = np.add(self._w2, - np.multiply(eta, np.multiply(w2update, 1./nbrAverage)))
                self._b1 = np.add(self._b1, - np.multiply(eta, np.multiply(b1update, 1./nbrAverage)))
                self._b2 = np.add(self._b2, -np.multiply(eta, np.multiply(b2update, 1./nbrAverage)))

            #Pour #9-10
            if iter % 10 == 10:
                self._calculateEfficiency()
                self._calculateAverageCosts()

            if len(self.epochData) > 0:
                print self.epochData[len(self.epochData)-1]

    def _calculateEfficiency(self):
        if self.Xtrain is None:
            pass
        else:
            predTrain = self.computePredictions(self.Xtrain)
            predValid = self.computePredictions(self.Xvalid)
            predTest = self.computePredictions(self.Xtest)

            self.trainError.append(100-utils.calculatePredictionsEfficiency(predTrain, self.ytrain))
            self.epochData.append("{:.3f}".format(self.trainError[len(self.trainError)-1]))

            self.validError.append(100-utils.calculatePredictionsEfficiency(predValid, self.yvalid))
            self.epochData[len(self.epochData)-1] += ";"+"{:.3f}".format(self.validError[len(self.validError)-1])

            self.testError.append(100-utils.calculatePredictionsEfficiency(predTest, self.ytest))
            self.epochData[len(self.epochData)-1] += ";"+"{:.3f}".format(self.testError[len(self.testError)-1])


    def _calculateAverageCosts(self):
        if self.Xtrain is None or self.Xvalid is None or self.Xtest is None:
            pass
        else:
            #Calcule coût moyen sur ensemble d'entrainement, de test et de validation
            sumLTrain = 0
            for i in range(len(self.Xtrain)):
                self.fprop(self.Xtrain[i])
                sumLTrain += self.calculateLoss(self.ytrain[i])
            self.trainSumL.append(sumLTrain/len(self.Xtrain))
            self.epochData[len(self.epochData)-1] += ";"+"{:.3f}".format(self.trainSumL[len(self.trainSumL)-1])

            sumLValid = 0
            for i in range(len(self.Xvalid)):
                self.fprop(self.Xvalid[i])
                sumLValid += self.calculateLoss(self.yvalid[i])
            self.validSumL.append(sumLValid/len(self.yvalid))
            self.epochData[len(self.epochData)-1] += ";"+"{:.3f}".format(self.validSumL[len(self.validSumL)-1])

            sumLTest = 0
            for i in range(len(self.Xtest)):
                self.fprop(self.Xtest[i])
                sumLTest += self.calculateLoss(self.ytest[i])
            self.testSumL.append(sumLTest/len(self.ytest))
            self.epochData[len(self.epochData)-1] += ";"+"{:.3f}".format(self.testSumL[len(self.testSumL)-1])

if __name__ == '__main__':
    self = NeuralNetwork(4, 6, 3)
    X = [30, 20, 45, 50]
    y = 1  # imaginons que c'est un point de la classe 3
    self.fprop(X)
    self.bprop(X, y)
