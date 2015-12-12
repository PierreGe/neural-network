# -*- coding: utf-8 -*-

from NeuralNetwork import NeuralNetwork
from NeuralNetworkEfficient import NeuralNetworkEfficient
from gradientVerification import verifGradient1d, verifGradientKd
import utils
import numpy as np
from datetime import datetime

def exo1234():
    print("\n\n>>EXERCICE 1 et 2")
    sigma = 1e-4
    neuralNetwork = NeuralNetwork(2, 2, 2)
    Xtrain = [0.4, 0.7]
    ytrain = 1  # imaginons que c'est un point de la classe
    print("Liste des ratio W1, b1, W2, b2")
    res = verifGradient1d(neuralNetwork, Xtrain, ytrain)
    print(res)
    print(">Tout les ratio sont bien entre 0.99 et 1.01" if False not in [0.99 < i < 1.01 for i in (
        np.array(res)).flatten()] else "Echec de la verif..")

    print("\n\n>>EXERCICE 3 et 4")
    neuralNetwork = NeuralNetwork(2, 2, 2)
    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMoonFile()
    K = 10
    Xtrain = Xtrain[:K]
    ytrain = ytrain[:K]

    print("Liste des ratio W1, b1, W2, b2")
    res = verifGradientKd(neuralNetwork, Xtrain, ytrain)
    print(res)
    print(">Tout les ratio sont bien entre 0.99 et 1.01" if False not in [0.99 < i < 1.01 for i in (
        np.array(res)).flatten()] else "Echec de la verif..")

def exo5():

    print("\n\n>>EXERCICE 5 Entrainement du reseau de neuronne + Variation des hyper-parametres")
    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMoonFile()

    default_h = 5
    sample_h = [2, 5, 10, 50]

    default_wd = 0.1
    sample_wd = [0, 0.00001, 0.0001, 0.001, 0.01]

    default_maxIter = 5
    sample_maxIter = [1, 2, 5, 10, 20, 100, 200]

    for h in sample_h:
        trainAndPrint(Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest, h, default_wd, default_maxIter)

    for wd in sample_wd:
        trainAndPrint(Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest, default_h, wd, default_maxIter)

    for maxIter in sample_maxIter:
        trainAndPrint(Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest, default_h, default_wd, maxIter)

def trainAndPrint(Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest, h, wd, maxIter):
    neuralNetwork = NeuralNetwork(len(Xtrain[0]), h, utils.getClassCount(ytrain), K=50, wd=wd)
    neuralNetwork.train(Xtrain, ytrain, maxIter)
    predTrain = neuralNetwork.computePredictions(Xtrain)
    predValid = neuralNetwork.computePredictions(Xvalid)
    predTest = neuralNetwork.computePredictions(Xtest)
    trainEfficiency = utils.calculatePredictionsEfficiency(predTrain, ytrain)
    validEfficiency = utils.calculatePredictionsEfficiency(predValid, yvalid)
    testEfficiency = utils.calculatePredictionsEfficiency(predTest, ytest)
    hparams = "[h: " + str(h) + " / wd: " + str(wd) + " / " + str(maxIter) + " epoques]"
    title = "Train Err: " + "{:.2f}".format(100 - trainEfficiency) + "%" \
            + " / Valid Err: " + "{:.2f}".format(100 - validEfficiency) + "%" \
            + " / Test Err: " + "{:.2f}".format(100 - testEfficiency) + "%"
    name = "regions_decision" + str(h) + "_" + str(wd) + "_" + str(maxIter)
    utils.plotRegionsDescision(Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest, neuralNetwork, title, name,hparams)

def exo67():
    print("\n\n>>EXERCICE 6 Calcul matriciel")
    nn = NeuralNetwork(4, 6, 3)
    nne = NeuralNetworkEfficient(4, 6, 3)
    nne._w1 = nn._w1
    nne._w2 = nn._w2
    X = [[30, 20, 40, 50], [25, 15, 35, 45]]
    y = [1,0]  # imaginons que c'est un point de la classe 3
    nne.fprop(X)
    nn.fprop(X[0])
    #nn.bprop(X, y)

def exo8():
    print("\n\n>>EXERCICE 8 MNIST")
    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMNISTfile()
    default_h = 30
    default_wd = 0.0001
    maxIter = 300
    t1 = datetime.now()
    neuralNetwork = NeuralNetwork(len(Xtrain[0]), default_h, utils.getClassCount(ytrain),K=100, wd=default_wd)
    neuralNetwork.train(Xtrain, ytrain, maxIter)
    predictions = neuralNetwork.computePredictions(Xvalid)
    trainEfficiency = utils.calculatePredictionsEfficiency(predictions, yvalid)
    t2 = datetime.now()
    delta = t2 - t1
    print("Train Err: " + str(100 - trainEfficiency))
    print("Cela a mis : " + str(delta.total_seconds()) + " secondes")

def exo9_10():
    print("\n\n>>EXERCICE 9-10")
    h = 30
    wd = 0.0001
    maxIter = 10
    K = 10

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMoonFile(15,15)
    neuralNetwork = NeuralNetwork(len(Xtrain[0]), h, utils.getClassCount(ytrain), K, wd)
    neuralNetwork.setDataSets(Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest)
    neuralNetwork.train(Xtrain, ytrain, maxIter)

    print "SUCCESS"



def main():
    #exo1234()
    #exo5()
    #exo67()
    exo8()
    #exo9_10()

if __name__ == '__main__':
    main()

