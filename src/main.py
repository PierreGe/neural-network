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
    sample_wd = [0, 0.00001, 0.0001, 0.001,
                 0.01]  # todo Valider terme de regularisation dans NeuralNetwork. Lorsque != 0, validations #1,2,3,4 ne sont plus bon...

    default_maxIter = 5
    sample_maxIter = [1, 2, 5, 10, 20, 100, 200]

    for h in sample_h:
        neuralNetwork = NeuralNetwork(len(Xtrain[0]), h, utils.getClassCount(ytrain), wd=default_wd)
        neuralNetwork.train(Xtrain, ytrain, default_maxIter)
        predictions = neuralNetwork.computePredictions(Xvalid)

        trainEfficiency = utils.calculatePredictionsEfficiency(predictions, yvalid)
        title = "h: " + str(h) + " / wd: " + str(default_wd) + " / " + str(
                default_maxIter) + " epoques" + " / Train Err: " + str(100 - trainEfficiency) + "%"
        name = "regions_decision" + str(h) + "_" + str(default_wd) + "_" + str(default_maxIter)
        utils.plotRegionsDescision(Xtrain, ytrain, neuralNetwork, title, name)

    for wd in sample_wd:
        neuralNetwork = NeuralNetwork(len(Xtrain[0]), default_h, utils.getClassCount(ytrain), wd=wd)
        neuralNetwork.train(Xtrain, ytrain, default_maxIter)
        predictions = neuralNetwork.computePredictions(Xvalid)

        trainEfficiency = utils.calculatePredictionsEfficiency(predictions, yvalid)
        title = "h: " + str(default_h) + " / wd: " + str(wd) + " / " + str(
                default_maxIter) + " epoques" + " / Train Err: " + str(100 - trainEfficiency) + "%"
        name = "regions_decision" + str(default_h) + "_" + str(wd) + "_" + str(default_maxIter)
        utils.plotRegionsDescision(Xtrain, ytrain, neuralNetwork, title, name)

    for maxIter in sample_maxIter:
        neuralNetwork = NeuralNetwork(len(Xtrain[0]), default_h, utils.getClassCount(ytrain), wd=default_wd)
        neuralNetwork.train(Xtrain, ytrain, maxIter)
        predictions = neuralNetwork.computePredictions(Xvalid)

        trainEfficiency = utils.calculatePredictionsEfficiency(predictions, yvalid)
        title = "h: " + str(default_h) + " / wd: " + str(default_wd) + " / " + str(
                maxIter) + " epoques" + " / Train Err: " + str(100 - trainEfficiency) + "%"
        name = "regions_decision" + str(default_h) + "_" + str(default_wd) + "_" + str(maxIter)
        utils.plotRegionsDescision(Xtrain, ytrain, neuralNetwork, title, name)

def exo67():
    print("\n\n>>EXERCICE 6 Calcul matriciel")
    nn = NeuralNetworkEfficient(4, 6, 3)
    X = [[30, 20, 45, 50], [20, 10, 35, 40]]
    y = [1,0]  # imaginons que c'est un point de la classe 3
    nn.fprop(X)
    #nn.bprop(X, y)

def exo8():
    print("\n\n>>EXERCICE 8 MNIST")
    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMNISTfile()
    default_h = 30
    default_wd = 0.0001
    maxIter = 30
    t1 = datetime.now()
    neuralNetwork = NeuralNetwork(len(Xtrain[0]), default_h, utils.getClassCount(ytrain),K=100, wd=default_wd)
    neuralNetwork.train(Xtrain, ytrain, maxIter)
    predictions = neuralNetwork.computePredictions(Xvalid)
    trainEfficiency = utils.calculatePredictionsEfficiency(predictions, yvalid)
    t2 = datetime.now()
    delta = t2 - t1
    print("Train Err: " + str(100 - trainEfficiency))
    print("Cela a mis : " + str(delta.total_seconds()) + " secondes")


def main():
    #exo1234()
    #exo5()
    #exo67()
    exo8()

if __name__ == '__main__':
    main()
