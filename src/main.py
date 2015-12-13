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
    Xtrain = [0.7, 0.7]
    ytrain = 1  # imaginons que c'est un point de la classe
    print("Liste des ratio W1, b1, W2, b2")
    res = verifGradient1d(neuralNetwork, Xtrain, ytrain)
    print(res)
    print(">Tout les ratio sont bien entre 0.99 et 1.01" if False not in [0.99 < i < 1.01 for i in (
        np.array(res)).flatten()] else "Echec de la verif..")

    print("\n\n>>EXERCICE 3 et 4")
    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMoonFile()
    neuralNetwork = NeuralNetwork(len(Xtrain[0]), 5, utils.getClassCount(ytrain))
    K = 10
    X = Xtrain[9]
    y = ytrain[9]

    print("Liste des ratio W1, b1, W2, b2")
    res = verifGradient1d(neuralNetwork, X, y)
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
    print("\n\n>>EXERCICE 6 et 7 : Calcul matriciel")
    print(" --- K=1 ---")
    #Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMoonFile()
    Xtrain = [[30, 20, 40, 50], [25, 15, 35, 45]]
    ytrain = [0,0]
    default_h = 2
    nn = NeuralNetwork(len(Xtrain[0]), default_h, utils.getClassCount(ytrain), K=1)
    nne = NeuralNetworkEfficient(len(Xtrain[0]), default_h, utils.getClassCount(ytrain), K=1)
    nne._w1 = nn._w1 # trick pour que l'aleatoire soit egale
    nne._w2 = nn._w2
    nn.train(Xtrain,ytrain,1)
    nne.train(Xtrain,ytrain,1)
    utils.compareNN(nn,nne)
    print(" --- K=10 ---")
    Xtrain = [[30, 20, 40, 50], [25, 15, 35, 45],[30, 76, 45, 44],[89, 27, 42, 52],[30, 24, 44, 53],[89, 25, 45, 50],[30, 20, 40, 50],[30, 65, 47, 50],[30, 34, 40, 50],[39, 20, 29, 58]]
    ytrain = [0,0,0,0,0,0,0,0,0,0]
    default_h = 2
    nn = NeuralNetwork(len(Xtrain[0]), default_h, utils.getClassCount(ytrain), K=10)
    nne = NeuralNetworkEfficient(len(Xtrain[0]), default_h, utils.getClassCount(ytrain), K=10)
    nne._w1 = nn._w1 # trick pour que l'aleatoire soit egale
    nne._w2 = nn._w2
    nn.train(Xtrain,ytrain,1)
    nne.train(Xtrain,ytrain,1)
    utils.compareNN(nn,nne,10)

    #print(nn.computePredictions(X))
    #print(nne.computePredictions(X))


def exo8():
    print("\n\n>>EXERCICE 8 MNIST")
    print("--- Reseau de depart ---")
    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMNISTfile()
    default_h = 30
    default_wd = 0.0001
    maxIter = 1
    neuralNetwork = NeuralNetwork(len(Xtrain[0]), default_h, utils.getClassCount(ytrain),K=100, wd=default_wd)
    neuralNetworkEfficient = NeuralNetworkEfficient(len(Xtrain[0]), default_h, utils.getClassCount(ytrain),K=100, wd=default_wd)
    neuralNetworkEfficient._w1 = neuralNetwork._w1
    neuralNetworkEfficient._w2 = neuralNetwork._w2
    t1 = datetime.now()
    neuralNetwork.train(Xtrain, ytrain, maxIter)
    t2 = datetime.now()
    delta = t2 - t1
    print("Cela a mis : " + str(delta.total_seconds()) + " secondes")
    print("--- Reseau optimise ---")
    t1 = datetime.now()
    neuralNetworkEfficient.train(Xtrain, ytrain, maxIter)
    t2 = datetime.now()
    delta = t2 - t1
    print("Cela a mis : " + str(delta.total_seconds()) + " secondes")

def exo9_10():
    print("\n\n>>EXERCICE 9-10")
    print("Train Err;Valid Err;Test Err;Avg Cost Train;Avg Cost Valid;Avg Cost Test")

    h = 30
    wd = 0.0001
    maxIter = 15
    K = 50

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMNISTfile()    #todo replace by MNIST
    neuralNetwork = NeuralNetwork(len(Xtrain[0]), h, utils.getClassCount(ytrain), K, wd)
    neuralNetwork.setDataSets(Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest)
    neuralNetwork.train(Xtrain, ytrain, maxIter)

    epochsData = ""
    for d in neuralNetwork.epochData:
        epochsData += d+"\n"

    f = open('no9.txt', 'w')
    f.write(epochsData)
    f.close()

    x = range(1, maxIter+1)
    title = "Taux d'erreur - "+str(maxIter)+" epoques"
    name = "taux_erreur"
    utils.plotCourbeApprentissage(neuralNetwork.trainError, neuralNetwork.validError, neuralNetwork.testError, x, title, name)

    title = "Cout moyen - "+str(maxIter)+" epoques"
    name = "cout_moyen"
    utils.plotCourbeApprentissage(neuralNetwork.trainSumL, neuralNetwork.validSumL, neuralNetwork.testSumL, x, title, name)


def test():

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMoonFile()
    h = 100
    wd = 0.0001
    maxIter = 200

    neuralNetwork = NeuralNetwork(len(Xtrain[0]), h, utils.getClassCount(ytrain), K=10, wd=wd)
    neuralNetworkEfficient = NeuralNetworkEfficient(len(Xtrain[0]), h, utils.getClassCount(ytrain), K=10, wd=wd)
    neuralNetworkEfficient._w1 = neuralNetwork._w1
    neuralNetworkEfficient._w2 = neuralNetwork._w2
    neuralNetwork.train(Xtrain, ytrain, maxIter)
    predTrain = neuralNetwork.computePredictions(Xtrain)
    predValid = neuralNetwork.computePredictions(Xvalid)
    predTest = neuralNetwork.computePredictions(Xtest)
    trainEfficiency = utils.calculatePredictionsEfficiency(predTrain, ytrain)
    validEfficiency = utils.calculatePredictionsEfficiency(predValid, yvalid)
    testEfficiency = utils.calculatePredictionsEfficiency(predTest, ytest)
    print( "Train Err: " + "{:.2f}".format(100 - trainEfficiency) + "%" \
            + " / Valid Err: " + "{:.2f}".format(100 - validEfficiency) + "%" \
            + " / Test Err: " + "{:.2f}".format(100 - testEfficiency) + "%")


    neuralNetworkEfficient.train(Xtrain, ytrain, maxIter)
    predTrain = neuralNetworkEfficient.computePredictions(Xtrain)
    predValid = neuralNetworkEfficient.computePredictions(Xvalid)
    predTest = neuralNetworkEfficient.computePredictions(Xtest)
    trainEfficiency = utils.calculatePredictionsEfficiency(predTrain, ytrain)
    validEfficiency = utils.calculatePredictionsEfficiency(predValid, yvalid)
    testEfficiency = utils.calculatePredictionsEfficiency(predTest, ytest)
    print( "Train Err: " + "{:.2f}".format(100 - trainEfficiency) + "%" \
            + " / Valid Err: " + "{:.2f}".format(100 - validEfficiency) + "%" \
            + " / Test Err: " + "{:.2f}".format(100 - testEfficiency) + "%")



def main():
    np.random.seed(123)
    exo1234()
    #exo5()
    exo67()
    exo8()
    exo9_10()

if __name__ == '__main__':
    main()

