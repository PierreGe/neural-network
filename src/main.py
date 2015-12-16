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

def trainAndPrint(Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest, h, wd, maxIter):
    neuralNetwork = NeuralNetwork(len(Xtrain[0]), h, utils.getClassCount(ytrain), K=100, wd=wd)
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
    print("Test efficiency : " + str(testEfficiency))
    utils.plotRegionsDescision(Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest, neuralNetwork, title, name,hparams)

def exo5():

    print("\n\n>>EXERCICE 5 Entrainement du reseau de neuronne + Variation des hyper-parametres")
    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMoonFile()

    sample_h = [2, 20, 100]
    sample_wd = [0, 0.0001, 0.01]
    sample_maxIter = [2, 50, 100, 200]

    for h in sample_h:
        for wd in sample_wd:
            for maxIter in sample_maxIter:
                trainAndPrint(Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest, h, wd, maxIter)


def exo67():
    print("\n\n>>EXERCICE 6 et 7 : Calcul matriciel")
    print(" --- K=1 ---")
    #Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMoonFile()
    Xtrain = [[30, 20, 40, 50], [25, 15, 35, 45]]
    ytrain = [0,0]
    default_h = 2
    nn = NeuralNetwork(len(Xtrain[0]), default_h, utils.getClassCount(ytrain), K=1, wd=0)
    nne = NeuralNetworkEfficient(len(Xtrain[0]), default_h, utils.getClassCount(ytrain), K=1, wd=0)
    nne._w1 = nn._w1 # trick pour que l'aleatoire soit egale
    nne._w2 = nn._w2
    nn.train(Xtrain,ytrain,1)
    nne.train(Xtrain,ytrain,1)
    utils.compareNN(nn,nne)
    print(" --- K=10 ---")
    Xtrain = [[30, 20, 40, 50], [25, 15, 35, 45],[30, 76, 45, 44],[89, 27, 42, 52],[30, 24, 44, 53],[89, 25, 45, 50],[30, 20, 40, 50],[30, 65, 47, 50],[30, 34, 40, 50],[39, 20, 29, 58]]
    ytrain = [0,0,0,0,0,0,0,0,0,0]
    default_h = 2
    nn = NeuralNetwork(len(Xtrain[0]), default_h, utils.getClassCount(ytrain), K=10, wd=0)
    nne = NeuralNetworkEfficient(len(Xtrain[0]), default_h, utils.getClassCount(ytrain), K=10, wd=0)
    nne._w1 = nn._w1 # trick pour que l'aleatoire soit egale
    nne._w2 = nn._w2
    nn.train(Xtrain,ytrain,1)
    nne.train(Xtrain,ytrain,1)
    utils.compareNN(nn,nne,10)

    #print(nn.computePredictions(X))
    #print(nne.computePredictions(X))


def exo8():
    print("\n\n>>EXERCICE 8 MNIST")
    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMNISTfile()
    default_h = 30
    maxIter = 1
    neuralNetwork = NeuralNetwork(len(Xtrain[0]), default_h, utils.getClassCount(ytrain),K=100)
    neuralNetworkEfficient = NeuralNetworkEfficient(len(Xtrain[0]), default_h, utils.getClassCount(ytrain),K=100)
    neuralNetworkEfficient._w1 = neuralNetwork._w1
    neuralNetworkEfficient._w2 = neuralNetwork._w2
    print("--- Reseau de depart ---")
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
    maxIter = 200
    K = 50

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMNISTfile()
    neuralNetwork = NeuralNetworkEfficient(len(Xtrain[0]), h, utils.getClassCount(ytrain), K, wd)
    neuralNetwork.setDataSets(Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest)
    neuralNetwork.train(Xtrain, ytrain, maxIter)

    epochsData = ""
    for d in neuralNetwork.epochData:
        epochsData += d+"\n"

    f = open('no9.txt', 'w')
    f.write(epochsData)
    f.close()

    x = range(1, maxIter+1, 10)
    title = "Taux d'erreur - "+str(maxIter)+" epoques"
    name = str(h) + "_" + str(wd)+ "_" + str(h)+ "_" + "taux_erreur"
    utils.plotCourbeApprentissage(neuralNetwork.trainError, neuralNetwork.validError, neuralNetwork.testError, x, title, name)

    title = "Cout moyen - "+str(maxIter)+" epoques"
    name = str(h) + "_" + str(wd)+ "_" + str(h)+ "_" +"cout_moyen"
    utils.plotCourbeApprentissage(neuralNetwork.trainSumL, neuralNetwork.validSumL, neuralNetwork.testSumL, x, title, name)


def test():

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = utils.readMoonFile()
    h = 200
    wd = 0.0001
    K = 200
    maxIter = 500

    # neuralNetwork = NeuralNetwork(len(Xtrain[0]), h, utils.getClassCount(ytrain), K=10, wd=wd)
    neuralNetworkEfficient = NeuralNetworkEfficient(len(Xtrain[0]), h, utils.getClassCount(ytrain), K, wd=wd)
    # neuralNetworkEfficient._w1 = neuralNetwork._w1
    # neuralNetworkEfficient._w2 = neuralNetwork._w2
    # neuralNetwork.train(Xtrain, ytrain, maxIter)
    # predTrain = neuralNetwork.computePredictions(Xtrain)
    # predValid = neuralNetwork.computePredictions(Xvalid)
    # predTest = neuralNetwork.computePredictions(Xtest)
    # trainEfficiency = utils.calculatePredictionsEfficiency(predTrain, ytrain)
    # validEfficiency = utils.calculatePredictionsEfficiency(predValid, yvalid)
    # testEfficiency = utils.calculatePredictionsEfficiency(predTest, ytest)
    # print( "Train Err: " + "{:.2f}".format(100 - trainEfficiency) + "%" \
    #         + " / Valid Err: " + "{:.2f}".format(100 - validEfficiency) + "%" \
    #         + " / Test Err: " + "{:.2f}".format(100 - testEfficiency) + "%")


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
    #np.random.seed(123)
    #exo1234()
    #exo67()
    #exo8
    exo5()
    #exo9_10()
    #test()

if __name__ == '__main__':
    main()

