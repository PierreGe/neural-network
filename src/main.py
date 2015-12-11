# -*- coding: utf-8 -*-

from NeuralNetwork import NeuralNetwork
from gradientVerification import verifGradient1d,verifGradientKd
from trainNetwork import trainNetwork
from utils import plotRegionsDescision, calculatePredictionsEfficiency
def readMoonFile():
    lines = open("2moons.txt").readlines()
    X = []
    y = []
    K = 10
    for l in lines:
        x1,x2,klass = l.split()
        X.append([float(x1),float(x2)])
        y.append(int(klass))
    return X,y

def main():

    print("\n\n>>EXERCICE 1 et 2")
    sigma = 1e-4
    neuralNetwork = NeuralNetwork(2,2,2)
    X = [0.4,0.7]
    y = 1 # imaginons que c'est un point de la classe
    print("Liste des ratio W1, b1, W2, b2")
    print(verifGradient1d(neuralNetwork, X, y))

    print("\n\n>>EXERCICE 3 et 4")
    neuralNetwork = NeuralNetwork(2,2,2)
    X,y = readMoonFile()
    K = 10
    X = X[:K]
    y = y[:K]

    print("Liste des ratio W1, b1, W2, b2")
    print(verifGradientKd(neuralNetwork, X, y))


    print("\n\n>>EXERCICE 5 Entrainement du reseau de neuronne + Variation des hyper-parametres")
    X, y = readMoonFile()

    #default_h = 5
    sample_h = [2,5,10,50,100,200,500,1000]

    #wd = 0.1
    sample_wd = [0,0.00001,0.0001,0.001,0.01]  #todo Valider terme de regularisation dans NeuralNetwork. Lorsque != 0, validations #1,2,3,4 ne sont plus bon...

    default_maxIter = 5
    sample_maxIter = [1,2,5,10,20,100]

    i = 0
    while True
        for h in sample_h:
            for wd in sample_wd:
                for maxIter in sample_maxIter:
                    i+=1
                    neuralNetwork = trainNetwork(X, y,  wd, h, maxIter)
                    predictions = neuralNetwork.computePredictions(X)

                    trainEfficiency = calculatePredictionsEfficiency(predictions, y)
                    if trainEfficiency > 80:
                        print("h: "+str(h)+" / wd: "+str(wd)+" / "+str(maxIter)+" epoques"+" / Train Err: "+str(100-trainEfficiency)+"%")
                    if trainEfficiency > 99:
                        title = "h: "+str(h)+" / wd: "+str(wd)+" / "+str(maxIter)+" epoques"+" / Train Err: "+str(100-trainEfficiency)+"%"
                        name = str(100-trainEfficiency) + "_regions_decision"+str(h)+"_"+str(wd)+"_"+str(maxIter) + "_" + str(i)
                        plotRegionsDescision(X, y, neuralNetwork, title, name)

if __name__ == '__main__':
    main()