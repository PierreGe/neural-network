# -*- coding: utf-8 -*-

from NeuralNetwork import NeuralNetwork
from gradientVerification import verifGradient1d,verifGradientKd
from trainNetwork import trainNetwork
from utils import plotRegionsDescision
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
    wd = 1  #todo Ajouter terme de régularisation dans le réseau de neuronne.
    h = 2
    maxIter = 5

    neuralNetwork = trainNetwork(X, y,  wd, h, maxIter)
    predictions = neuralNetwork.computePredictions(X)

    success = 0
    fail = 0
    for i in range(len(predictions)):
        if predictions[i] == y[i]:
            success += 1
        else:
            fail += 1

    print "Success: "+str(success)
    print "Error: "+str(fail)
    print "Ratio: "+str(100.0*success/len(y))

    plotRegionsDescision(X, y, neuralNetwork)


if __name__ == '__main__':
    main()