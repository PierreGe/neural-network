

from NeuralNetwork import NeuralNetwork
from gradientVerification import verifGradient1d,verifGradientKd

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



if __name__ == '__main__':
    main()