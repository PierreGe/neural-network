

from NeuralNetwork import NeuralNetwork
from gradientVerification import verifGradient1d

def main():

    print("\n\n>>EXERCICE 1 et 2")
    sigma = 1e-4
    neuralNetwork = NeuralNetwork(2,2,2)
    X = [0.4,0.7]
    y = 1 # imaginons que c'est un point de la classe
    print("Liste des ratio W1, b1, W2, b2")
    print(verifGradient1d(neuralNetwork, X, y))

    print("\n\n>>EXERCICE 3 et 4")

    lines = open("2moons.txt").readlines()
    X = []
    y = []
    K = 10
    for l in lines:
        x1,x2,klass = l.split()
        X.append([float(x1),float(x2)])
        y.append(int(klass))

    neuralNetwork = NeuralNetwork(2,4,2)
    for i in range(len(X)):
        neuralNetwork.fprop(X[i])
        neuralNetwork.bprop(X[i],y[i])



if __name__ == '__main__':
    main()