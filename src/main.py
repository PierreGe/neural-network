

from NeuralNetwork import NeuralNetwork

def main():
    lines = open("2moons.txt").readlines()
    X = []
    y = []
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