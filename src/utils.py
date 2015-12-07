
import random
import numpy
import matplotlib.pyplot as plt

def softmax(vec):
    """ ok"""
    max_vec = max(0, numpy.max(vec))
    rebased_q = [q - max_vec for q in vec]
    norm = 1./numpy.sum(numpy.exp(rebased_q))
    return numpy.multiply(norm, numpy.exp(rebased_q))


def uniform(nc):
    borne = 1. / numpy.sqrt(nc)
    res = numpy.random.uniform(-borne, borne)
    return res


def randomArray(nc,ligne, col):
    return numpy.array([numpy.array([uniform(nc) for i in range(col)]) for j in range(ligne)])


def relu(M):
    try: # matrix
        return numpy.array([numpy.array([max(0,val) for val in j]) for j in M])
    except: # vector
        return numpy.array([max(0,val) for val in M])


def onehot(m,y): # attention le y indice a partir de 0
    res = numpy.zeros(m)
    res[y] = 1
    return res


def ratioGrad(vec1, vec2):
    res = []
    for i in range(len(vec1)):
        if vec2[i] != 0:
            res.append(vec1[i]/vec2[i])
        elif vec1[i]== vec2[i]:
            res.append(1)
        else:
            res.append(float('nan'))
    return res

def getClassCount(y):
    classes = []
    for i in y:
        if i not in classes:
            classes.append(i)

    return len(classes)

def plotRegionsDescision(X, y, neuralNetwork):
    X = numpy.array(X)

    minX1 = min(X[:, 0])
    maxX1 = max(X[:, 0])
    minX2 = min(X[:, 1])
    maxX2 = max(X[:, 1])

    grille = []
    step = 0.05
    i = minX1
    while i < maxX1:
        j = minX2
        while j < maxX2:
            grille.append([i, j])
            j += step
        i += step
    grille = numpy.array(grille)

    predictions = neuralNetwork.computePredictions(grille)
    plt.scatter(grille[:, 0], grille[:, 1], s=50, c=predictions, alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='v', s=100)

    #todo pas de set de validation en ce moment
    #plt.scatter(self.validationSet[:, 0], self.validationSet[:, 1], c=self.iris[self.trainSetSize:, -1], marker='s', s=100)

    plt.title("Regions de decision")
    plt.show()

    #fileTitle = 'bayes_parzen_'+str(sigma)+'.png'
    #plt.savefig(fileTitle)
    #print("[Created] file : " + fileTitle)
    plt.close()
