import numpy
import matplotlib.pyplot as plt
import gzip, pickle


def softmax(vec):
    """ ok"""
    if len(vec[0]) == 1:
        max_vec = max(0, numpy.max(vec))
        rebased_q = [q - max_vec for q in vec]
        norm = 1. / numpy.sum(numpy.exp(rebased_q))
        return numpy.multiply(norm, numpy.exp(rebased_q))
    else:
        matrix = numpy.transpose(vec)
        res = []
        for i in matrix:
            v = numpy.array([[j] for j in i])
            res.append(softmax(v))

        res2 = []
        for colonne in res:
            temp = []
            for element in colonne:
                temp.append(element[0])
            res2.append(temp)
        res2 = numpy.array(res2)
        res2 = numpy.transpose(res2)
        return res2


def uniform(nc):
    borne = 1. / numpy.sqrt(nc)
    res = numpy.random.uniform(-borne, borne)
    return res


def randomArray(nc, ligne, col):
    return numpy.array([numpy.array([uniform(nc) for i in range(col)]) for j in range(ligne)])


def relu(M):
    try:  # matrix
        return numpy.array([numpy.array([max(0, val) for val in j]) for j in M])
    except:  # vector
        return numpy.array([max(0, val) for val in M])


def onehot(m, Y):  # attention le y indice a partir de 0
    try:
        res = numpy.zeros((m, len(Y)))
        for i,y in enumerate(Y):
            res[y][i] = 1
    except:
        res = numpy.zeros(m)
        res[Y] = 1
        res = [[i] for i in res] # vecteur colonne
    return res


def ratioGrad(vec1, vec2):
    res = []
    for i in range(len(vec1)):
        if vec2[i] != 0:
            res.append(vec1[i] / vec2[i])
        elif vec1[i] == vec2[i]:
            res.append(1)
        else:
            res.append(float('nan'))
    return res


def calculatePredictionsEfficiency(preds, trueVals):
    success = 0
    fail = 0
    for i in range(len(preds)):
        if preds[i] == trueVals[i]:
            success += 1
        else:
            fail += 1

    return 100.0 * success / len(trueVals)


def readMoonFile(validationSizePercent = 15, testSizePercent= 15):
    lines = open("data/2moons.txt").readlines()
    X = []
    y = []
    for l in lines:
        x1, x2, klass = l.split()
        X.append([float(x1), float(x2)])
        y.append(int(klass))
    size = len(X)
    born1 = int(size * ((100. - validationSizePercent - testSizePercent)/100))
    born2 = int(size * ((100. - testSizePercent)/100))
    Xtrain = X[:born1]  #: matrice de train data
    ytrain = y[:born1]  #: vecteur des train labels

    Xvalid = X[born1:born2]  #: matrice de valid data
    yvalid = y[born1:born2]  #: vecteur des valid labels

    Xtest = X[born2:]  #: matrice de test data
    ytest = y[born2:]  #: vecteur des test labels
    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def readMNISTfile():
    f = gzip.open('data/mnist.pkl.gz')
    data = pickle.load(f)

    Xtrain = data[0][0]  #: matrice de train data
    ytrain = data[0][1]  #: vecteur des train labels

    Xvalid = data[1][0]  #: matrice de valid data
    yvalid = data[1][1]  #: vecteur des valid labels

    Xtest = data[2][0]  #: matrice de test data
    ytest = data[2][1]  #: vecteur des test labels
    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def getClassCount(y):
    classes = []
    for i in y:
        if i not in classes:
            classes.append(i)

    return len(classes)

def plotRegionsDescision(XTrain, yTrain, XValid, yValid, XTest, yTest, neuralNetwork, title, name, hparams):
    XTrain = numpy.array(XTrain)
    XValid = numpy.array(XValid)
    XTest = numpy.array(XTest)
    X = numpy.concatenate((XTrain, XValid, XTest))

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

    plt.scatter(XTrain[:, 0], XTrain[:, 1], c=yTrain, marker='o', s=100)
    plt.scatter(XValid[:, 0], XValid[:, 1], c=yValid, marker='s', s=100)
    plt.scatter(XTest[:, 0], XTest[:, 1], c=yTest, marker='*', s=100)

    plt.title("Regions de decision "+hparams+"\n" + title)
    # plt.show()

    plt.savefig(name + ".png")
    print("[Created] file : " + name + ".png")
    plt.close()


def compareNN(nn1,nn2, K=1):

    if K>1:
        ub2 = numpy.sum(nn2._gradb2, axis=1) * K
        gradb2 = numpy.array([[i] for i in ub2])
    else:
        gradb2= nn2._gradb2

    if numpy.array_equal(nn1._gradb2, gradb2):
        print("Test Ok : gradient b2")
    else:
        print("FAIL : gradient b2")

    if numpy.array_equal(nn1._gradw2,nn2._gradw2):
        print("Test Ok : gradient w2")
    else:
        print("FAIL : gradient w2")

    if K>1:
        ub1 = numpy.sum(nn2._gradb1, axis=1) * K
        gradb1 = numpy.array([[i] for i in ub1])
    else:
        gradb1= nn2._gradb1
    if numpy.array_equal(nn1._gradb1,gradb1):
        print("Test Ok : gradient b1")
    else:
        print("FAIL : gradient b1")

    if numpy.array_equal(nn1._gradw1,nn2._gradw1):
        print("Test Ok : gradient w1")
    else:
        print("FAIL : gradient w1")
