
import random
import numpy


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