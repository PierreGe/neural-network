
import random
import numpy

def sigmoid(x): # TODO
    """Numerically-stable sigmoid function.
    Source : https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/"""
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = exp(x)
        return z / (1 + z)


def softmax(q):  # TODO PAS OK
    """ multinomial logistic sigmoid
    Source : https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/"""
    max_q = max(0.0, numpy.max(q))
    rebased_q = q - max_q
    return numpy.exp(rebased_q - numpy.logaddexp(-max_q, numpy.logaddexp.reduce(rebased_q)))



def uniform(nc):
    borne = 1. / numpy.sqrt(nc)
    res = numpy.random.uniform(-borne, borne)
    return res


def randomArray(nc,ligne, col):
    return [[uniform(nc) for i in range(col)] for j in range(ligne)]


def relu(M):
    try: # matrix
        return numpy.array([numpy.array([max(0,val) for val in j]) for j in M])
    except: # vector
        return numpy.array([max(0,val) for val in M])


def onehot(m,y): # attention le y indice a partir de 1
    res = numpy.zeros(m)
    res[y-1] = 1
    return res