import numpy as np
import sklearn as sk
import pylab
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import math
import random


def sigmod(x):
    temp = 1/(1+np.exp(-x))
    return temp


def compute_error(m,x,y,rp):
    totalError = 0
    N = len(y)
    x.shape=(N,124)
    temp = 0
    y_predict = sigmod(x.dot(m))
    for i in range(N):
        temp+=float(1/N)*np.log(1+np.exp(-y[i]*y_predict[i]))

    return temp.mean()


def optimizer_NAG(x,y,x_test,y_test,starting_m,rp,num_iter):
    m=starting_m
    m.shape=(124,1)
    error_list2 = []
    v=np.zeros((124,1))

    for i in range(num_iter):
        m,v = compute_gradient_NAG(v,m, x, y, rp)
        temp = compute_error(m,x,y,rp)
        print ('iter {0}:error={1}'.format(i,temp))
        temp2 = compute_error(m,x_test,y_test,rp)
        error_list2.append(temp2)
    return[m,error_list2]


def compute_gradient_NAG(v,m_current,x,y,rp,size=200):

    m_gradient = 0
    gama = 0.9
    v.shape=(124,1)

    n=len(y)
    N = float(n)
    y.shape=(n,1)
    m_current.shape=(124,1)
    temp = random.randint(0, n - 200)
    for i in range(size):
        tt=x[temp+i].dot(m_current)
        m_gradient+=(sigmod(tt)-y[temp+i])*x[temp+i]
    m_gradient=m_gradient/N
    v = gama*v+rp*m_gradient.T

    new_m = m_current-v
    return new_m,v



def optimizer_RMSProp(x,y,x_test,y_test,starting_m,rp,num_iter):
    m=starting_m
    m.shape=(124,1)
    error_list2 = []
    v=np.zeros((124,1))

    for i in range(num_iter):
        m,v = compute_gradient_RMSProp(v,m, x, y, rp)
        temp = compute_error(m,x,y,rp)
        print ('iter {0}:error={1}'.format(i,temp))
        temp2 = compute_error(m,x_test,y_test,rp)
        error_list2.append(temp2)
    return[m,error_list2]


def compute_gradient_RMSProp(v,m_current,x,y,rp,size=200):
    m_gradient = 0
    gama = 0.9
    v.shape=(124,1)

    n=len(y)
    N = float(n)
    y.shape=(n,1)
    m_current.shape=(124,1)
    temp = random.randint(0, n - 200)
    for i in range(size):
        tt=x[temp+i].dot(m_current)
        m_gradient+=(sigmod(tt)-y[temp+i])*x[temp+i]
    m_gradient=(m_gradient/N)
    v = gama*v+(1-gama)*(m_gradient*m_gradient.T)

    temp = rp/np.sqrt(v+np.exp(-8))
    new_m = m_current-np.multiply(temp,m_gradient.T)
    return new_m,v



def optimizer_AdaDelta(x,y,x_test,y_test,starting_m,rp,num_iter):
    m=starting_m
    m.shape=(124,1)
    error_list2 = []
    v=np.zeros((124,1))
    delta = np.zeros((124,1))

    for i in range(num_iter):
        m,v,delta = compute_gradient_AdaDelta(delta,v,m, x, y, rp)
        temp = compute_error(m,x,y,rp)
        print ('iter {0}:error={1}'.format(i,temp))
        temp2 = compute_error(m,x_test,y_test,rp)
        error_list2.append(temp2)
    return[m,error_list2]


def compute_gradient_AdaDelta(delta,v,m_current,x,y,rp,size=200):

    m_gradient = 0
    gama = 0.95
    v.shape=(124,1)

    n=len(y)
    N = float(n)
    y.shape=(n,1)
    m_current.shape=(124,1)
    temp = random.randint(0, n - 200)
    for i in range(size):
        tt=x[temp+i].dot(m_current)
        m_gradient+=(sigmod(tt)-y[temp+i])*x[temp+i]
    m_gradient=m_gradient/N
    v = gama*v+(1-gama)*np.multiply(m_gradient.T,m_gradient.T)
    temp = np.sqrt(delta+np.exp(-8))/np.sqrt(v+np.exp(-8))
    temp2 = -np.multiply(temp,m_gradient.T)

    new_m = m_current+temp2
    delta=gama*delta+(1-gama)*np.multiply(temp2,temp2)
    return new_m,v,delta



def optimizer_Adam(x,y,x_test,y_test,starting_m,rp,num_iter):
    m=starting_m
    m.shape=(124,1)
    error_list2 = []
    v=np.zeros((124,1))
    g = np.zeros((124,1))

    for i in range(num_iter):
        m,v,g = compute_gradient_Adam(g,v,m, x, y, rp)
        temp = compute_error(m,x,y,rp)
        print ('iter {0}:error={1}'.format(i,temp))
        temp2 = compute_error(m,x_test,y_test,rp)
        error_list2.append(temp2)
    return[m,error_list2]


def compute_gradient_Adam(g,v,m_current,x,y,rp,size=200):

    m_gradient = 0
    gama = 0.999
    beta = 0.9
    v.shape=(124,1)

    n=len(y)
    N = float(n)
    y.shape=(n,1)
    m_current.shape=(124,1)
    temp = random.randint(0, n - 200)
    for i in range(size):
        tt=x[temp+i].dot(m_current)
        m_gradient+=(sigmod(tt)-y[temp+i])*x[temp+i]
    m_gradient=m_gradient/N
    v = beta*v+(1-beta)*m_gradient.T
    g = gama*g+(1-gama)*np.multiply(g,g)
    alpha = rp*math.sqrt(1-gama)/(1-beta)
    new_m = m_current-alpha*v/np.sqrt(g+np.exp(-8))
    return new_m,v,g


def plot_data(error1,error2,error3,error4):
    n = range(len(error2))
    pylab.plot(n,error1,label='NAG')
    pylab.plot(n, error2, label='RMSProp')
    pylab.plot(n, error3, label='AdaDelta')
    pylab.plot(n, error4, label='Adam')
    plt.legend()
    pylab.show()


def Linear_regression():
    x_train,y_train = load_svmlight_file('a9a.txt')
    x_test,y_test = load_svmlight_file("a9a.t")
    y_t = len(y_train)
    y_e = len(y_test)
    y_train.reshape((y_t,1))
    y_test.reshape((y_e,1))
    rp = 0.5
    train = np.ones(y_t)
    test = np.ones(y_e)
    testt = np.zeros(y_e)
    x_train=x_train.todense()
    x_test=x_test.todense()
    x_train = np.c_[x_train,train]
    x_test = np.c_[x_test,testt]
    x_test = np.c_[x_test,test]
    init_m = np.zeros(124)
    num_iter = 100
    error1 = 0
    error2 = 0
    error3 = 0
    error4 = 0

    [m1,error1] = optimizer_NAG(x_train,y_train,x_test,y_test,init_m,rp,num_iter)
    [m2,error2] = optimizer_RMSProp(x_train,y_train,x_test,y_test,init_m,rp,num_iter)
    [m3,error3] = optimizer_AdaDelta(x_train,y_train,x_test,y_test,init_m,rp,num_iter)
    [m4,error4] = optimizer_Adam(x_train,y_train,x_test,y_test,init_m,rp,num_iter)

    print ('final formula parmaters:\n b = in m final number\n m of NAG={1} \n m of RMSProp = {2}\n /'
           'm of AdaDelta = {3}\n m of Adam = {4}'.format(num_iter,m1,m2,m3,m4))

    plot_data(error1,error2,error3,error4)


if __name__ =='__main__':
    Linear_regression()

