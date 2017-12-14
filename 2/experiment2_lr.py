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
    temp = 0.5*rp*m.T.dot(m)
    y_predict = sigmod(x.dot(m))
    #tt = -y*y_predict.T
    #temp+=float(1/N)*np.sum(np.log(1+np.exp(tt)))
    for i in range(N):
        temp+=float(1/N)*np.log(1+np.exp(-y[i]*y_predict[i]))
    #totalError=(y-x.todense().dot(m)-b)
    #totalError=totalError.T.dot(totalError)
    #totalError=np.sum(totalError)

    return temp/float(len(y))


def optimizer_NAG(x,y,x_test,y_test,starting_m,rp,num_iter):
    #b=starting_b
    m=starting_m
    #error_list1 = np.arange(num_iter)
    error_list2 = np.arange(num_iter)
    #mr = compute_gradient_NAG(starting_m, x_test, y_test, rp)

    for i in range(num_iter):
        m = compute_gradient_NAG(m, x, y, rp)
        #if(i%10==0):
        temp = compute_error(m,x,y,rp)
        print ('iter {0}:error={1}'.format(i,temp))
        #temp = compute_error(b, m, x, y)
        #error_list1[i] = temp
        temp2 = compute_error(m,x_test,y_test,rp)
        error_list2[i] = temp2
    return[m,error_list2]


def compute_gradient_NAG(m_current,x,y,rp,size=200):

    #b_gradient = 0
    m_gradient = 0

    n=len(y)
    N = float(n)
    y.shape=(n,1)
    m_current.shape=(124,1)
    #b_gradient = -(2/N)*(y-x.todense().dot(m_current)-b_current)
    #b_gradient = b_gradient.mean()
    #m_gradient = -(2/N)*x.todense().T*(((y-x.todense().dot(m_current)-b_current)))
    #m_gradient = m_gradient.mean(axis=0
    temp = random.randint(0, n - 200)
    for i in range(size):
        m_gradient+=(sigmod(x[temp+i,:].dot(m_current))-y[temp+i])*(x[temp+i,:].dot(m_current))
    m_gradient=m_gradient/N

    #new_b = b_current-(rp*b_gradient)
    new_m = m_current-(rp*m_gradient)
    return new_m


def plot_data(error,error2):
    n = range(len(error))
    pylab.plot(n,error,label='train')
    pylab.plot(n,error2,label='test')
    plt.legend()
    pylab.show()


def Linear_regression():
    x_train,y_train = load_svmlight_file('a9a.txt')
    x_test,y_test = load_svmlight_file("a9a.t")
    y_t = len(y_train)
    y_e = len(y_test)
    y_train.reshape((y_t,1))
    y_test.reshape((y_e,1))
    rp = 0.01
    train = np.ones(y_t)
    test = np.ones(y_e)
    testt = np.zeros(y_e)
    x_train=x_train.todense()
    x_test=x_test.todense()
    x_train = np.c_[x_train,train]
    x_test = np.c_[x_test,testt]
    x_test = np.c_[x_test,test]
    #init_b = 0.0
    init_m = np.zeros(124)
    num_iter = 10
    error = 0
    error2 = 0

    [m,error2] = optimizer_NAG(x_train,y_train,x_test,y_test,init_m,rp,num_iter)

    print ('final formula parmaters:\n b = in m final number\n m={1} \n'.format(num_iter,m))

    plot_data(error,error2)


if __name__ =='__main__':
    Linear_regression()

