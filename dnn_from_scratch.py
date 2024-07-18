import numpy as np
import matplotlib.pyplot as plt
from planar_utils import load_planar_dataset,plot_decision_boundary

#Logaritmalara 0 girdisi vermemek için değerler kırpılır.
EPSILON = 10e-10

X,Y = load_planar_dataset()


plt.scatter(X[0,:],X[1,:],c=Y,cmap=plt.cm.Spectral)
plt.show()

#Örnek sayısı
m = X.shape[1]

def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}

    L = len(layer_dims)

    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))
        print(l)

    return parameters

def forward_linear(A_prev,W,b):

    Z = np.dot(W,A_prev) + b

    linear_cache = (A_prev,W,b)

    return Z, linear_cache

def sigmoid(z):
    a =  1/(1+np.exp(-z))
    activation_cache = z
    return a, activation_cache

def relu(z):
    a = np.maximum(0,z)
    activation_cache = z
    return a, activation_cache

def forward_activation(A_prev,W,b, activation):
    Z, linear_cache = forward_linear(A_prev,W,b)

    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)

    cache = (linear_cache,activation_cache)
    return A, cache


def forward_propagation(X,parameters):

    caches = []
    A = X
    L = len(parameters)//2

    for i in range(1,L):

        A_prev = A

        W = parameters['W'+str(i)]
        b = parameters['b'+str(i)]
        A, cache = forward_activation(A_prev,W,b,activation='relu')

        caches.append(cache)



    AL, cache = forward_activation(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches

def compute_cost(AL,Y):

    AL = np.clip(AL,EPSILON,1-EPSILON)
    cost = (-1 / m) * (np.dot(Y, np.log(AL).T) + np.dot((1 - Y), np.log(1 - AL).T))

    cost = np.squeeze(cost)

    return cost

def linear_backward(dZ, cache):

    A_prev, W, b = cache


    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)


    return dA_prev, dW, db

def relu_derivative(Z):
    der = np.where(Z>0,1,0)
    return der

def sigmoid_derivative(Z):
    s,_ = sigmoid(Z)
    der = s*(1-s)
    return der

def linear_activation_backward(dA, cache,activation):

    linear_cache, activation_cache = cache

    if activation=='sigmoid':
        dZ = dA* sigmoid_derivative(activation_cache)#activation cache activation fonka sokulan zdir.
    elif activation=='relu':
        dZ = dA * relu_derivative(activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def model_backward(AL,Y,caches):
    grads = {}

    AL = np.clip(AL,EPSILON,1-EPSILON)# logaritmanın içi 0 olmasına karşın bir önlem

    #output layerınının türevi alınarak başlanır.
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      "sigmoid")

    for l in reversed(range(L - 1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    #print("len"+str(len(grads)))
    return grads

def update_parameters(parameters,grads,learning_rate):


    for l in range(1,L+1):# 1.layer parametrelerinden başlar ve L. parametrelere kadar hepsinini günceller.
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters

def dnn_model(X,Y,layer_dims,learning_rate,epoch):

    parameters = initialize_parameters(layer_dims)

    A = X

    for i in range(epoch):
        AL, cache = forward_propagation(A,parameters)
        #print("al:"+str(AL))

        cost = compute_cost(AL,Y)
        if i % 1000 == 0:
            print("epoch:",i,"cost:",str(cost))

        grads = model_backward(AL,Y,cache)

        parameters = update_parameters(parameters,grads,learning_rate)

    return parameters

def predict(X,parameters):
    AL, caches = forward_propagation(X, parameters)
    predictions = (AL > 0.5)
    return predictions


layer_dims = [2,20,10,10,1]
L = len(layer_dims) - 1 #input layerı saymıyoruz
parameters = dnn_model(X,Y,layer_dims,0.1,150000)



plot_decision_boundary(lambda x : predict(x.T,parameters),X,Y)
plt.title("Decision Boundary")
plt.show()
