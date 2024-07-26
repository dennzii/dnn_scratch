import numpy as np
import matplotlib.pyplot as plt
from planar_utils import load_planar_dataset,plot_decision_boundary,load_extra_datasets
from load_images import load_images_from_disk

#Logaritmalara 0 girdisi vermemek için değerler kırpılır.
EPSILON = 10e-10
IMAGE_SHAPE = 64
INPUT_LAYER_SIZE = IMAGE_SHAPE * IMAGE_SHAPE

#X,Y = load_extra_datasets()

X,Y = load_images_from_disk(IMAGE_SHAPE)

print(X.shape)
print(Y.shape)

X = X.T
Y = Y.T

#plt.scatter(X[0,:],X[1,:],c=Y,cmap=plt.cm.Spectral)
#plt.show()

#Örnek sayısı
m = X.shape[1]

def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}

    L = len(layer_dims)

    for l in range(1,L):
        #HE initalization yapıoruz en sağda
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))

    return parameters

def forward_linear(A_prev,W,b):

    Z = np.dot(W,A_prev) + b

    linear_cache = (A_prev,W,b)

    return Z, linear_cache

#aktivasyona sokulan z'yi ve aktive olmuş değeri döndürür
def sigmoid(z):
    a =  1/(1+np.exp(-z))
    activation_cache = z
    return a, activation_cache

def relu(z):
    a = np.maximum(0,z)
    activation_cache = z
    return a, activation_cache


#forward prop'un aktivasyonlu hali. backpropagation için cache de döndürüyor
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
    #logaritmaların içi 0 olmaması için kırpıyoruz
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

#relunun türevi abi ne yazim anla işte
def relu_derivative(Z):
    der = np.where(Z>0,1,0)
    return der

#türv...
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


def learning_rate_decay(epoch,initial_learning_rate,decay_rate):
    return initial_learning_rate / (1 + decay_rate * epoch)

costs = []
def dnn_model(X,Y,layer_dims,learning_rate,decay_rate,epoch, batch_size = 512):

    parameters = initialize_parameters(layer_dims)

    n_batches = X.shape[1] // batch_size

    print("Layers:"+str(layer_dims) +" Epoch:"+str(epoch)+" Batch Count:"+str(n_batches) + " Batch Size"+ str(batch_size))
    print("Lambda:"+str(learning_rate) + " Decay Rate:"+str(decay_rate))


    for i in range(epoch):
        _X, _Y = np.copy(X),np.copy(Y)
        cost_total = 0
        #Learning_rate her epochta bir miktar küçülür. Daha küçük adımlar atarak global minimaya yakınlaştıkça
        #Osilasyon riskini bir miktar azaltır. Adam algoritması bu konuda daha iyi çalışır.
        learing_rate = learning_rate_decay(epoch,learning_rate,decay_rate)

        #mini batchler alınır.
        mini_batches = get_mini_batches(_X, _Y, batch_size)

        #Tüm veri seti mini_batchler halinde girdi olarak verilir.
        for mini_batch in mini_batches:
            #Bazı kodlar bi şeyler yapıolara
            batch_X, batch_Y = mini_batch

            AL, cache = forward_propagation(batch_X,parameters)

            cost_total += compute_cost(AL, batch_Y)

            grads = model_backward(AL,batch_Y,cache)

            parameters = update_parameters(parameters,grads,learing_rate)

        cost_total = cost_total / len(mini_batches)
        costs.append(cost_total)

        print("epoch:", i, "cost:", str(cost_total))

    return parameters


def get_mini_batches(X,Y,batch_size=512):
    mini_batches = []

    #Her iterasyonda karıştırılarak stokastik bir gradyan iniş elde edilir.
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    #Tam mini batchler hesaplanır.
    num_complete_mini_batches = shuffled_X.shape[1] // batch_size

    for i in range(num_complete_mini_batches):
        batch_X = shuffled_X[:, i * batch_size:(i + 1)*batch_size]
        batch_Y = shuffled_Y[:, i * batch_size:(i + 1) * batch_size]

        mini_batch = (batch_X, batch_Y)
        mini_batches.append(mini_batch)

    #Eğer bi tanesi complete bir batch değilse
    if X.shape[1] % batch_size != 0:
        batch_X =  shuffled_X[:, batch_size*num_complete_mini_batches:]
        batch_Y = shuffled_Y[:, batch_size*num_complete_mini_batches:]
        mini_batches.append((batch_X, batch_Y))

    return mini_batches

def predict(X,parameters):
    AL, caches = forward_propagation(X, parameters)
    predictions = (AL > 0.5)
    return predictions


layer_dims = [INPUT_LAYER_SIZE,128,64,1]

L = len(layer_dims) - 1 #input layerı saymıyoruz
parameters = dnn_model(X,Y,layer_dims,0.5,0.0002,5 )

#plot_decision_boundary(lambda x : predict(x.T,parameters),X,Y)
#plt.title("Decision Boundary")

plt.plot(costs,label="cost")
plt.show()

predictions = predict(X,parameters)

trues = 0

for i in range(predictions.shape[1]):  # Her bir örnek için
    if predictions[0, i] == Y[0, i]:  # Tahmin ve gerçek etiket aynı ise
        trues += 1

print(str(trues) + " correct predictions / " + str(m))
print("Accuracy:"+str(100*trues/m))

