import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

training_set = pd.read_csv('/home/lukas/Desktop/digit-recognizer/train.csv')
test = pd.read_csv('/home/lukas/Desktop/digit-recognizer/test.csv')

test_set = test.to_numpy()
train_set = training_set.to_numpy()

def ReLU(x):
    x = np.maximum(0, x)
    return x



def Softmax(x):
    
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    output = exp_x / np.sum(exp_x, axis=1, keepdims=True)

    return output



def layer(weights, bias, x_in, func):
    
    z = x_in @ weights + bias
    activation = func(z)

    return z, activation



def init_parameters(layer_neurons):
    num_layers = len(layer_neurons)
    list_weights = []
    list_bias = []
    
    for i in range(1, num_layers):
        n_in = layer_neurons[i - 1]
        n_out = layer_neurons[i]

        weight = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
        list_weights.append(weight)
        list_bias.append(np.zeros((1, n_out)))

    return list_weights, list_bias



def forward_prop(x_in, w, b):
    num_layers = len(w)
    activation_dict = {}
    z_dict = {}
    
    for i in range(num_layers - 1):
        z, a = layer(w[i], b[i], x_in, ReLU)
        x_in = a
        z_dict[i + 1] = z
        activation_dict[i + 1] = a
        
    z, a = layer(w[-1], b[-1], x_in, Softmax)
    z_dict[num_layers] = z
    activation_dict[num_layers] = a
    
    return activation_dict, z_dict



def compute_cost(softmax_vector, labels):
    eps = 1e-15
    rows = np.arange(len(labels))
    cols = labels
    x = softmax_vector[rows, cols]
    
    loss = np.mean(-np.log(x + eps))
    return loss



def ReLU_derivative(z):
    return (z > 0).astype(float)



def encoder(labels):
    num_samples = len(labels)
    y_vector = np.zeros((num_samples, 10))
    rows = np.arange(num_samples)
    y_vector[rows, labels] = 1

    return y_vector



def backpropagation(w, b, activation_dict, z_dict, y_in, x_in):

    error_vector = encoder(y_in)
    
    num_layers = len(w)
    gradients_w = [None] * num_layers
    gradients_b = [None] * num_layers

    ###  Pochodna dC/dw
    A_out = activation_dict[num_layers] ### Funkcja aktywacji którą wyrzuca output layer
    A_prev = activation_dict[num_layers - 1] ### Funkcja aktywacji która trafia do output layer

    # dC/da(L)
    gamma = A_out - error_vector

    gradients_w[-1] = A_prev.T @ gamma / x_in.shape[0]
    gradients_b[-1] = np.mean(gamma, axis = 0, keepdims=True)

    for L in reversed(range(1, num_layers)):

        if L == 1:
            A_prev = x_in
        else:
            A_prev = activation_dict[L-1]

        # gamma z następnej warstwy * waga następnej warstwy * pochodna funkcji aktywacji aktualnej warstwy
        gamma = (gamma @ w[L].T) * ReLU_derivative(z_dict[L])
        
        gradients_w[L-1] = A_prev.T @ gamma / x_in.shape[0]
        gradients_b[L-1] = np.mean(gamma, axis = 0, keepdims=True)


    return gradients_w, gradients_b



def create_batches(arr, batch_size):
    rng.shuffle(arr)
    container = []
    labels = []
    
    for i in range(0, len(arr), batch_size):
        sample = arr[i : i + batch_size, 1:]
        sample = sample / 255.0
        label = arr[i : i + batch_size, 0]
        labels.append(label)
        container.append(sample)

    return container, labels


rng = np.random.default_rng()
layer_neurons = [784, 256, 10, 10, 10]
w, b = init_parameters(layer_neurons)
epochs = 101
learning_rate = 0.01


def make_predictions(dataset, w, b):
    a_dict, _, = forward_prop(dataset, w, b)
    predictions = np.argmax(a_dict[len(w)], axis = 1)
    return predictions

predictions = make_predictions(test_set, w, b)


def accuracy(dataset, w, b):
    a_dict, _, = forward_prop(dataset[:, 1:], w, b)
    mapa = np.argmax(a_dict[4], axis=1) == dataset[:, 0]
    accuracy = np.sum(mapa) / len(mapa)

    return accuracy



for epoch in range(epochs):
    
    x_train, y_train = create_batches(train_set, 128)
    
    for batch_idx in range(len(x_train)):
        a_dict, z_dict = forward_prop(x_train[batch_idx], w, b)
        gradients_w, gradients_b = backpropagation(w, b, a_dict, z_dict, y_train[batch_idx], x_train[batch_idx])

        for i in range(len(gradients_w)):
            w[i] -= learning_rate * gradients_w[i]
            b[i] -= learning_rate * gradients_b[i]

    if epoch % 40 == 0:
        epoch_loss = 0
        
        for batch_idx in range(len(x_train)):
            a_dict, z_dict = forward_prop(x_train[batch_idx], w, b)
            loss = compute_cost(a_dict[len(w)], y_train[batch_idx])
            epoch_loss += loss

        epoch_loss /= len(x_train)
        print(f"Koszt w epoce nr: {epoch}, wyniósł: {epoch_loss}")

