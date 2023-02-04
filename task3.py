import tensorflow as tf
import numpy as np
import random


def Phi(X):
    return 1 / (1 + tf.math.exp(-X))

def C(h, data):
    X = data['X']
    y = data['y']
    
    N = len(X)
    cost = 0
    for i in range(N):
        cost += (h - y[i]) ** 2

    return cost

def compute_dC_dw(h, W, data):

    with tf.GradientTape() as tape:
            cost = C(h, data)

    gradient = tape.gradient(cost, W)

    return gradient

def compute_dC_db(h, b, data):

    with tf.GradientTape() as tape:
            cost = C(h, data)

    gradient = tape.gradient(cost, b)

    return gradient

def init_W(n, m):
    return [[random.random() for j in range(m)] for i in range(n)]

def init_b(n):
    return [random.random() for i in range(n)]

def softmax(vector):
    e = tf.math.exp(vector)
    return e / tf.reduce_sum(e)

X = []
y = []


with open('iris.data', 'r') as file:
    while (line := file.readline().rstrip()):
        data = line.split(",")
        feature = [float(data[i]) for i in range(len(data) - 1)]
        label = data[-1]

        X.append(feature)

        if label == 'Iris-setosa':
            y.append(0)
        elif label == 'Iris-versicolor':
            y.append(1)
        elif label == 'Iris-virginica':
            y.append(2)


data = {'X': X, 'y': y}
N = len(X)
d = len(X[0])
p = d * 2 - 1
q = p * 2

X = tf.convert_to_tensor(X, dtype=tf.float64)
y = tf.convert_to_tensor(y, dtype=tf.float64)

W1 = tf.convert_to_tensor(init_W(p, d), dtype=tf.float64)
b1 = tf.convert_to_tensor(init_b(p), dtype=tf.float64)

W2 = tf.convert_to_tensor(init_W(q, p), dtype=tf.float64)
b2 = tf.convert_to_tensor(init_b(q), dtype=tf.float64)

W3 = tf.convert_to_tensor(init_W(3, q), dtype=tf.float64)
b3 = tf.convert_to_tensor(init_b(3), dtype=tf.float64)

params = [W1, b1, W2, b2, W3, b3]


for i in range(N):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(params)

        h1 = Phi((tf.reduce_sum(W1 * X[i], axis=1)) + b1)

        h2 = Phi((tf.reduce_sum(W2 * h1, axis=1)) + b2)

        y_hat = softmax((tf.reduce_sum(W3 * h2, axis=1)) + b3)

        cost = C(y_hat, data)

    gradient = tape.gradient(cost, params)

    print(gradient)

