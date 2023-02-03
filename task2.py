import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def draw_line(data2d, w, b):

    X = data2d['X']
    y = data2d['y']
    x_vals = X[:, 0]
    y_vals = (w[0] * X[:, 0] + b) / (-w[1])

    for i in range(len(X)):
        plt.plot(x_vals, y_vals, '-', color='g')
    

def scatter_2d_data(data2d, w, b):
    X = data2d['X']
    y = data2d['y']

    for i in range(len(X)):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], color='r')
        elif y[i] == 0:
            plt.scatter(X[i, 0], X[i, 1], color='b')

def Phi(X):
    return 1 / (1 + tf.math.exp(-X))


def C(w, b, data):
    X = data[0]
    y = data[1]
    
    N = len(X)

    cost = 0
    for i in range(N):
        cost += (Phi(tf.tensordot(X[i], w, 1) + b) - y[i]) ** 2

    return cost

def cross_entropy(w, b, data):
    X = data[0]
    y = data[1]
    
    N = len(X)

    cost = 0
    for i in range(N):
        cost += y[i] * tf.math.log(Phi(tf.tensordot(X[i], w, 1) + b)) + (1 - y[i]) * tf.math.log(1 - Phi(tf.tensordot(X[i], w, 1) + b))

    return -1 * cost

def compute_dC_dw(w,b, data):

    with tf.GradientTape() as tape:
            cost = C(w, b, data)

    gradient = tape.gradient(cost, w)

    return gradient

def compute_dC_db(w,b, data):

    with tf.GradientTape() as tape:
            cost = C(w, b, data)

    gradient = tape.gradient(cost, b)

    return gradient

def compute_dC_dw_numeric(w,b, data):
    eps = 1e-6
    
    C1 = C(w, b, data)
    C2 = C(w + eps, b, data)

    return (C2 - C1) / eps

def compute_dC_db_numeric(w,b, data):
    eps = 1e-6
    
    C1 = C(w, b, data)
    C2 = C(w, b + eps, data)

    return (C2 - C1) / eps

def classify(X, y, w, b):
    phi = [Phi(tf.tensordot(X[i], w, 1) + b) for i in range(len(X))]
    y_hat = []
    for i in range(len(phi)):
        if phi[i] < 0.5:
            y_hat.append(0)
        else:
            y_hat.append(1)

    y_hat = np.array(y_hat)
    # print(y_hat)
    # print(y)

    misclassified = 0
    for i in range(len(y)):
        if y[i] != y_hat[i]:
            misclassified += 1
    N = len(y)
    print("Training error ratio = ", misclassified / N) 

    return {'X': X, 'y': y_hat}

    
raw_data = np.load('data2d.npz')
X = tf.convert_to_tensor(raw_data['X'])
y = tf.convert_to_tensor(raw_data['y'])
data = (X, y)
d = len(X[0])
w = tf.Variable(tf.random.uniform([d], minval=0, maxval=1, dtype=tf.float64))
b = tf.Variable(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float64))

dC_dw = compute_dC_dw(w,b, data)
dC_db = compute_dC_db(w,b, data)
dC_dw_n = compute_dC_dw_numeric(w,b, data)
dC_db_n = compute_dC_db_numeric(w,b, data)

print("Absolute error of gradient of C w.r.t w = ", np.linalg.norm(dC_dw - dC_dw_n)) # absolute error
print("Relative error of gradient of C w.r.t w = ", np.linalg.norm(dC_dw - dC_dw_n) / np.linalg.norm(dC_dw)) # relative error
print("Absolute error of gradient of C w.r.t b = ", np.linalg.norm(dC_db - dC_db_n)) # absolute error
print("Relative error of gradient of C w.r.t b = ", np.linalg.norm(dC_db - dC_db_n) / np.linalg.norm(dC_db)) # relative error

# Gradient descent
lmbda = tf.Variable(0.00005, dtype=tf.float64)

w0 = w
b0 = b
step = 0
while True:

    step += 1
    pre_b = b
    pre_w = w

    dC_dw = compute_dC_dw(w,b, data)
    dC_db = compute_dC_db(w,b, data)
    w = tf.Variable(w - tf.multiply(dC_dw, lmbda))
    b = tf.Variable(b - tf.multiply(dC_db, lmbda))
    print("Cost function = ", C(w, b, data))
    print(np.linalg.norm(np.array(dC_dw)))
    print(np.linalg.norm(np.array(dC_db)))

    
    data2d = classify(X, y, w, b)
    

    plt.cla()
    draw_line(raw_data, w0, b0)
    scatter_2d_data(data2d, w, b)
    plt.draw()
    plt.pause(.1)

    if step > 1 and np.linalg.norm(np.array(dC_dw)) < 5 and np.linalg.norm(np.array(dC_db)) < 2:
        print("Best solution in step ", step)
        break

    if step == 5000:
        print("Best solution in step ", step)
        break

scatter_2d_data(data2d, w, b)
plt.show()