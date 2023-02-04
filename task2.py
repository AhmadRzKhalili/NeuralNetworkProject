import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def draw_line(data, w, b):
    X = data['X']
    y = data['y']
    x0_vals = X[:, 0]
    x1_vals = (w[0] * X[:, 0] + b) / (-w[1])

    for i in range(len(X)):
        plt.plot(x0_vals, x1_vals, '-', color='g')


def plot_data(data, y, w, b, d):
    if d == 2:
        X = data['X']
        y_hat = data['y']

        for i in range(len(X)):
            if y_hat[i] == 1:

                if y_hat[i] == y[i]:
                    plt.scatter(X[i, 0], X[i, 1], color='r')
                else:
                    plt.scatter(X[i, 0], X[i, 1], facecolors='None', color='r')

            elif y_hat[i] == 0:
                
                if y_hat[i] == y[i]:
                    plt.scatter(X[i, 0], X[i, 1], color='b')
                else:
                    plt.scatter(X[i, 0], X[i, 1], facecolors='None', color='b')


def Phi(X):
    return 1 / (1 + tf.math.exp(-X))


def C(w, b, data):
    X = data['X']
    y = data['y']
    
    N = len(X)

    cost = 0
    for i in range(N):
        cost += (Phi(tf.tensordot(X[i], w, 1) + b) - y[i]) ** 2

    return cost


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
    h = 1e-6
    
    C1 = C(w, b, data)
    C2 = C(w + h, b, data)

    return (C2 - C1) / h


def compute_dC_db_numeric(w,b, data):
    h = 1e-6
    
    C1 = C(w, b, data)
    C2 = C(w, b + h, data)

    return (C2 - C1) / h


def cal_missclassification_err(y, y_hat):

    misclassified = 0
    N = len(y)

    for i in range(N):
        if y[i] != y_hat[i]:
            misclassified += 1

    return misclassified / N


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


def gradient_descent(data, w, b):
    X = data['X']
    y = data['y']

    lmbda = tf.Variable(5e-5, dtype=tf.float64)
    w0 = w
    b0 = b

    for i in range(5000):
        pre_b = b
        pre_w = w

        dC_dw = compute_dC_dw(w, b, data)
        dC_db = compute_dC_db(w, b, data)
        w = tf.Variable(w - tf.multiply(dC_dw, lmbda))
        b = tf.Variable(b - tf.multiply(dC_db, lmbda))
        print("Cost function = ", C(w, b, data))
        
        classified_data = classify(X, y, w, b)
        

        plt.cla()
        if d == 2:
            draw_line(raw_data, w0, b0)
        plot_data(classified_data, y, w, b, d)
        plt.draw()
        plt.pause(.1)

        y_hat = classified_data['y']
        if i > 0 and np.linalg.norm(np.array(dC_dw)) < 5 and np.linalg.norm(np.array(dC_db)) < 2:
            break
        elif cal_missclassification_err(y_hat, y) == 0.0:
            break

    print('Done')
    plot_data(classified_data, w, b, d)
    plt.show()


raw_data = np.load('data2d.npz')
X = tf.convert_to_tensor(raw_data['X'])
y = tf.convert_to_tensor(raw_data['y'])
data = {'X': X, 'y': y}
d = len(X[0])
w = tf.Variable(tf.random.uniform([d], minval=0, maxval=1, dtype=tf.float64))
b = tf.Variable(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float64))

dC_dw = compute_dC_dw(w, b, data)
dC_db = compute_dC_db(w, b, data)
dC_dw_n = compute_dC_dw_numeric(w,b, data)
dC_db_n = compute_dC_db_numeric(w,b, data)

print("Absolute error of ∂C/∂w = ", np.linalg.norm(dC_dw - dC_dw_n))
print("Relative error of ∂C/∂w = ", np.linalg.norm(dC_dw - dC_dw_n) / np.linalg.norm(dC_dw))
print("Absolute error of ∂C/∂b = ", np.linalg.norm(dC_db - dC_db_n))
print("Relative error of ∂C/∂b = ", np.linalg.norm(dC_db - dC_db_n) / np.linalg.norm(dC_db))

gradient_descent(data, w, b)