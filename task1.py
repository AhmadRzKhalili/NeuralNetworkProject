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
                    
    elif d == 5:
        X = data['X']
        y_hat = data['y']

        for i in range(len(X)):
            if y_hat[i] == 1:

                if y_hat[i] == y[i]:
                    plt.scatter(X[i, 1], X[i, 2], color='r')
                else:
                    plt.scatter(X[i, 1], X[i, 2], facecolors='None', color='r')

            elif y_hat[i] == 0:

                if y_hat[i] == y[i]:
                    plt.scatter(X[i, 1], X[i, 2], color='b')
                else:
                    plt.scatter(X[i, 1], X[i, 2], facecolors='None', color='b')


def Phi(X):
    return 1 / (1 + np.exp(-X))


def C(w, b, data):
    X = data['X']
    y = data['y']
    
    N = len(X)

    cost = 0
    for i in range(N):
        cost += (Phi(np.dot(X[i], w) + b) - y[i]) ** 2

    return cost
    

def compute_dC_dw(w,b, data):
    X = data['X']
    y = data['y']
    
    N = len(X)

    gradient = 0
    for i in range(N):
        gradient += (Phi(np.dot(X[i], w) + b) - y[i]) * (X[i] * Phi(np.dot(X[i], w) + b) * (1 - Phi(np.dot(X[i], w) + b)))

    gradient = 2 * gradient
    return gradient


def compute_dC_db(w,b, data):
    X = data['X']
    y = data['y']
    
    N = len(X)

    gradient = 0
    for i in range(N):
        gradient += (Phi(np.dot(X[i], w) + b) - y[i]) * (Phi(np.dot(X[i], w) + b) * (1 - Phi(np.dot(X[i], w) + b)))

    gradient = 2 * gradient
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
    phi = [Phi(np.dot(X[i], w) + b) for i in range(len(X))]
    y_hat = []

    for i in range(len(phi)):
        if phi[i] < 0.5:
            y_hat.append(0)
        else:
            y_hat.append(1)

    y_hat = np.array(y_hat)

    print("Misclassification ratio = ", cal_missclassification_err(y_hat, y)) 
    return {'X': X, 'y': y_hat}


def gradient_descent(data, w, b):
    X = data['X']
    y = data['y']

    lmbda = 5e-2
    w0 = w
    b0 = b

    for i in range(5000):
     

        dC_dw = compute_dC_dw(w, b, data)
        dC_db = compute_dC_db(w, b, data)
        w = w - lmbda * dC_dw
        b = b - lmbda * dC_db
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
X = raw_data['X']
y = raw_data['y']
data = {'X': X, 'y': y}
d = len(X[0])
w = np.random.rand(d)
b = random.uniform(0, 1)

dC_dw = compute_dC_dw(w,b, data)
dC_db = compute_dC_db(w,b, data)
dC_dw_n = compute_dC_dw_numeric(w,b, data)
dC_db_n = compute_dC_db_numeric(w,b, data)

print("Absolute error of ∂C/∂w = ", np.linalg.norm(dC_dw - dC_dw_n))
print("Relative error of ∂C/∂w = ", np.linalg.norm(dC_dw - dC_dw_n) / np.linalg.norm(dC_dw))
print("Absolute error of ∂C/∂b = ", np.linalg.norm(dC_db - dC_db_n))
print("Relative error of ∂C/∂b = ", np.linalg.norm(dC_db - dC_db_n) / np.linalg.norm(dC_db))

gradient_descent(data, w, b)
































# def C(w, b, data):
#     X = data['X']
#     y = data['y']
    
#     N = len(X)

#     cost = 0
#     for i in range(N):
#         cost += y[i] * math.log(Phi(np.dot(X[i], w) + b)) + (1 - y[i]) * math.log(1 - Phi(np.dot(X[i], w) + b))

#     return -1 * cost

# def compute_dC_dw(w,b, data):
#     X = data['X']
#     y = data['y']
    
#     N = len(X)

#     gradient = 0
#     for i in range(N):
#         ebarat = X[i] * (Phi(np.dot(X[i], w) + b) * (1 - Phi(np.dot(X[i], w) + b)))
#         gradient += (ebarat * y[i] * (1 / (math.log(2) * Phi(np.dot(X[i], w) + b)))) + ((1 - y[i]) * (-1 * ebarat) * (1 / (math.log(2) * (1 - Phi(np.dot(X[i], w) + b)))))

#     gradient = -1 * gradient
#     return gradient

# def compute_dC_db(w,b, data):
#     X = data['X']
#     y = data['y']
    
#     N = len(X)

#     gradient = 0
#     for i in range(N):
#         ebarat = (Phi(np.dot(X[i], w) + b) * (1 - Phi(np.dot(X[i], w) + b)))
#         gradient += (ebarat * y[i] * (1 / (math.log(2) * Phi(np.dot(X[i], w) + b)))) + ((1 - y[i]) * (-1 * ebarat) * (1 / (math.log(2) * (1 - Phi(np.dot(X[i], w) + b)))))

#     gradient = -1 * gradient
#     return gradient