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

# def scatter_5d_data():
#     data5d = np.load('data5d.npz')
#     X = data5d['X']
#     plt.scatter(X[:,1], X[:,2], )
#     plt.show()

def Phi(X):
    return 1 / (1 + np.exp(-X))


def C(w, b, data):
    X = data[0]
    y = data[1]
    
    N = len(X)

    cost = 0
    for i in range(N):
        cost += (Phi(np.dot(X[i], w) + b) - y[i]) ** 2

    return cost
    

def compute_dC_dw(w,b, data):
    X = data[0]
    y = data[1]
    
    N = len(X)

    gradient = 0
    for i in range(N):
        gradient += (Phi(np.dot(X[i], w) + b) - y[i]) * (X[i] * Phi(np.dot(X[i], w) + b) * (1 - Phi(np.dot(X[i], w) + b)))

    gradient = 2 * gradient
    return gradient

def compute_dC_db(w,b, data):
    X = data[0]
    y = data[1]
    
    N = len(X)

    gradient = 0
    for i in range(N):
        gradient += (Phi(np.dot(X[i], w) + b) - y[i]) * (Phi(np.dot(X[i], w) + b) * (1 - Phi(np.dot(X[i], w) + b)))

    gradient = 2 * gradient
    return gradient

# def C(w, b, data):
#     X = data[0]
#     y = data[1]
    
#     N = len(X)

#     cost = 0
#     for i in range(N):
#         cost += y[i] * math.log(Phi(np.dot(X[i], w) + b)) + (1 - y[i]) * math.log(1 - Phi(np.dot(X[i], w) + b))

#     return -1 * cost

# def compute_dC_dw(w,b, data):
#     X = data[0]
#     y = data[1]
    
#     N = len(X)

#     gradient = 0
#     for i in range(N):
#         ebarat = X[i] * (Phi(np.dot(X[i], w) + b) * (1 - Phi(np.dot(X[i], w) + b)))
#         gradient += (ebarat * y[i] * (1 / (math.log(2) * Phi(np.dot(X[i], w) + b)))) + ((1 - y[i]) * (-1 * ebarat) * (1 / (math.log(2) * (1 - Phi(np.dot(X[i], w) + b)))))

#     gradient = -1 * gradient
#     return gradient

# def compute_dC_db(w,b, data):
#     X = data[0]
#     y = data[1]
    
#     N = len(X)

#     gradient = 0
#     for i in range(N):
#         ebarat = (Phi(np.dot(X[i], w) + b) * (1 - Phi(np.dot(X[i], w) + b)))
#         gradient += (ebarat * y[i] * (1 / (math.log(2) * Phi(np.dot(X[i], w) + b)))) + ((1 - y[i]) * (-1 * ebarat) * (1 / (math.log(2) * (1 - Phi(np.dot(X[i], w) + b)))))

#     gradient = -1 * gradient
#     return gradient

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
    phi = [Phi(np.dot(X[i], w) + b) for i in range(len(X))]
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
X = raw_data['X']
y = raw_data['y']
data = (X,y)
w = np.random.rand(len(X[0]))
b = random.uniform(0, 1)

# for i in range(len(X)):
#         if y[i] == 1:
#             plt.scatter(X[i, 0], X[i, 1], color='r')
#         elif y[i] == 0:
#             plt.scatter(X[i, 0], X[i, 1], color='b')
# plt.show()

dC_dw = compute_dC_dw(w,b, data)
dC_db = compute_dC_db(w,b, data)
dC_dw_n = compute_dC_dw_numeric(w,b, data)
dC_db_n = compute_dC_db_numeric(w,b, data)

print("Absolute error of gradient of C w.r.t w = ", np.linalg.norm(dC_dw - dC_dw_n)) # absolute error
print("Relative error of gradient of C w.r.t w = ", np.linalg.norm(dC_dw - dC_dw_n) / np.linalg.norm(dC_dw)) # relative error
print("Absolute error of gradient of C w.r.t b = ", np.linalg.norm(dC_db - dC_db_n)) # absolute error
print("Relative error of gradient of C w.r.t b = ", np.linalg.norm(dC_db - dC_db_n) / np.linalg.norm(dC_db)) # relative error


# Gradient descent
lmbda = 0.00005
step = 0
w0 = w
b0 = b
while True:


    step += 1
    pre_b = b
    pre_w = w

    dC_dw = compute_dC_dw(w,b, data)
    dC_db = compute_dC_db(w,b, data)
    w = w - lmbda * dC_dw
    b = b - lmbda * dC_db
    print("Cost function = ", C(w, b, data))
    # print(np.linalg.norm(np.array(dC_dw)))
    # print(np.linalg.norm(np.array(dC_db)))
    
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

# todo: scatter5d, change lines, 100% error