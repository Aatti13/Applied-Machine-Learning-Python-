# -*- encoding: utf-8 -*-
# Data-set

'''
  Basic Equation: f(w, b) = wx+b
  Sigmoid Function: h(w,b) = h(x) = 1/1+e^(-wx+b)
  Derivation given separately
'''

# Imports
# try:
#   import numpy as np
#   import matplotlib.pyplot as mplt
#   import pandas as pd
# except ImportError as e:
#   print(e.msg)

# def sigmoidFunction(z):
#   return 1/(1+np.exp(-z))

# def costFunction(X, y, theta):
#   m = len(y)
#   h = sigmoidFunction(np.dot(X, theta))
#   cost = (-1/m) * (np.dot(y, np.log(h))+np.dot((1-y), np.log(1-h)))
#   return cost

# def gradientDescent(X, y, theta, alpha, epochs):
#   m = len(y)
#   costHistory = []

#   for _ in range(epochs):
#     h = sigmoidFunction(np.dot(X, theta))
#     gradient = np.dot(X.T, (h-y))/m

#     theta -= alpha * gradient
#     cost = costFunction(X, y, theta)
#     costHistory.append(cost)
  
#   return cost, costHistory

# def plottingData(X, y, theta):
#   mplt.scatter(X[y == 0][:, 1], X[y == 0][:, 1], c='red', label='Class 0')
#   mplt.scatter(X[y == 1][:, 1], X[y == 1][:, 1], c='blue', label='Class 1')

#   xValues = [np.min(X[:, 1]), np.max(X[:, 1])]
#   yValues = -(theta[0] +  np.dot(theta[1], xValues))/theta[2]
#   mplt.plot(xValues, yValues, label="Decision Boundary")

#   mplt.xLabel = ('Feature-1')
#   mplt.ylabel = ('Feature-2')
#   mplt.legend()
#   mplt.show()

# if __name__ == "__main__":
#   X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
#   y = np.array([0, 0, 1, 1])

#   theta = np.zeros(X.shape[1])
#   alpha = 0.001 
#   epochs = 1000

#   theta, costHistory = gradientDescent(X, y, theta, alpha, epochs)
#   plottingData(X, y, theta)

  # print(f"Optimized theta: {theta}\nCost History: {costHistory}")


import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (-1/m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    return cost

# Gradient descent function
def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (h - y)) / m
        theta = theta - alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Plotting function
def plot_data_and_decision_boundary(X, y, theta):
    # Plot the data points
    plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], c='red', label='Class 0')
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], c='blue', label='Class 1')

    # Plot the decision boundary
    x_values = [np.min(X[:, 1]), np.max(X[:, 1])]
    y_values = - (theta[0] + np.dot(theta[1], x_values)) / theta[2]
    plt.plot(x_values, y_values, label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Synthetic dataset
    X = np.array([[1, 2, 1], [1, 3, 2], [1, 4, 3], [1, 5, 4]])
    y = np.array([0, 0, 1, 1])

    # Initialize parameters
    theta = np.zeros(X.shape[1])
    alpha = 0.01
    num_iterations = 1000

    # Perform gradient descent
    theta, cost_history = gradient_descent(X, y, theta, alpha, num_iterations)

    print("Optimized theta:", theta)
    print("Cost history:", cost_history)

    # Plot data and decision boundary
    plot_data_and_decision_boundary(X, y, theta)
