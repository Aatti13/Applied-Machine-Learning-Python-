# -*- encoding: utf-8 -*-
#  Assume the data-set to have data for the fields --> Study-time and Marks obtained

'''
  Students: 1-n
  Study-time: 0h to 10h
  Marks-Obtained: 0-100
'''

'''
  Basic Equation: y=mx+b
  Derivation given separately
'''
# imports
try:
  import pandas as pd
  import matplotlib.pyplot as mplt
except ImportError as e:
  print(e.msg)

csvData = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')


mplt.show()

# Method-1
'''
  Here the independent axis (X-axis) will have the hours studies.
  Dependent axis (Y-axis) will have marks obtained.

  Here we define a loss function --> we check the error function 
'''

def lossFunction(m, b, points):
  totalError = 0

  for i in range(len(points)):
    x = points.iloc[i].Hours
    y = points.iloc[i].Scores

    totalError += (y-(m*x+b))**2

  totalError / float(len(points))


def gradientDescent(currentM, currentB, points, L):
  mGradient = 0
  bGradient = 0

  n = len(points)

  for i in range(n):
    x = points.iloc[i].Hours
    y = points.iloc[i].Scores

    mGradient += -(2/n)*x*(y-(currentM*x+currentB))
    bGradient += -(2/n)*(y-(currentM*x+currentB))

  m = currentM - mGradient*L
  b = currentB - bGradient*L
  return m,b

if __name__ == "__main__":
  m = 0
  b = 0
  L = 0.001
  epochs = 100

  for _ in range(epochs):
    m, b = gradientDescent(m, b, csvData, L)

  print(m, b)

  mplt.scatter(csvData.Hours, csvData.Scores)
  mplt.title("Mathematical Linear Regression")
  mplt.xlabel('Study Time (in hours)')
  mplt.ylabel('Scores: (max: 100)')
  mplt.plot(list(range(1, 12)), [m*x+b for x in range(1, 12)], color="red")
  mplt.show()