try:
  import numpy as np
except ImportError as e:
  print(e.msg)

def gradientDescent(x, y):
  currentM = currentB = 0
  n = len(x)
  epochs = 50
  alpha = 0.001
  for i in range(epochs):
    yPred = currentM*x + currentB
    costFunction = (1/n)*sum([val**2 for val in (y-yPred)])
    md = (-2/n)*sum(x*(y-(yPred)))
    bd = (-2/n)*sum(y-(yPred))

    currentM -= alpha*md
    currentB -= alpha*bd 

    print(f'm: {currentM}; b: {currentB}; cost: {costFunction}; iteration: {i+1}')
    

if __name__ == "__main__":
  x = np.array([1,2,3,4,5])
  y = np.array([5,7,9,11,13])
  gradientDescent(x, y)
