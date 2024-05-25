try:
  import matplotlib.pyplot as mplt
  import numpy as np
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import train_test_split as ttSpit
  from sklearn.datasets import make_regression

except ImportError as e:
  print(e.msg)


X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

lrModel = LinearRegression()
lrModel.fit(X, y)

yPred = lrModel.predict(X)

mplt.scatter(X, y, color="green", label="Data Points")
mplt.plot(X, yPred, color="red", label="Regression Line")
mplt.title('Linear Regression using scikit-learn')
mplt.xlabel('Independent Var.')
mplt.ylabel('Dependent Var.')
mplt.show()