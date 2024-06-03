try:
  from math import exp
  import pandas as pd
  import matplotlib.pyplot as mplt
  from sklearn.model_selection import train_test_split as tts
  from sklearn.linear_model import LogisticRegression
except ImportError as e:
  print(e.msg)

df = pd.read_csv('./Datasets/insurance_data.csv')

X = df.age
y = df.bought_insurance

X_train, X_test, y_train, y_test = tts(df[['age']], y, train_size=0.8, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

yPred = model.predict(X_test)
print(model.predict_proba(X_test))

print(model.coef_)
print(model.intercept_)


mplt.scatter(X, y, marker='+', color='red')


mplt.show()