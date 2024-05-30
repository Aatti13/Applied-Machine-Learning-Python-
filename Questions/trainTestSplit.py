try:
  import pandas as pd
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import train_test_split as tts
except ImportError as e:
  print(e.msg)

df = pd.read_csv('Datasets/carprices.csv')


X = df[['Age(yrs)']]
y = df['Sell Price($)']

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, train_size=0.8)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.predict(X_test))
print(model.score(X_test, y_test))