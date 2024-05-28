# CSV Dataset --> Take from codebasics.io --> 

try:
  import pandas as pd
  from sklearn.linear_model import LinearRegression
  import matplotlib.pyplot as mplt
except ImportError as e:
  print(e.msg)

df = pd.read_csv('Datasets/canada_per_capita_income.csv')

X = df[['year']]
y = df['per capita income (US$)']

model = LinearRegression()
model.fit(X, y)

yPred = model.predict(X)

print(model.coef_)
print(model.intercept_)
print(model.predict([[2020]]))

mplt.scatter(X, y, marker='+', color="red")
mplt.plot(df[['year']], yPred, color="blue")
mplt.xlabel("Year")
mplt.ylabel("Per Capita Income (US$)")
mplt.show()