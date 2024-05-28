try:
  import pandas as pd
  from sklearn.linear_model import LinearRegression
except ImportError as e:
  print(e.msg)

df = pd.read_csv('Datasets/homeprices.csv')

# Data Pre-processing
df[['bedrooms']] = df[['bedrooms']].fillna(float(df.iloc[0].bedrooms.mean()))

X1 = df[['area']]
X2 = df[['bedrooms']]
X3 = df[['age']]


y = df['price']

X = df.drop('price', axis='columns')

lrModel = LinearRegression()
lrModel.fit(X, y)

print(lrModel.predict([[3000, 3, 40]]))
print(lrModel.predict([[2500, 4, 5]]))
