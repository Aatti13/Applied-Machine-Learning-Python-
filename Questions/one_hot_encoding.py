try:
  import pandas as pd
  from sklearn.linear_model import LinearRegression
  from sklearn.preprocessing import LabelEncoder
except ImportError as e:
  print(e.msg)

data = pd.read_csv('Datasets/homeprices_encoding.csv')
print(data)

dummyVars = pd.get_dummies(data.town)
# print(dummyVars)

mergedData = pd.concat([data, dummyVars], axis='columns')
# print(mergedData)

final = mergedData.drop(['town', 'west windsor'], axis='columns')
# print(final)

lrModel = LinearRegression()

X = final.drop('price', axis='columns')
y = final.price

lrModel.fit(X, y)
print(lrModel.predict([[2800, 0, 1]]))
print(lrModel.score(X, y))

# Label Encoding Method
labEnc = LabelEncoder()
dataLE = data

dataLE.town = labEnc.fit_transform(dataLE.town)
print(dataLE)