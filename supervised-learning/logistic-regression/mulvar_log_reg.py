try:
  from sklearn.datasets import load_digits
  from sklearn.model_selection import train_test_split as tts
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import confusion_matrix
  import matplotlib.pyplot as mplt
  import seaborn as sb
except ImportError as e:
  print(e.msg)

digits = load_digits()


X_train, X_test, y_train, y_test = tts(digits.data, digits.target, train_size=0.8, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(digits.target[67])

print(model.predict(digits.data[0:5]))

yPred = model.predict(X_test)

confusionMatrix = confusion_matrix(y_test, yPred)
print(confusionMatrix)

mplt.figure(figsize=(10, 7))
sb.heatmap(confusionMatrix, annot=True)
mplt.xlabel('Predicted')
mplt.ylabel('true')
mplt.show()