from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
#load data from skelarn

digits = load_digits()
X = digits.data
y = digits.target

model = RandomForestClassifier()
# train the model
model.fit(X, y)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy}")
precision = precision_score(y, y_pred, average='macro')
print(f"Precision: {precision}")

