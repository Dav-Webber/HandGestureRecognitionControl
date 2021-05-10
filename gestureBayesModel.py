from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score

data = pd.read_csv('CSV/train3.csv')
data = data.sample(frac=1).reset_index(drop=True)
gnb = GaussianNB()

testdata = pd.read_csv('CSV/test3.csv')

y = data['class'].values
X = data.drop(columns=['class']).values

y_test = testdata['class'].values
X_test = testdata.drop(columns=['class']).values
model = gnb.fit(X, y)
y_pred = model.predict(X_test)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print(accuracy_score(y_test, y_pred))

# filename = 'models/GNB_model.sav'
# joblib.dump(model, filename)