from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
import joblib


scaler = StandardScaler()


data = pd.read_csv('CSV/train3.csv')
testdata = pd.read_csv('CSV/test3.csv')
data = data.sample(frac=1).reset_index(drop=True)

y = data['class'].values
X = data.drop(columns=['class']).values
scaler.fit(X)
X = scaler.transform(X)

y_test = testdata['class'].values
X_test = testdata.drop(columns=['class']).values
X_test = scaler.transform(X_test)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
model = clf.fit(X, y)
pred_train = model.predict(X)
pred_test = model.predict(X_test)
print('TRAINING !!!!!')
print(confusion_matrix(y,pred_train))
print(classification_report(y,pred_train))

print('TESTING !!!!!')
print(confusion_matrix(y_test,pred_test))
print(classification_report(y_test,pred_test))

# model_filename = 'models/SKL_NN_model.sav'
# scaler_filename = 'models/SKL_scaler.bin'
# joblib.dump(model, model_filename)
# joblib.dump(scaler, scaler_filename)
