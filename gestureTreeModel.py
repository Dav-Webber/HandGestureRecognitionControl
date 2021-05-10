from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score

##Train tree
data = pd.read_csv('CSV/train3.csv')
data = data.sample(frac=1).reset_index(drop=True)
classifier = tree.DecisionTreeClassifier()

y = data['class'].values
X = data.drop(columns=['class']).values

model= classifier.fit(X,y)

text_representation = tree.export_text(model)
print(text_representation)

##Test
prediction = classifier.predict(X)
print(accuracy_score(y,prediction))


# filename = 'models/tree_model.sav'
# joblib.dump(model, filename)