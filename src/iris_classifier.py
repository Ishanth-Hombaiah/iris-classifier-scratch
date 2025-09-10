import numpy as np
from sklearn import datasets
from logistic_regression_model import LogisticRegression

iris = datasets.load_iris()

X, y = iris.data, iris.target

iris_classes = np.unique(y)

models = []

for i in iris_classes: # Training 3 models, one for each iris species, so we can compare their results for each input to determine which species each is
    curr_y = (y == i).astype(int)
    curr_model = LogisticRegression(0.01, 3000) 
    curr_model.fit(X, curr_y, 26) 
    models.append(curr_model)

def ovr_predict(x):
    x = np.asarray(x, dtype=float) # In case input is a 1d sample
    probs = np.column_stack([model.predict_probs(x) for model in models])
    return np.argmax(probs, axis=1)