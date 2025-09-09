import torch
from sklearn import datasets
from logistic_regression_model import LogisticRegression
from torch import nn

iris = datasets.load_iris()

model_0 = LogisticRegression(0.01, 1000)

loss_fn = nn.CrossEntropyLoss()
optimizer =  torch.optim.SGD(model_0, lr=0.01)

