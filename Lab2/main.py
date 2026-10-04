import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

import DataGenerator as dg
import model

#вот тут выборка А (т.е. хорошо разделимая)
mu0 = [0, 2, 3]
mu1 = [3, 5, 1]
sigma0 = [2, 1, 2]
sigma1 = [1, 2, 1]

N = 1000
mu = [mu0, mu1]
sigma = [sigma0, sigma1]

X, Y, class0, class1 = dg.norm_dataset(mu, sigma, N)

trainCount = round(0.7 * N * 2)
Xtrain = X[0:trainCount]
Xtest = X[trainCount:N * 2 + 1]
Ytrain = Y[0:trainCount]
Ytest = Y[trainCount:N * 2 + 1]

Pred_test, Pred_train = model.train(Xtrain, Xtest, Ytrain, Ytest, N)

print('Выборка А (Хорошо разделимые):')
acc, sens, spec = model.calculateAccSensSpec(Pred_test, Ytest)
print(f"Test:  точность = {acc:.2%}, чувствительность = {sens:.2%}, специфичность = {spec:.2%}")
acc, sens, spec = model.calculateAccSensSpec(Pred_train, Ytrain)
print(f"Train:  точность = {acc:.2%}, чувствительность = {sens:.2%}, специфичность = {spec:.2%}")

#тут начинается выборка Б (т.е. плохо разделимая)
mu0 = [0, 2, 3]
mu1 = [1, 3, 2]
sigma0 = [2, 1, 2]
sigma1 = [2, 2, 3]

mu = [mu0, mu1]
sigma = [sigma0, sigma1]

X, Y, class0, class1 = dg.norm_dataset(mu, sigma, N)

trainCount = round(0.7 * N * 2)
Xtrain = X[0:trainCount]
Xtest = X[trainCount:N * 2 + 1]
Ytrain = Y[0:trainCount]
Ytest = Y[trainCount:N * 2 + 1]

Pred_test, Pred_train = model.train(Xtrain, Xtest, Ytrain, Ytest, N)
print('Выборка Б (Плохо разделимые):')
acc, sens, spec = model.calculateAccSensSpec(Pred_test, Ytest)
print(f"Test:  точность = {acc:.2%}, чувствительность = {sens:.2%}, специфичность = {spec:.2%}")
acc, sens, spec = model.calculateAccSensSpec(Pred_train, Ytrain)
print(f"Train:  точность = {acc:.2%}, чувствительность = {sens:.2%}, специфичность = {spec:.2%}")

#вот тут вот выборка В
X, Y, class0, class1 = dg.nonlinear_dataset_10(N)

trainCount = round(0.7 * N * 2)
Xtrain = X[0:trainCount]
Xtest = X[trainCount:N * 2 + 1]
Ytrain = Y[0:trainCount]
Ytest = Y[trainCount:N * 2 + 1]

Pred_test, Pred_train = model.train(Xtrain, Xtest, Ytrain, Ytest, N)
print('Выборка В (Нелинейно-пересекаемые):')
acc, sens, spec = model.calculateAccSensSpec(Pred_test, Ytest)
print(f"Test:  точность = {acc:.2%}, чувствительность = {sens:.2%}, специфичность = {spec:.2%}")
acc, sens, spec = model.calculateAccSensSpec(Pred_train, Ytrain)
print(f"Train:  точность = {acc:.2%}, чувствительность = {sens:.2%}, специфичность = {spec:.2%}")