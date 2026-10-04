import numpy as np
from sklearn.tree import DecisionTreeClassifier

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

Xtrain, Xtest, Ytrain, Ytest = model.dataPratision(X,Y,N)

#DecisionTreeClassifier
Pred_test, Pred_test_proba, Pred_train, Pred_train_proba = model.train(Xtrain, Xtest, Ytrain, Ytest, N, False)
model.printAccSensSpec(Pred_test, Ytest, Pred_train, Ytrain)

#RandomForestClassifier
Pred_test, Pred_test_proba, Pred_train, Pred_train_proba = model.train(Xtrain, Xtest, Ytrain, Ytest, N, True)
model.printAccSensSpec(Pred_test, Ytest, Pred_train, Ytrain)
model.plotHist(Pred_test_proba, Ytest, Pred_train_proba, Ytrain)

model.tuneDepth(Xtrain, Xtest, Ytrain, Ytest)
model.tuneForest(Xtrain, Xtest, Ytrain, Ytest)


#тут начинается выборка Б (т.е. плохо разделимая)
mu0 = [0, 2, 3]
mu1 = [1, 3, 1]
sigma0 = [2, 1, 2]
sigma1 = [2, 2, 1]

N = 1000
mu = [mu0, mu1]
sigma = [sigma0, sigma1]

X, Y, class0, class1 = dg.norm_dataset(mu, sigma, N)

Xtrain, Xtest, Ytrain, Ytest = model.dataPratision(X,Y,N)

#DecisionTreeClassifier
Pred_test, Pred_test_proba, Pred_train, Pred_train_proba = model.train(Xtrain, Xtest, Ytrain, Ytest, N, False)
model.printAccSensSpec(Pred_test, Ytest, Pred_train, Ytrain)

#RandomForestClassifier
Pred_test, Pred_test_proba, Pred_train, Pred_train_proba = model.train(Xtrain, Xtest, Ytrain, Ytest, N, True)
model.printAccSensSpec(Pred_test, Ytest, Pred_train, Ytrain)
model.plotHist(Pred_test_proba, Ytest, Pred_train_proba, Ytrain)

model.tuneDepth(Xtrain, Xtest, Ytrain, Ytest)
model.tuneForest(Xtrain, Xtest, Ytrain, Ytest)


#вот тут вот выборка В
print("Нелинейно-пересекаемые классы")
X, Y, class0, class1 = dg.nonlinear_dataset_10(N)

Xtrain, Xtest, Ytrain, Ytest = model.dataPratision(X,Y,N)

#DecisionTreeClassifier
Pred_test, Pred_test_proba, Pred_train, Pred_train_proba = model.train(Xtrain, Xtest, Ytrain, Ytest, N, False)
model.printAccSensSpec(Pred_test, Ytest, Pred_train, Ytrain)

#RandomForestClassifier
Pred_test, Pred_test_proba, Pred_train, Pred_train_proba = model.train(Xtrain, Xtest, Ytrain, Ytest, N, True)
model.printAccSensSpec(Pred_test, Ytest, Pred_train, Ytrain)
model.plotHist(Pred_test_proba, Ytest, Pred_train_proba, Ytrain)

model.tuneDepth(Xtrain, Xtest, Ytrain, Ytest)
model.tuneForest(Xtrain, Xtest, Ytrain, Ytest)