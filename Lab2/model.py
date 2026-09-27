import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def train(Xtrain, Xtest, Ytrain, Ytest, N):

    Nvar = 10
    clf = LogisticRegression(random_state=Nvar, solver='saga').fit(Xtrain, Ytrain)

    Pred_train = clf.predict(Xtrain)
    Pred_train_proba = clf.predict_proba(Xtrain)

    Pred_test = clf.predict(Xtest)
    Pred_test_proba = clf.predict_proba(Xtest)

    acc_train = clf.score(Xtrain, Ytrain)
    acc_test = clf.score(Xtest, Ytest)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(Pred_train_proba[Ytrain, 1], bins='auto', alpha=0.7, label='Класс 1')
    axes[0].hist(Pred_train_proba[~Ytrain, 1], bins='auto', alpha=0.7, label='Класс 0')
    axes[0].set_title("Результаты классификации, трейн")

    axes[1].hist(Pred_test_proba[Ytest, 1], bins='auto', alpha=0.7, label='Класс 1')
    axes[1].hist(Pred_test_proba[~Ytest, 1], bins='auto', alpha=0.7, label='Класс 0')
    axes[1].set_title("Результаты классификации, тест")

    plt.tight_layout()
    plt.show()
    
    return Pred_test, Pred_train

def calculateAccSensSpec(Pred, Y):
    true_positive = sum((Pred == 1) & (Y == 1))
    true_negative = sum((Pred == 0) & (Y == 0))
    false_positive = sum((Pred == 1) & (Y == 0))
    false_negative = sum((Pred == 0) & (Y == 1))

    acc = (true_positive + true_negative)/len(Y)
    sens = true_positive/(true_positive + false_negative)
    spec = true_negative/(true_negative + false_positive)

    return(acc, sens, spec)