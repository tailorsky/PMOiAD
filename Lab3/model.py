import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.inspection import DecisionBoundaryDisplay


def train(Xtrain, Xtest, Ytrain, Ytest, N, isForest = False):
    Nvar = 0
    if isForest == False:   
        clf = DecisionTreeClassifier(random_state=Nvar).fit(Xtrain, Ytrain)
    else:
        clf = RandomForestClassifier(random_state=Nvar).fit(Xtrain, Ytrain)

    Pred_train = clf.predict(Xtrain)
    Pred_train_proba = clf.predict_proba(Xtrain)

    Pred_test = clf.predict(Xtest)
    Pred_test_proba = clf.predict_proba(Xtest)

    acc_train = clf.score(Xtrain, Ytrain)
    acc_test = clf.score(Xtest, Ytest)

    RocCurveDisplay.from_predictions(Ytest, Pred_test_proba[:, 1])
    plt.show()

    from sklearn.metrics import roc_auc_score

    AUC = roc_auc_score(Ytest, Pred_test_proba[:, 1])
    print("AUC tree:" + str(AUC))
    
    return Pred_test, Pred_test_proba, Pred_train, Pred_train_proba

def dataPratision(X, Y, N):
    trainCount = round(0.7 * N * 2)
    Xtrain = X[0:trainCount]
    Xtest = X[trainCount:N * 2 + 1]
    Ytrain = Y[0:trainCount]
    Ytest = Y[trainCount:N * 2 + 1]
    return Xtrain, Xtest, Ytrain, Ytest

def printAccSensSpec(Pred_test, Ytest, Pred_train, Ytrain):
    acc, sens, spec = calculateAccSensSpec(Pred_test, Ytest)
    print(f"Test:  точность = {acc:.2%}, чувствительность = {sens:.2%}, специфичность = {spec:.2%}")
    acc, sens, spec = calculateAccSensSpec(Pred_train, Ytrain)
    print(f"Train:  точность = {acc:.2%}, чувствительность = {sens:.2%}, специфичность = {spec:.2%}")

def calculateAccSensSpec(Pred, Y):
    true_positive = sum((Pred == 1) & (Y == 1))
    true_negative = sum((Pred == 0) & (Y == 0))
    false_positive = sum((Pred == 1) & (Y == 0))
    false_negative = sum((Pred == 0) & (Y == 1))

    acc = (true_positive + true_negative)/len(Y)
    sens = true_positive/(true_positive + false_negative)
    spec = true_negative/(true_negative + false_positive)

    return(acc, sens, spec)

def plotHist(Pred_test_proba, Ytest, Pred_train_proba, Ytrain):
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.hist(Pred_test_proba[Ytest, 1], bins='auto', alpha=0.7, label='Класс 1')
    plt.hist(Pred_test_proba[~Ytest, 1], bins='auto', alpha=0.7, label='Класс 0')
    plt.title('Результаты классификации, тест')
    plt.xlabel('Вероятность класса 1')
    plt.ylabel('Число объектов')
    plt.legend()

    plt.subplot(122)
    plt.hist(Pred_train_proba[Ytrain, 1], bins='auto', alpha=0.7, label='Класс 1')
    plt.hist(Pred_train_proba[~Ytrain, 1], bins='auto', alpha=0.7, label='Класс 0')
    plt.title('Результаты классификации, трейн')
    plt.xlabel('Вероятность класса 1')
    plt.ylabel('Число объектов')
    plt.legend()

    plt.show()

def tuneDepth(Xtrain, Xtest, Ytrain, Ytest):
    Nvar = 10
    depths = range(1, 21)
    acc_train = []
    acc_test = []

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=Nvar).fit(Xtrain, Ytrain)
        acc_train.append(clf.score(Xtrain, Ytrain))
        acc_test.append(clf.score(Xtest, Ytest))

    best = depths[np.argmax(acc_test)]
    print(f"Лучшая глубина дерева: {best}, точность на тесте = {max(acc_test):.2%}")

    plt.plot(depths, acc_train, 'o-', label='Train')
    plt.plot(depths, acc_test, 'o-', label='Test')
    plt.xlabel('Глубина дерева (max_depth)')
    plt.ylabel('Точность')
    plt.title('Точность дерева в зависимости от глубины')
    plt.legend()
    plt.show()

    return best

def tuneForest(Xtrain, Xtest, Ytrain, Ytest):
    Nvar = 10
    n_range = range(1, 301, 10)
    aucs = []

    for n in n_range:
        clf = RandomForestClassifier(n_estimators=n, random_state=Nvar).fit(Xtrain, Ytrain)
        aucs.append(roc_auc_score(Ytest, clf.predict_proba(Xtest)[:, 1]))

    best = n_range[np.argmax(aucs)]
    print(f"Лучшее число деревьев: {best}, AUC на тесте = {max(aucs):.4f}")

    plt.plot(n_range, aucs, 'o-')
    plt.xlabel('Число деревьев (n_estimators)')
    plt.ylabel('AUC (тест)')
    plt.title('AUC случайного леса в зависимости от числа деревьев')
    plt.show()

    return best

def plotBoundary(clf, X, Y, title):
    DecisionBoundaryDisplay.from_estimator(clf, X, response_method="predict", alpha=0.4)
    plt.scatter(X[~Y, 0], X[~Y, 1], s=8, label="Класс 0")
    plt.scatter(X[Y, 0], X[Y, 1], s=8, label="Класс 1")
    plt.title(title)
    plt.legend()
    plt.show()