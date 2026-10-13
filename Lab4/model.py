import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_derivative(p):
    return p * (1 - p)


class NeuralNetwork:
    def __init__(self, x, y, n_neuro=4, lr=2):
        self.input = x
        n_inp = self.input.shape[1]
        self.lr = lr

        self.weights1 = np.random.rand(n_inp, n_neuro)
        self.weights2 = np.random.rand(n_neuro, 1)

        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backprop(self):
        m = self.input.shape[0]
        err = 2 * (self.y - self.output) * sigmoid_derivative(self.output)
        d_weights2 = np.dot(self.layer1.T, err)
        d_weights1 = np.dot(self.input.T,
                            np.dot(err, self.weights2.T) * sigmoid_derivative(self.layer1))

        self.weights1 += self.lr * d_weights1 / m
        self.weights2 += self.lr * d_weights2 / m

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()

    def test(self, X):
        layer1 = sigmoid(np.dot(X, self.weights1))
        return sigmoid(np.dot(layer1, self.weights2))


def dataPartition(X, Y, N):
    trainCount = round(0.7 * N * 2)
    Xtrain = X[0:trainCount]
    Xtest = X[trainCount:]
    Ytrain = Y[0:trainCount]
    Ytest = Y[trainCount:]
    return Xtrain, Xtest, Ytrain, Ytest


def trainNetwork(NN, Xtrain, Ytrain, Xtest, Ytest, N_epoch, verbose=True):
    history = {'loss_train': [], 'acc_train': [], 'loss_test': [], 'acc_test': []}
    for i in range(N_epoch):
        pred_train = NN.feedforward()
        pred_test = NN.test(Xtest)

        history['loss_train'].append(np.mean(np.square(Ytrain - pred_train)))
        history['acc_train'].append(np.mean((pred_train > 0.5) == Ytrain))
        history['loss_test'].append(np.mean(np.square(Ytest - pred_test)))
        history['acc_test'].append(np.mean((pred_test > 0.5) == Ytest))

        if verbose and i % 100 == 0:
            print(f"Эпоха {i}: loss = {history['loss_train'][-1]:.4f}, "
                  f"accuracy = {history['acc_train'][-1]:.4f}")

        NN.train(Xtrain, Ytrain)
    return history


def calculateAccSensSpec(Pred, Y):
    Pred = Pred.ravel()
    Y = Y.ravel()
    true_positive = np.sum((Pred == 1) & (Y == 1))
    true_negative = np.sum((Pred == 0) & (Y == 0))
    false_positive = np.sum((Pred == 1) & (Y == 0))
    false_negative = np.sum((Pred == 0) & (Y == 1))

    acc = (true_positive + true_negative) / len(Y)
    sens = true_positive / (true_positive + false_negative)
    spec = true_negative / (true_negative + false_positive)
    return acc, sens, spec


def printAccSensSpec(Pred_test, Ytest, Pred_train, Ytrain):
    acc, sens, spec = calculateAccSensSpec(Pred_test, Ytest)
    print(f"Test:  точность = {acc:.2%}, чувствительность = {sens:.2%}, специфичность = {spec:.2%}")
    acc, sens, spec = calculateAccSensSpec(Pred_train, Ytrain)
    print(f"Train: точность = {acc:.2%}, чувствительность = {sens:.2%}, специфичность = {spec:.2%}")


def plotLearning(history, name=''):
    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.plot(history['loss_train'], label='Обучающая')
    plt.plot(history['loss_test'], label='Тестовая')
    plt.title(f'Функция потерь (MSE), выборка {name}')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(122)
    plt.plot(history['acc_train'], label='Обучающая')
    plt.plot(history['acc_test'], label='Тестовая')
    plt.title(f'Точность, выборка {name}')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'Lab4/learning_{name}.png')
    plt.show()


def plotHist(Pred_test_proba, Ytest, Pred_train_proba, Ytrain, name=''):
    Pred_test_proba = Pred_test_proba.ravel()
    Pred_train_proba = Pred_train_proba.ravel()
    Ytest = Ytest.ravel() == 1
    Ytrain = Ytrain.ravel() == 1

    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.hist(Pred_test_proba[Ytest], bins='auto', alpha=0.7, label='Класс 1')
    plt.hist(Pred_test_proba[~Ytest], bins='auto', alpha=0.7, label='Класс 0')
    plt.title(f'Результаты классификации, тест, выборка {name}')
    plt.xlabel('Вероятность класса 1')
    plt.ylabel('Число объектов')
    plt.legend()

    plt.subplot(122)
    plt.hist(Pred_train_proba[Ytrain], bins='auto', alpha=0.7, label='Класс 1')
    plt.hist(Pred_train_proba[~Ytrain], bins='auto', alpha=0.7, label='Класс 0')
    plt.title(f'Результаты классификации, трейн, выборка {name}')
    plt.xlabel('Вероятность класса 1')
    plt.ylabel('Число объектов')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'Lab4/hist_{name}.png')
    plt.show()


def plotBoundary(NN, X, Y, name=''):
    n_features = X.shape[1]
    pairs = list(combinations(range(n_features), 2))
    means = X.mean(axis=0)
    y = Y.ravel()

    cmap = LinearSegmentedColormap.from_list('orange_blue', ['tab:orange', 'white', 'tab:blue'])

    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, (i, j) in zip(axes, pairs):
        xi = np.linspace(X[:, i].min() - 1, X[:, i].max() + 1, 200)
        xj = np.linspace(X[:, j].min() - 1, X[:, j].max() + 1, 200)
        gi, gj = np.meshgrid(xi, xj)

        grid = np.tile(means, (gi.size, 1))
        grid[:, i] = gi.ravel()
        grid[:, j] = gj.ravel()

        proba = NN.test(grid).reshape(gi.shape)

        cf = ax.contourf(gi, gj, proba, levels=np.linspace(0, 1, 21),
                         cmap=cmap, alpha=0.5)
        ax.contour(gi, gj, proba, levels=[0.5], colors='black', linewidths=2)

        ax.scatter(X[y == 0, i], X[y == 0, j], s=10, c='tab:orange', label='Класс 0')
        ax.scatter(X[y == 1, i], X[y == 1, j], s=10, c='tab:blue', label='Класс 1')

        fixed = [f'x{k + 1} = {means[k]:.2f}' for k in range(n_features) if k not in (i, j)]
        subtitle = f'  ({", ".join(fixed)})' if fixed else ''
        ax.set_xlabel(f'x{i + 1}')
        ax.set_ylabel(f'x{j + 1}')
        ax.set_title(f'x{i + 1} и x{j + 1}{subtitle}')
        ax.legend(loc='upper left')

    fig.colorbar(cf, ax=list(axes), label='Вероятность класса 1')
    fig.suptitle(f'Разделяющая граница, выборка {name}')
    plt.savefig(f'Lab4/boundary_{name}.png')
    plt.show()


def tuneNeurons(Xtrain, Ytrain, Xtest, Ytest, neurons_list, N_epoch, name=''):
    plt.figure(figsize=(12, 4))
    for n in neurons_list:
        NN = NeuralNetwork(Xtrain, Ytrain, n_neuro=n)
        history = trainNetwork(NN, Xtrain, Ytrain, Xtest, Ytest, N_epoch, verbose=False)
        print(f"Нейронов: {n}, итоговая точность на тесте: {history['acc_test'][-1]:.4f}")

        plt.subplot(121)
        plt.plot(history['loss_test'], label=f'{n} нейр.')
        plt.subplot(122)
        plt.plot(history['acc_test'], label=f'{n} нейр.')

    plt.subplot(121)
    plt.title(f'Функция потерь на тесте, выборка {name}')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(122)
    plt.title(f'Точность на тесте, выборка {name}')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'Lab4/tune_{name}.png')
    plt.show()