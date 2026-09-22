import numpy as np
import matplotlib.pyplot as plt

mu0 = [0, 2, 3]
mu1 = [3, 5, 1]
sigma0 = [2, 1, 2]
sigma1 = [1, 2, 1]

N = 1000
col = len(mu0)

class0 = np.random.normal(mu0[0], sigma0[0], [N, 1])
class1 = np.random.normal(mu1[0], sigma1[0], [N, 1])

for i in range(1, col):
    v0 = np.random.normal(mu0[i], sigma0[i], [N, 1])
    class0 = np.hstack((class0, v0))

    v1 = np.random.normal(mu1[i], sigma1[i], [N, 1])
    class1 = np.hstack((class1, v1))

Y1 = np.ones((N, 1), dtype=bool)
Y0 = np.zeros((N, 1), dtype=bool)

X = np.vstack((class0, class1))
Y = np.vstack((Y0, Y1)).ravel()

rng = np.random.default_rng()
arr = np.arange(2 * N)
rng.shuffle(arr)

X = X[arr]
Y = Y[arr]

trainCount = round(0.7 * N * 2)
Xtrain = X[0:trainCount]
Xtest = X[trainCount:N * 2 + 1]
Ytrain = Y[0:trainCount]
Ytest = Y[trainCount:N * 2 + 1]

for i in range(0, col):
    _ = plt.hist(class0[:, i], bins='auto', alpha=0.7, label='Класс 0')
    _ = plt.hist(class1[:, i], bins='auto', alpha=0.7, label='Класс 1')
    plt.title(f'Распределение признака {i + 1}')
    plt.xlabel(f'Значение признака {i + 1}')
    plt.ylabel('Частота')
    plt.legend()
    plt.savefig('hist_' + str(i + 1) + '.png')
    plt.show()

plt.scatter(class0[:, 0], class0[:, 2], marker=".", alpha=0.7, label='Класс 0')
plt.scatter(class1[:, 0], class1[:, 2], marker=".", alpha=0.7, label='Класс 1')
plt.title('Скаттерограмма')
plt.xlabel('Признак 1')
plt.ylabel('Признак 3')
plt.legend()
plt.savefig('scatter.png')
plt.show()