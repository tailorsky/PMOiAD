import DataGenerator as dg
import matplotlib.pyplot as plt

mu0 = [0, 2, 3]
mu1 = [3, 5, 1]
sigma0 = [2, 1, 2]
sigma1 = [1, 2, 1]

N = 1000
mu = [mu0, mu1]
sigma = [sigma0, sigma1]

X_lin, Y_lin, class0_lin, class1_lin = dg.norm_dataset(mu, sigma, N)
X_nl, Y_nl, class0_nl, class1_nl = dg.nonlinear_dataset_10(N)

col = X_lin.shape[1]
for i in range(col):
    plt.hist(class0_lin[:, i], bins='auto', alpha=0.7, label='Класс 0')
    plt.hist(class1_lin[:, i], bins='auto', alpha=0.7, label='Класс 1')
    plt.title(f'Распределение признака {i + 1}')
    plt.xlabel(f'Значение признака {i + 1}')
    plt.ylabel('Частота')
    plt.legend()
    plt.savefig(f'hist_lin_{i + 1}.png')
    plt.show()

plt.scatter(class0_lin[:, 0], class0_lin[:, 2], marker=".", alpha=0.7, label='Класс 0')
plt.scatter(class1_lin[:, 0], class1_lin[:, 2], marker=".", alpha=0.7, label='Класс 1')
plt.title('Линейный датасет: признак 1 vs признак 3')
plt.xlabel('Признак 1')
plt.ylabel('Признак 3')
plt.legend()
plt.savefig('scatter_lin.png')
plt.show()

plt.scatter(class0_nl[:, 0], class0_nl[:, 1], marker=".", alpha=0.7, label='Класс 0')
plt.scatter(class1_nl[:, 0], class1_nl[:, 1], marker=".", alpha=0.7, label='Класс 1')
plt.title('Нелинейный датасет: вариант 10')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.legend()
plt.savefig('scatter_nonlinear.png')
plt.show()