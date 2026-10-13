import numpy as np
import model
import DataGenerator as dg

np.random.seed(42)
N = 1000


def runExperiment(X, Y, name, n_neuro=4, N_epoch=500, neurons_list=(1, 2, 4, 8, 16)):
    print(f'\nВыборка {name}')
    Y = np.reshape(Y, [2 * N, 1]).astype(float)
    Xtrain, Xtest, Ytrain, Ytest = model.dataPartition(X, Y, N)

    NN = model.NeuralNetwork(Xtrain, Ytrain, n_neuro=n_neuro, lr=2)
    history = model.trainNetwork(NN, Xtrain, Ytrain, Xtest, Ytest, N_epoch)
    model.plotLearning(history, name)

    pred_train = NN.test(Xtrain)
    pred_test = NN.test(Xtest)
    model.printAccSensSpec((pred_test > 0.5).astype(float), Ytest,
                           (pred_train > 0.5).astype(float), Ytrain)
    model.plotHist(pred_test, Ytest, pred_train, Ytrain, name)
    model.plotBoundary(NN, Xtest, Ytest, name)

    weights_hidden = NN.weights1.copy()
    weights_output = NN.weights2.copy()
    print('Веса скрытого слоя (weights1):')
    print(weights_hidden)
    print('Веса выходного слоя (weights2):')
    print(weights_output)

    model.tuneNeurons(Xtrain, Ytrain, Xtest, Ytest, neurons_list, N_epoch, name)

    return weights_hidden, weights_output


mu0 = [0, 2, 3]
mu1 = [3, 5, 1]
sigma0 = [2, 1, 2]
sigma1 = [1, 2, 1]
X, Y, class0, class1 = dg.norm_dataset([mu0, mu1], [sigma0, sigma1], N)
weights_A = runExperiment(X, Y, 'Хорошо разделимые')

mu0 = [0, 2, 3]
mu1 = [1, 3, 1]
sigma0 = [2, 1, 2]
sigma1 = [2, 2, 1]
X, Y, class0, class1 = dg.norm_dataset([mu0, mu1], [sigma0, sigma1], N)
weights_B = runExperiment(X, Y, 'Плохо разделимые')

X, Y, class0, class1 = dg.nonlinear_dataset_10(N, noise=0.15, seed=10)
weights_C = runExperiment(X, Y, 'Нелинейные', n_neuro=16, N_epoch=2000)