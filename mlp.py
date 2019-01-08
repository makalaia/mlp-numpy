import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from utils import diff_sigmoid, sigmoid, mse


def glorot_initializer(fan_in, fan_out):
    factor = 4 * np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-factor, factor, (fan_in, fan_out))


class MLP(object):
    def __init__(self, layers, weights_initializer=glorot_initializer, error_function=mse):
        self.num_layers = len(layers)
        self.layers = layers
        self.biases = [np.zeros((y, 1)) for y in layers[1:]]
        self.weights = [weights_initializer(x, y) for x, y in zip(layers[:-1], layers[1:])]
        self.error_function = error_function
        self.zs, self.activations = list(), list()

    def forwardprop(self, x):
        self.zs.clear()
        self.activations.clear()

        activation = x
        self.activations.append(activation)
        for i in range(self.num_layers - 2):
            z = np.dot(self.weights[i].T, activation.T) + self.biases[i]
            z = z.T
            self.zs.append(z)
            activation = sigmoid(z)
            self.activations.append(activation)
        z = np.dot(self.weights[self.num_layers - 2].T, activation.T) + self.biases[self.num_layers - 2]  # linear output
        z = z.T
        self.zs.append(z)
        activation = z
        self.activations.append(activation)

    def predict(self, x):
        x = x.T
        for i in range(self.num_layers - 2):
            x = np.dot(self.weights[i].T, x) + self.biases[i]
            x = sigmoid(x)
        x = np.dot(self.weights[self.num_layers - 2].T, x) + self.biases[self.num_layers - 2]  # linear output
        x = x.T
        return x

    def backprop(self, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        # backward pass
        dlda = self.error_function(y, self.activations[-1], diff=True)
        grad_b[-1] = dlda.sum(axis=0, keepdims=True).T
        grad_w[-1] = np.dot(dlda.T, self.activations[-2]).T  # linear output
        for i in range(2, self.num_layers):
            z = self.zs[-i]
            dadz = diff_sigmoid(z)
            dlda = np.dot(self.weights[-i + 1], dlda.T).T * dadz
            grad_b[-i] = dlda.sum(axis=0, keepdims=True).T
            dzdw = self.activations[-i - 1]
            grad_w[-i] = np.dot(dlda.T, dzdw).T
        return grad_b, grad_w

    def SGD(self, training_data, epochs, batch_size, lr, val_data=None, shuffle=True):
        x_train, y_train = training_data
        loss = np.zeros(epochs)
        m = len(y_train)
        for epoch in range(epochs):
            if shuffle:
                rng = np.random.permutation(m)
                x_train, y_train = x_train[rng], y_train[rng]
            batches = [(x_train[i:i + batch_size], y_train[i:i + batch_size]) for i in range(0, m, batch_size)]

            for batch in batches:
                self.update_batch(batch, lr)
            if val_data:
                error = self.evaluate(val_data)
                print('Epoch {epoch} - VAL_ERROR: {error}'.format(epoch=epoch, error=error))
                loss[epoch] = error
            else:
                print('Epoch {epoch} complete'.format(epoch=epoch))
        if val_data:
            return loss

    def update_batch(self, batch, lr):
        x, y = batch
        m = len(y)

        # get gradients
        self.forwardprop(x)
        grad_b, grad_w = self.backprop(y)

        # update weights
        self.weights = [weights - (lr / m) * gw for weights, gw in zip(self.weights, grad_w)]
        self.biases = [b - (lr / m) * biases for b, biases in zip(self.biases, grad_b)]

    def evaluate(self, val_data):
        x, y = val_data
        y_pred = self.predict(x)
        return self.error_function(y, y_pred)


if __name__ == '__main__':
    val_size = 120
    test_size = 60
    epochs = 300
    df = pd.read_csv('data/test.csv', index_col=0)

    y_total = df.iloc[:, -1:].values
    x_total = df.iloc[:, :-1].values
    y_test = y_total[-test_size:, :]
    x_test = x_total[-test_size:, :]
    y_train = y_total[:-val_size - test_size, :]
    x_train = x_total[:-val_size - test_size, :]
    y_val = y_total[-val_size - test_size - 1:-test_size, :]
    x_val = x_total[-val_size - test_size - 1:-test_size, :]
    n_samples = x_train.shape[0]

    scalerX = RobustScaler(quantile_range=(10, 90))
    scalerY = RobustScaler(quantile_range=(10, 90))
    x_train = scalerX.fit_transform(x_train)
    y_train = scalerY.fit_transform(y_train)
    x_val = scalerX.transform(x_val)
    y_val = scalerY.transform(y_val)
    x_test = scalerX.transform(x_test)
    y_test = scalerY.transform(y_test)

    input_size = x_total.shape[1]
    output_size = y_total.shape[1]
    training_data = (x_train, y_train)
    val_data = (x_val, y_val)
    test_data = (x_test, y_test)

    mlp = MLP([input_size, 128, 64, 32, output_size])
    loss = mlp.SGD(training_data, epochs=epochs, batch_size=32, lr=0.01, val_data=val_data, shuffle=False)

    plt.plot(y_train)
    plt.plot(mlp.predict(x_train))
    plt.title('TRAINING DATA')
    plt.show()
    #
    plt.plot(loss)
    plt.title('VAL_LOSS x EPOCHS')
    plt.show()

    print('TEST ERROR: {error}'.format(error=mlp.evaluate(val_data=test_data)))
