import os
import random as rn
import time

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from pandas import read_csv
from sklearn.preprocessing import RobustScaler

val_size = 120
test_size = 60
df = read_csv('data/test.csv')
df.set_index(list(df)[0], inplace=True)
columns = list(df)

y_total = df.iloc[:, -1:].values
x_total = df.iloc[:, :-1].values
y_test = y_total[-test_size:, :]
x_test = x_total[-test_size:, :]
y_train = y_total[:-val_size - test_size, :]
x_train = x_total[:-val_size - test_size, :]
y_val = y_total[-val_size - test_size - 1:-test_size, :]
x_val = x_total[-val_size - test_size - 1:-test_size, :]
n_samples = x_train.shape[0]
m = len(y_train)

scalerX = RobustScaler(quantile_range=(10, 90))
scalerY = RobustScaler(quantile_range=(10, 90))
x_train = scalerX.fit_transform(x_train)
y_train = scalerY.fit_transform(y_train)
x_val = scalerX.transform(x_val)
y_val = scalerY.transform(y_val)
x_test = scalerX.transform(x_test)
y_test = scalerY.transform(y_test)
tempo = time.time()
epochs = 200
learning_rate = 0.01
batch_size = m

# random seed
os.environ['PYTHONHASHSEED'] = '0'
seed = 123456
if seed is not None:
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)
    K.set_session(tf.Session(graph=tf.get_default_graph()))

# Neural net
epochs = 200
optmizer = keras.optimizers.SGD(lr=learning_rate)
model = Sequential()
model.add(Dense(128, input_shape=(x_train.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# fit
model.compile(loss='mean_squared_error', optimizer=optmizer)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=2)
print('TIME: ' + str(time.time() - tempo))

# predict
y_trained = model.predict(x_train)
y_validated = model.predict(x_val)
y_tested = model.predict(x_test)

plt.plot(y_train)
plt.plot(y_trained)
plt.show()

