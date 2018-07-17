import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = tf.Session(config=config)
set_session(session)

import datetime
from math import sqrt

import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from matplotlib import pyplot
from numpy import concatenate
from pandas import DataFrame, concat
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from keras.optimizers import RMSprop
import keras
from keras import backend as K
from keras.activations import elu,relu
from scipy.ndimage.interpolation import shift


n_lags = 1
n_features = 1
n_epochs = 500
batch_size = 120
save_path = 'drive/fan/models/A.model'
data_path = 'drive/fan/data/A_mean.csv'
scaler_path = 'drive/fan/scalers/A.scaler'
validation_split = 0.1
optimizer_lr = 0.0003
n_cells = 50
train_perent = 0.9


def pi_score(y_true, y_pred):
    sorat = sum((y_pred-y_true)**2)
    shifted = shift(y_true, 1, cval=y_true[0])
    makhraj = sum((y_true-shifted)**2)
    return float(1-sorat/makhraj)

def remove_outliers(data):
    for i in range(data.shape[0]):
        if data[i] == 0:
            b = i - 1
            f = i + 1
            while(data[b] == 0):
                b -= 1
            while(data[f] == 0):
                f += 1
            data[i] = (data[b] + data[f])/2
    return data

def series_to_supervised(df, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    	data: Sequence of observations as a list or NumPy array.
    	n_in: Number of lag observations as input (X).
    	n_out: Number of observations as output (y).
    	dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    	Pandas DataFrame of series framed for supervised learning.
    """
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
    	cols.append(df.shift(i))
    	names += [('%s(t-%d)' % (j, i)) for j in df.columns.values]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
    	cols.append(df.shift(-i))
    	if i == 0:
    		names += [('%s(t)' % (j)) for j in df.columns.values]
    	else:
    		names += [('%s(t+%d)' % (j, i)) for j in df.columns.values]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
    	agg.dropna(inplace=True)
    return agg



def parser(x):
    return datetime.datetime.fromtimestamp(float(x))

def parser2(x):
    return datetime.datetime.strptime(x)


df = pd.read_csv(data_path, date_parser=parser, parse_dates=['date'], index_col=0)
df['price'] = remove_outliers(df['price'])
sale_df = df[['price']]
lagged_sale_df = series_to_supervised(sale_df, n_lags, 1)

if n_features > 1:
    data = lagged_sale_df.values[:,:-(n_features-1)]
else:
    data = lagged_sale_df.values

scaler = MinMaxScaler()
# data = scaler.fit_transform(data)

trains_size = int(data.shape[0]*train_perent)

d_train = data[:trains_size, :]
scaler = scaler.fit(d_train)
joblib.dump(scaler, scaler_path)
d_train = scaler.transform(d_train)
X_train = d_train[:,:-1]
y_train = d_train[:,-1]

d_test = data[trains_size:, :]
d_test = scaler.transform(d_test)
X_test = d_test[:, :-1]
y_test = d_test[:, -1]


X_train = X_train.reshape((X_train.shape[0], n_lags, n_features))
X_test = X_test.reshape((X_test.shape[0], n_lags, n_features))


# design network
model = Sequential()
model.add(LSTM(n_cells, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation=elu))

optm = RMSprop(lr=optimizer_lr)
model.compile(loss='mse', optimizer=optm)

# fit network
history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=validation_split, verbose=2, shuffle=False)
model.save(save_path, overwrite=True)

# plot history
pyplot.figure(1)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.xlabel('n_epochs')
pyplot.ylabel('mse error')
pyplot.title('errors during training')
pyplot.legend()


# make a prediction
yhat = model.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], n_lags*n_features))
# invert scaling for forecast
inv_yhat = concatenate((X_test, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[n_lags:,-1]
# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = concatenate((X_test, y_test), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:-n_lags,-1]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
mae = mean_absolute_error(inv_y, inv_yhat)
rmae = sqrt(mae)
r2 = r2_score(inv_y, inv_yhat)
pi = pi_score(inv_y, inv_yhat)
print('Test RMSE: %.3f' % rmse)
print('Test mae:  %.3f' % mae)
print('Test rmae:  %.3f' % rmae)
print('Test r2:  %.3f' % r2)
print('Test pi:  %.3f' % pi)


pyplot.figure(2)
pyplot.plot(inv_y, color='black', label = 'Original data')
pyplot.plot(inv_yhat, color='blue', label = 'Predicted data')
pyplot.legend(loc='best') 
pyplot.title('actual and predicted for last 20% of data') 
pyplot.xlabel('price index')
pyplot.ylabel('price')
pyplot.show()