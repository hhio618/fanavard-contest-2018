import tensorflow as tf
import datetime
from math import sqrt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from numpy import concatenate
from pandas import DataFrame, concat
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from keras.optimizers import RMSprop
import keras
from keras import backend as K
from keras.activations import elu, relu
from scipy.ndimage.interpolation import shift

import matplotlib
gui_env = ['Agg', 'TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
for gui in gui_env:
    try:
        print "Testing matplotlib backend...", gui
        matplotlib.use(gui, warn=False, force=True)
        matplotlib.interactive(False)
        from matplotlib import pyplot

        break
    except Exception as e:
        continue


def pi_score(y_true, y_pred):
    sorat = sum((y_pred-y_true)**2)
    shifted = shift(y_true, 1, cval=y_true[0])
    makhraj = sum((y_true-shifted)**2)
    return float(1-sorat/makhraj)


class PriceModel(object):
    def __init__(self, item, input_shape, scaler, n_cells ,n_lags,
                       n_features):
        self.input_shape = input_shape
        self.scaler = scaler
        self.item = item
        self.n_cells = n_cells
        self.n_lags=n_lags
        self.n_features=n_features

    def _build_model(self, lr):
        # design network
        model = Sequential()
        model.add(LSTM(self.n_cells, input_shape=(self.input_shape)))
        model.add(Dense(1, activation=relu))

        optm = RMSprop(lr=lr)
        model.compile(loss='mse', optimizer=optm)
        return model

    def _train(self, model, X_train, y_train, validation_split, n_epochs, batch_size):
        # fit network
        self.history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size,
                                 validation_split=validation_split, verbose=2, shuffle=False)
        model.save('output/models/%s.model' % self.item, overwrite=True)
        return model

    def train(self, X_train, y_train, lr=0.0003, validation_split=0.1, n_epochs=500, batch_size=120):
        _model = self._build_model(lr)
        self.model = self._train(_model,  X_train, y_train,
                    validation_split, n_epochs, batch_size)
        return self

    def plot_history(self):
        pyplot.figure()
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='validation')
        pyplot.xlabel('n_epochs')
        pyplot.ylabel('mse error')
        pyplot.title('errors during training')
        pyplot.legend()
        pyplot.savefig("output/price_train.png")

    def predict(self, X_test, y_test):
        # make a prediction
        yhat = self.model.predict(X_test)
        X_test = X_test.reshape((X_test.shape[0], self.n_lags * self.n_features))
        # invert scaling for forecast
        inv_yhat = concatenate((X_test, yhat), axis=1)
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[self.n_lags:, -1]
        # invert scaling for actual
        y_test = y_test.reshape((len(y_test), 1))
        inv_y = concatenate((X_test, y_test), axis=1)
        inv_y = self.scaler.inverse_transform(inv_y)
        inv_y = inv_y[:-self.n_lags, -1]
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

        pyplot.figure()
        pyplot.plot(inv_y, color='black', label='Original data')
        pyplot.plot(inv_yhat, color='blue', label='Predicted data')
        pyplot.legend(loc='best')
        pyplot.title('actual and predicted for last 20% of data')
        pyplot.xlabel('price index')
        pyplot.ylabel('price')
        pyplot.savefig("output/price_predict.png")
