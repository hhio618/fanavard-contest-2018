import tensorflow as tf
import datetime
from math import sqrt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout,LeakyReLU
from keras.models import Sequential
from numpy import concatenate
from pandas import DataFrame, concat
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from keras.optimizers import RMSprop, SGD, Adam
import keras
from keras import backend as K
from keras.activations import elu, relu
from scipy.ndimage.interpolation import shift
import random as rn
import matplotlib
import os
import math
import sys
from keras import backend as K
gui_env = ['Agg', 'TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
for gui in gui_env:
    try:
        print("Testing matplotlib backend...", gui)
        matplotlib.use(gui, warn=False, force=True)
        matplotlib.interactive(False)
        from matplotlib import pyplot

        break
    except Exception as e:
        continue
from datetime import datetime


def pi_score(y_true, y_pred):
    sorat = sum((y_pred-y_true)**2)
    shifted = shift(y_true, 1, cval=y_true[0])
    makhraj = sum((y_true-shifted)**2)
    return float(1-sorat/makhraj)


class ComplexModel(object):
    def __init__(self, item, input_shape,output_size, scaler, n_cells, n_lags,
                 n_features):
        # model seeding ########################################################
        seed = hash(item) & 0xffffffff
        os.environ['PYTHONHASHSEED'] = item

        np.random.seed(seed)
        rn.seed(seed)
        ########################################################################

        self.input_shape = input_shape
        self.output_size = output_size
        self.scaler = scaler
        self.item = item
        self.n_cells = n_cells
        self.n_lags = n_lags
        self.n_features = n_features

    def _build_model(self, lr):
        # design network
        model = Sequential()

        model.add(LSTM(self.n_cells,activation='tanh', input_shape=self.input_shape,return_sequences=False))
        model.add(Dropout(0.8))
        model.add(Dense(self.output_size))
        model.add(LeakyReLU())

        optm = RMSprop(lr=lr)
        model.compile(loss='mse', optimizer=optm)
        return model

    def _train(self, model, X_train, y_train, validation_split, n_epochs, batch_size):
        # fit network
        self.history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size,
                                 validation_split=validation_split, verbose=2, shuffle=False)
        model.save('output/models/%s.model' %
                   self.item, overwrite=True)
        return model

    def train(self, X_train, y_train, lr=0.0003, validation_split=0.1, n_epochs=500, batch_size=120):
        _model = self._build_model(lr)
        try:
            print("Try load model from cache...")
            self.model = keras.models.load_model('output/models/%s.model' % self.item)
            print("Model loaded")
        except Exception as e:
            print("Training a new model...")
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
        pyplot.savefig("output/figures/%s_loss.png" % self.item)

    def predict(self, X_test, y_test):
        # make a prediction
        yhat = self.model.predict(X_test)

        # # X_test for forecasting
        inv_yhat = np.ravel(yhat)
        # invert scaling for actual
        inv_y = np.ravel(y_test)
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        mae = mean_absolute_error(inv_y, inv_yhat)
        rmae = sqrt(mae)
        r2 = r2_score(inv_y, inv_yhat)
        pi = pi_score(inv_y, inv_yhat)
        report = []
        report += ['Test RMSE: %.3f' % rmse]
        report += ['Test mae:  %.3f' % mae]
        report += ['Test rmae:  %.3f' % rmae]
        report += ['Test r2:  %.3f' % r2]
        report += ['Test pi:  %.3f' % pi]
        report = '\r\n'.join(report)
        print(report)
        with open("output/models/%s.txt" % self.item, 'w') as outf:
            outf.write(report)

        pyplot.figure()
        pyplot.plot(inv_y, color='black', label='Original data')
        pyplot.plot(inv_yhat, color='blue', label='Predicted data')
        pyplot.legend(loc='best')
        pyplot.title('actual and predicted for last 20% of data')
        pyplot.xlabel('price index')
        pyplot.ylabel("price")
        pyplot.savefig("output/figures/%s.png" % self.item)
