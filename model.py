from base.complex_block import ComplexModel
from data import data
import sys
import numpy as np
from sklearn.externals import joblib


def reshape_2d(X):
    return X.reshape((X.shape[0], X.shape[2]))


def run(item, n_epochs, lr):
    # Train the price model
    n_lags = 50
    n_features = 5
    output_size = 10
    print("Start traning base block for prediction...")
    X_train, y_train, X_test, y_test = data.prepare_data_new(item=item,
                                                             n_lags=n_lags,
                                                             n_features=n_features)
    n_lags = 20
    Xhat = X_train[:,:n_lags]
    yhat = y_train
    X_t = X_test[:,:n_lags]
    y_t = y_test

    scaler = joblib.load("data/scalers/%scost.scaler" % item)
    model = ComplexModel(item, input_shape=Xhat.shape[1:],output_size=output_size,
                         scaler=scaler,
                         n_cells=50,
                         n_lags=n_lags,
                         n_features=n_features)
    model.train(Xhat,yhat , lr=lr,
                validation_split=0.2, n_epochs=n_epochs, batch_size=100)
    model.predict(X_t, y_t)
    print("Base block for prediction ready!")
