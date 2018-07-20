from base.complex_block import ComplexModel
from data import data
import sys
import numpy as np


def reshape_2d(X):
    return X.reshape((X.shape[0],X.shape[2]))

def run(item ,n_epochs, lr):
    # Train the price model
    print("Start traning base block for price...")
    n_lags = 1
    n_features = 1
    feature = 'price'
    X1_train, y1_train, X1_test, y1_test, scaler1 = data.prepare_data(item=item,
                                                                      feature=feature,
                                                                      n_lags=n_lags,
                                                                      n_features=n_features)
    feature = 'vol'
    X2_train, y2_train, X2_test, y2_test, scaler2 = data.prepare_data(item=item,
                                                                      feature=feature,
                                                                      n_lags=n_lags,
                                                                      n_features=n_features)
    # Using histogram features
    feature = 'f1'
    X3_train, y3_train, X3_test, y3_test, scaler3 = data.prepare_data_new(item=item,
                                                                      feature=feature,
                                                                      n_lags=n_lags,
                                                                      n_features=n_features)
    feature = 'f2'
    X4_train, y4_train, X4_test, y4_test, scaler4 = data.prepare_data_new(item=item,
                                                                      feature=feature,
                                                                      n_lags=n_lags,
                                                                      n_features=n_features)
    X_train = np.c_[X1_train,X3_train,X4_train]
    X_test = np.c_[X1_test, X3_test, X4_test]
    model = ComplexModel(item, input_shape=X_train[0].shape,
                            scaler=scaler1,
                            n_cells=50,
                            n_lags=n_lags,
                            n_features=n_features)
    model.train(X_train, y1_train, lr=lr,
                      validation_split=0.1, n_epochs=n_epochs, batch_size=120)
    model.predict(X_test, y1_test)
    print("Base block for price ready!")
