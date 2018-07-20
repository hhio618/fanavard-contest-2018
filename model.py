from base.complex_block import ComplexModel
from data import data
import sys
import numpy as np



def reshape_2d(X):
    return X.reshape((X.shape[0],X.shape[2]))

def run(item ,n_epochs, lr):
    # Train the price model
    n_lags = 256
    n_features = 3
    print("Start traning base block for prediction...")
    X_train, y_train, X_test, y_test, scaler = data.prepare_data_new(item=item,
                                                                      n_lags=n_lags,
                                                                      n_features=n_features)
    
    model = ComplexModel(item, input_shape=X_train.shape[1:],
                            scaler=scaler,
                            n_cells=50,
                            n_lags=n_lags,
                            n_features=n_features)
    model.train(X_train, y_train, lr=lr,
                      validation_split=0.2, n_epochs=n_epochs, batch_size=8)
    model.predict(X_test, y_test)
    print("Base block for prediction ready!")
