from base.base_block import BaseModel
from base.dense_block import DenseModel
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
    price_model = BaseModel(feature, item, input_shape=X1_train.shape[1:],
                            scaler=scaler1,
                            n_cells=50,
                            n_lags=n_lags,
                            n_features=n_features)
    price_model.train(X1_train, y1_train, lr=lr,
                      validation_split=0.1, n_epochs=n_epochs, batch_size=120)
    price_model.predict(X1_test, y1_test)
    print("Base block for price ready!")
    # Train the volume model
    print("Start traning base block for vol...")
    n_lags = 1
    n_features = 1
    feature = 'vol'
    X2_train, y2_train, X2_test, y2_test, scaler2 = data.prepare_data(item=item,
                                                                      feature=feature,
                                                                      n_lags=n_lags,
                                                                      n_features=n_features)
    vol_model = BaseModel(feature, item, input_shape=X2_train.shape[1:],
                          scaler=scaler2,
                          n_cells=50,
                          n_lags=n_lags,
                          n_features=n_features)

    vol_model.train(X2_train, y2_train, lr=lr,
                    validation_split=0.1, n_epochs=n_epochs, batch_size=120)
    vol_model.predict(X2_test, y2_test)
    print("Base block for vol ready!")
    # Train the dense block
    print("Training the dense block started...")
    n_lags = 1
    n_features = 1
    X_train = reshape_2d(np.c_[X1_train, X2_train])
    X_test = reshape_2d(np.c_[X1_test, X2_test])
    model = DenseModel(item, base_blocks=[price_model, vol_model], layers=[100,50],
                       scaler=scaler1,
                       n_lags=n_lags,
                       n_features=n_features)

    model.train(X_train, y1_train, lr=lr,
                validation_split=0.1, n_epochs=n_epochs, batch_size=120)
    print("Training finished!")
    model.predict(X_test, y1_test)
