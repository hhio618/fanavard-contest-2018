import datetime
from math import sqrt

import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from matplotlib import pyplot
from numpy import concatenate
from pandas import DataFrame, concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


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
    return datetime.datetime.fromtimestamp(int(x))

n_lags = 2
n_features = 1

data_path = 'data/A_ticker.csv'
df = pd.read_csv(data_path, date_parser=parser, parse_dates=['date'], index_col=0)

sale_df = df[['max_sale_price']]

lagged_sale_df = series_to_supervised(sale_df, n_lags, 1)

if n_features > 1:
    data = lagged_sale_df.values[:,:-(n_features-1)]
else:
    data = lagged_sale_df.values
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

trains_size = int(data.shape[0]*0.7)

X,y = data[:, :-1], data[:, -1]
X_train = X[:trains_size, :]
y_train = y[:trains_size]
X_test = X[trains_size:, :]
y_test = y[trains_size:]

X_train = X_train.reshape((X_train.shape[0], n_lags, n_features))
X_test = X_test.reshape((X_test.shape[0], n_lags, n_features))



# design network
model = Sequential()
model.add(LSTM(25, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

# fit network
history = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)
# plot history
pyplot.figure(1)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()



# make a prediction
yhat = model.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], n_lags*n_features))
# invert scaling for forecast
inv_yhat = concatenate((X_test, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = concatenate((X_test, y_test), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


pyplot.figure(2)
pyplot.plot(inv_y, color='black', label = 'Original data')
pyplot.plot(inv_yhat, color='blue', label = 'Predicted data')
pyplot.legend(loc='best') 
pyplot.title('Actual and predicted') 
pyplot.show()