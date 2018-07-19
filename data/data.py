import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib


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
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
    	agg.dropna(inplace=True)
    return agg


def parser(x):
    return datetime.datetime.fromtimestamp(float(x))


def parser2(x):
    return datetime.datetime.strptime(x)


def prepare_data(item='A', feature='price', n_features=1, n_lags=1 ,train_perent = 0.9):
        scaler_path = 'data/scalers/%s.scaler' % item
        data_path = 'data/csv/%s_mean.csv' % item
        df = pd.read_csv(data_path, date_parser=parser,
                         parse_dates=['date'], index_col=0)
        df['price'] = remove_outliers(df[feature])
        sale_df = df[[feature]]
        lagged_sale_df = series_to_supervised(sale_df, n_lags, 1)

        if n_features > 1:
            data = lagged_sale_df.values[:, :-(n_features-1)]
        else:
            data = lagged_sale_df.values

        scaler = MinMaxScaler()
        # data = scaler.fit_transform(data)

        trains_size = int(data.shape[0]*train_perent)
        d_train = data[:trains_size, :]
        scaler = scaler.fit(d_train)
        joblib.dump(scaler, scaler_path)
        d_train = scaler.transform(d_train)
        X_train = d_train[:, :-1]
        y_train = d_train[:, -1]

        d_test = data[trains_size:, :]
        d_test = scaler.transform(d_test)
        X_test = d_test[:, :-1]
        y_test = d_test[:, -1]
        X_train = X_train.reshape((X_train.shape[0], n_lags, n_features))
        X_test = X_test.reshape((X_test.shape[0], n_lags, n_features))
        return X_train, y_train, X_test, y_test, scaler
