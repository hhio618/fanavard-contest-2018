from base.price_model import PriceModel
from data import data
import sys


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: main.py <item>\r\nItems are A,B,...")
        sys.exit(0)
    item = sys.argv[1]
    n_lags = 1
    n_features = 1
    X_train, y_train, X_test, y_test, scaler = data.prepare_data(item=item, n_lags=n_lags,
                                                                 n_features=n_features)
    model = PriceModel(item, input_shape=X_train.shape[1:],
                       scaler=scaler,
                       n_cells=50,
                       n_lags=n_lags,
                       n_features=n_features)
    model.train(X_train, y_train, lr=0.003,
                validation_split=0.1, n_epochs=500, batch_size=120)
    model.predict(X_test, y_test)
