from model import run
import sys
import keras
import numpy as np
from data import data


def x_from_input():
    x = np.asarray(sys.argv[2],sys.argv[4],sys.argv[6],sys.argv[8], sys.argv[9])
    return x


if __name__ == '__main__':
    if len(sys.argv) < 100:
        print("Usage: main.py < item > <data-seperated-by-space >  # items are : A,B,...")
        sys.exit(0)
    item = sys.argv[1]
    model = keras.models.load_model("output/models/%s.model" % item)
    n_lags = 50
    n_features = 5
    print("Start traning base block for prediction...")
    _, _, X_test, _, scaler = data.prepare_data_new(item=item,
                                                    n_lags=n_lags,
                                                    n_features=n_features)
    X_in = np.append(X_test[-(n_lags-1):], x_from_input())
    y = model.predict(X_in)
    print(y)
