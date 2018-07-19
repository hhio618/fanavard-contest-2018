import numpy as np
from scipy.ndimage.interpolation import shift

a = [[1, 2], [3,4]]
b = [[1],[3]]

print(np.hstack((a, b)))



def pi_score(y_pred, y_true):
    sorat = sum((y_pred-y_true)**2)
    shifted = shift(y_true, 1, cval=y_true[0])
    makhraj = sum((y_true-shifted)**2)
    return float(1-sorat/makhraj)


y_pred = np.array([1, 2, 3])
y_true = np.array([1, 2, 5])

print('actual and predicted for last 20% of data')

a = 1
print(np.concatenate(([a], [1]))[0].reshape((1,1,1)))
