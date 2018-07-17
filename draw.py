import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np


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


def parser(x):
    return datetime.datetime.fromtimestamp(float(x))

data_path = 'data/A_ticker.csv'

df = pd.read_csv(data_path)
print(list(df))


x1 = df['date']
y1 = df['max_sale_price']
y1 = remove_outliers(y1)

plt.figure(1)
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_scientific(False)

y1 = df['daily_transactions_volume']


# plt.scatter(y1[1:], y2[:-1], s=1)
# plt.xlabel('price at t')
# plt.ylabel('max sale volume at t-1')

# print(np.corrcoef(y1, y2))

data_path = 'data/A_mean.csv'
df = pd.read_csv(data_path, date_parser=parser, parse_dates=['date'], index_col=0)
y2 = df['price']
y2 = remove_outliers(y2)
df['price'] = y2

print(np.corrcoef(y2[1:], y1[:-1]))

# plt.plot(x1, y1, 'r', label = 'ticker max price')
# plt.plot(x1, y2, 'b', label = 'mean of trades')
# plt.legend(loc='best')


# plt.figure(2)
# plt.scatter(y2[:-1], y1[1:], s=1)
# plt.xlabel('price at time t ')
# plt.ylabel('price at time t-1 ')


# plt.figure(3)
# plt.subplot(221)
# plt.scatter(y1[2:], y2[:-2], s=1)
# plt.xlabel('price at time t ')
# plt.ylabel('price at time t-2 ')
# plt.subplot(222)
# plt.scatter(y1[3:], y2[:-3], s=1)
# plt.xlabel('price at time t ')
# plt.ylabel('price at time t-3 ')
# plt.subplot(223)
# plt.scatter(y1[4:], y2[:-4], s=1)
# plt.xlabel('price at time t ')
# plt.ylabel('price at time t-4 ')
# plt.subplot(224)
# plt.scatter(y1[5:], y2[:-5], s=1)
# plt.xlabel('price at time t ')
# plt.ylabel('price at time t-5 ')

plt.show()


