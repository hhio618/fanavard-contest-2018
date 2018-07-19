import pandas as pd
import datetime
import numpy as np

data_path = 'data/I_trades.csv'
save_path = 'data/I_mean.csv'

def parser(x):
    return datetime.datetime.fromtimestamp(int(x))

header = []
remove_cols=[]
header.append('date')
for i in range(120):
    header.append('id'+str(i))
    header.append('date'+ str(i))
    header.append('vol'+str(i))
    header.append('price'+str(i))
    remove_cols.append('id'+str(i))
    remove_cols.append('date'+str(i))

df = pd.read_csv(data_path, names=header, parse_dates=['date'], index_col=0,date_parser=parser)
df.drop(columns=remove_cols, inplace=True)
df.index.name='date'

avg_data=[]
for i in range(df.shape[0]):
    prices = []
    volumes = []
    for j in range(0,240,2):
        if df.iloc[i,j] > 0:
            volumes.append(df.iloc[i,j])
        prices.append(df.iloc[i,j+1])
    avg_data.append([np.mean(prices), np.sum(volumes)])
avg_data = np.array(avg_data)
indexes = df.index.values

data = []
for i, d in zip(indexes, avg_data):
    ts = (i - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    data.append([int(ts), d[0], d[1]])

df = pd.DataFrame(data)
df.to_csv(save_path, index=False)
