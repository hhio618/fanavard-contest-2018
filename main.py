from keras.models import load_model
from sklearn.externals import joblib
import numpy as np


models = []
scalers = []

model_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

models_dir = 'models'
scalers_dir = 'scalers'

for name in model_names:
    models.append(load_model(models_dir + '/' + name + '.model'))
    scalers.append(joblib.load(scalers_dir + '/' + name + '.scaler'))


lines = int(input())

indexes = [[12], [651], [1290], [1929], [2568], [3207], [3846], [4485], [5124]]

for i in range(119):
    for j in range(len(indexes)):
        indexes[j].append(indexes[j][-1] + 4)

for i in range(lines):
    data = input().strip().split()
    input_data = []

    for j in range(len(indexes)):
        jdata = [int(data[x]) for x in indexes[j]]
        input_data.append(float(sum(jdata)/len(jdata)))

    for j in range(len(models)):
        scaled = scalers[j].transform(np.concatenate(([input_data[j]], [1])).reshape((1,2)))
        m_input = scaled[0,0].reshape((1,1,1))
        for k in range(10):
            m_input = models[j].predict(m_input.reshape((1,1,1)))
            inversed = scalers[j].inverse_transform(np.concatenate((m_input[0], [1])).reshape((1,2)))
            print(str(inversed[0,1]) + ' ', end='')
    print()
    



