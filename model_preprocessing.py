import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# загрузка одного из наборов данных из папки train
data = np.loadtxt('train/data1.csv', delimiter=',')

# стандартизация данных с помощью StandardScaler
scaler = StandardScaler()
scaler.fit(data[:, 1].reshape(-1, 1))
data[:, 1] = scaler.transform(data[:, 1].reshape(-1, 1)).ravel()

# визуализация оригинальных и стандартизированных данных
plt.plot(data[:, 1])
plt.plot(scaler.inverse_transform(data[:, 1].reshape(-1, 1)))
plt.xlabel('День')
plt.ylabel('Температура')
plt.legend(['Оригинальные данные', 'Стандартизированные данные'])
plt.savefig('train/preprocessed_data_vis.png')
plt.show()

# сохранение предобработанных данных в папку train
np.savetxt('train/preprocessed_data.csv', data, delimiter=',')