import numpy as np
import os

# загрузка данных из папки test
test_data = np.loadtxt('test/data4.csv', delimiter=',')

# загрузка обученной модели из папки train
model_params = np.loadtxt('train/trained_model.csv', delimiter=',')
coef = model_params[0]
intercept = model_params[1]

# проверка модели на данных из папки test
predicted = coef * test_data[:, 0] + intercept

# сравнение предсказаний модели с оригинальными данными
error = np.mean((predicted - test_data[:, 1])**2)

print(f"Среднеквадратичная ошибка: {error}")