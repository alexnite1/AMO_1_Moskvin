import numpy as np
import random
import os
import matplotlib.pyplot as plt

# Создание директорий train и test
if not os.path.exists('train'):
    os.makedirs('train')

if not os.path.exists('test'):
    os.makedirs('test')

# Создание простейших данных о дневной температуре с помощью функции np.sin
# Для добавления шума воспользуемся функцией random.uniform
# Первые 20 значений каждого набора данных будут с шумом

x = np.arange(0, 100, 1)  # дата в днях
y_list = []

for i in range(5):
    y = np.sin(x/5) + np.random.normal(loc=0, scale=0.1, size=len(x))
    y[:20] += np.random.normal(loc=0, scale=1, size=20)  # добавление шума в первые 20 дней
    y_list.append(y)

# визуализация первых трех сгенерированных наборов данных
plt.plot(x, y_list[0])
plt.plot(x, y_list[1])
plt.plot(x, y_list[2])
plt.xlabel('День')
plt.ylabel('Температура')
plt.legend(['Набор данных 1', 'Набор данных 2', 'Набор данных 3'])
plt.savefig('train/data_vis.png')  # сохранение графика в папку train
plt.show()

# Сохранение данных в папки train и test
for i, y in enumerate(y_list):
    if i < 3:
        np.savetxt(f'train/data{i+1}.csv', np.column_stack([x, y]), delimiter=',')
    else:
        np.savetxt(f'test/data{i+1}.csv', np.column_stack([x, y]), delimiter=',')
