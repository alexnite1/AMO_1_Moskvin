import numpy as np
import os

# �������� ������ �� ����� test
test_data = np.loadtxt('test/data4.csv', delimiter=',')

# �������� ��������� ������ �� ����� train
model_params = np.loadtxt('train/trained_model.csv', delimiter=',')
coef = model_params[0]
intercept = model_params[1]

# �������� ������ �� ������ �� ����� test
predicted = coef * test_data[:, 0] + intercept

# ��������� ������������ ������ � ������������� �������
error = np.mean((predicted - test_data[:, 1])**2)

print(f"������������������ ������: {error}")