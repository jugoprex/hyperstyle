import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.concatenate((np.array([[3,6], [8,5], [2, 2]]),np.array([[3,8]])), axis=0)
reg = LinearRegression().fit(X, y)
print(y)

print(reg.predict(np.array([[3, 5]])))
