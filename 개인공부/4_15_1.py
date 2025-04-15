import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# X : 0~10 사이 무작위 수 100개
# y : 2.5x + 약간의 노이즈
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + np.random.rand(100, 1) * 2
y = y.ravel()

model = SGDRegressor(max_iter=1000,
                     learning_rate='constant',
                     eta0=0.01,
                     penalty=None,
                     random_state=0)

model.fit(X, y)

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
print(f"평균 제곱 오차(MSE): {mse:.4f}")

x_line = np.linspace(0, 10, 100).reshape(-1, 1)
y_line = model.predict(x_line)


plt.scatter(X, y, label='Data', alpha=0.6)
plt.plot(x_line, y_line, color='red', label='SGD Regression Line')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regresstion with SGDRegressor")
plt.legend()
plt.grid(True)
plt.show()