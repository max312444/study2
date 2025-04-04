import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# x: 0~10 사이 무작위 수 100개
# y: 2.5x + 약간의 노이즈
np.random.seed(0)
X = np.random.rand(3, 1) * 10
y = 2.5 * X + np.random.randn(3, 1) * 2
y = y.ravel() # SGDRegressor는 1차원 타겟값을 요구

# bar = np.random.rand(3, 1)
# print(bar)
# print("--" * 10)
# print(bar.ravel())

# 모델 생성 후 하이퍼파라메터 설정
model = SGDRegressor(max_iter=1000, # 학습 반복 횟수 (epoch 수)
                     learning_rate='constant',
                     eta0=0.01,    # 고정 학습률
                     penalty=None, # 정규화 없음
                     random_state=0)

model.fit(X, y) # 모델 학습

# 평가
# Loss
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
print(mse)