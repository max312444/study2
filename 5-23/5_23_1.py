import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# 훈련/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 스케일링 (입력값만)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# y_train은 2차원으로 reshape (행렬 연산을 위해)
y_train = y_train.reshape(-1, 1)

# 초기 가중치와 편향 설정
w = np.random.randn(X_train.shape[1], 1)  # (30, 1)
b = np.random.randn()  # 스칼라
learning_rate = 0.01
epochs = 10000


# 선형 조합
z = X_train @ w + b  # (n_samples, 1)

# 시그모이드 함수
prediction = 1 / (1 + np.exp(-z))  # (n_samples, 1)

# 오차 (예측값 - 실제값)
error = prediction - y_train  # (n_samples, 1)

gradient_w = X_train.T @ error / len(X_train)
# print(gradient_w.shape)
gradient_b = error.mean()

w -= learning_rate * gradient_w
b -= learning_rate * gradient_b

loss = -y_train * np.log(prediction + 1e-15) - (1 - y_train)*np.log(1 - prediction + 1e-15)
print(loss.mean())