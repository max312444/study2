import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 및 분할
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# 2. 훈련/테스트 셋 분리 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 3. 특성 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

w = np.random.randn(X_train.shape[1], 1)
b = np.random.randn()
learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):
    # prediction
    # z = w * x + b
    z = X_train @ w + b

    # sigmoid(z) -> 1 / (1+e^-z)
    prediction = 1 / (1 + np.exp(-z))

    # Error
    error = prediction - y_train

    # Get gradient for w and b
    gradient_w = X_train.T @ error / len(X_train)
    gradient_b = error.mean()

    # Update parameter values of each w and b
    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    # Display the loss value of the current epoch
    if epoch % 1000 == 0:
        loss = -y_train*np.log(prediction + 1e-15) - ( 1 - y_train)*np.log(1-prediction + 1e-15)
        print(loss.mean())
        
        
# w -> 30개와 b 값
np.set_printoptions(suppress=True, precision=15)
test_z = X_test @ w + b
test_prediction = 1 / (1 + np.exp(-test_z))
test_result = (test_prediction >= 0.5).astype(int)

accuracy = np.mean((test_result == y_test).astype(int))

print(accuracy)
