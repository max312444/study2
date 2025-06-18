from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터셋 로딩 및 분할
digits = load_digits()
features = digits.data                    # (1797, 64): 8x8 이미지 벡터
labels = digits.target                    # (1797,): 0~9 클래스 정수

# 2. 학습/테스트 셋 분할
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

np.set_printoptions(suppress=True)

# 4. 변수 정의
num_features = X_train_std.shape[1]
num_samples = X_train_std.shape[0]
num_classes = 10

W = np.random.randn(num_features, num_classes)
b = np.zeros(num_classes)

learning_rate = 0.05
epochs = 1000  # 10000은 너무 길 수 있으니 테스트용으로 1000

# One-hot 인코딩
I_matrix = np.eye(num_classes)
one_hot = I_matrix[y_train]  # (1437, 10)

# 5. 학습 루프
for epoch in range(epochs):
    # forward
    logit = X_train_std @ W + b  # (1437, 10)
    logit -= np.max(logit, axis=1, keepdims=True)  # stability
    exp_logit = np.exp(logit)
    softmax = exp_logit / np.sum(exp_logit, axis=1, keepdims=True)  # (1437, 10)

    # loss (크로스 엔트로피)
    loss = -np.sum(one_hot * np.log(softmax + 1e-8)) / num_samples  # 작은 수 더해 안정화

    # backward
    error = softmax - one_hot  # (1437, 10)
    gradient_w = X_train_std.T @ error / num_samples  # (64, 10)
    gradient_b = np.sum(error, axis=0) / num_samples  # (10,)

    # update
    W -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
