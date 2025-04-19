import numpy as np

# 🔹 (1) 가상의 X(입력)와 y(출력)
# X는 m개의 샘플과 n개의 특성 (여기선 4샘플, 3특성)
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [2, 3, 4]
])
y = np.array([10, 20, 30, 12])  # 정답값 (1차원 벡터)

# 🔹 (2) 파라미터 초기화
m, n = X.shape     # m: 샘플 수, n: 특성 수
w = np.zeros(n)    # 가중치 벡터 w (n개)
b = 0.0            # 절편 b

# 🔹 (3) 하이퍼파라미터
lr = 0.01
epochs = 1000

# 🔹 (4) 경사하강법
for epoch in range(epochs):
    # 예측값 계산 (벡터 내적 + 절편)
    y_pred = np.dot(X, w) + b

    # 오차(loss) = 예측 - 실제
    error = y_pred - y

    # 🔸 손실 함수: 평균 제곱 오차(MSE)
    loss = np.mean(error ** 2)

    # 🔸 경사(gradient) 계산 (편미분 결과)
    dw = (2 / m) * np.dot(X.T, error)   # w에 대한 미분
    db = (2 / m) * np.sum(error)        # b에 대한 미분

    # 🔸 파라미터 업데이트 (기울기 반대방향)
    w -= lr * dw
    b -= lr * db

    # 🔸 100회마다 손실 출력
    if epoch % 100 == 0:
        print(f"{epoch}번째 손실: {loss:.4f}")

# 🔹 최종 결과 출력
print("\n최종 가중치 w:", w)
print("최종 절편 b:", b)
