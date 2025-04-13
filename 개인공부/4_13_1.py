# 입력 데이터 (예시)
samples = [
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 0.0],
    [3.0, 2.0, 1.0]
]
y = [1.2, 0.5, 2.3]

# 초기 가중치와 편향
w = [0.1, 0.2, 0.3]  # feature 3개니까 w도 3개
b = 0.05

# 경사값 초기화
gradient_w = [0.0, 0.0, 0.0]
gradient_b = 0.0

# 1 epoch 수행
for dp, y_true in zip(samples, y):
    # 예측값 계산
    predict_y = sum([w[i] * dp[i] for i in range(3)]) + b
    
    # 오차 계산
    error = predict_y - y_true

    # 경사 누적
    for i in range(3):
        gradient_w[i] += dp[i] * error
    gradient_b += error

# 경사하강법으로 w, b 업데이트
for i in range(3):
    w[i] = w[i] - gradient_w[i] / len(samples)
b = b - gradient_b / len(samples)

# 출력 확인
print("업데이트된 w:", w)
print("업데이트된 b:", b)
