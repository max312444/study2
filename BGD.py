import matplotlib.pyplot as plt
import numpy as np

# ✅ 1. 데이터 생성 (리스트 활용, numpy 함수 최소화)
x_data = [np.random.rand() * 10 for _ in range(50)]  # 입력값 X (0~10 범위의 난수 50개)
y_data = [2 * x + np.random.rand() * 4 for x in x_data]  # 실제값 Y (y = 2x + 노이즈)

# ✅ 2. 학습 파라미터 초기화
weight = 4  # 초기 가중치 (W)
learning_rate = 0.01  # 학습률
epochs = 100  # 학습 반복 횟수
loss_history = []  # 손실 함수 값 저장 리스트

# ✅ 3. 배치 경사 하강법 (Batch Gradient Descent, BGD)
for _ in range(epochs):
    total_loss = 0  # 손실 값 초기화
    gradient_sum = 0  # 기울기(Gradient) 합 초기화

    for i in range(len(x_data)):  # 개별 데이터 처리
        x_train = x_data[i]
        y_train = y_data[i]

        # (1) 예측값 계산
        predicted_y = weight * x_train

        # (2) 오차(error) 계산
        error = predicted_y - y_train

        # (3) 손실 값 계산 (MSE를 위한 합산)
        total_loss += error ** 2

        # (4) 평균 기울기(Gradient) 계산을 위한 합산
        gradient_sum += x_train * error

    # (5) 평균 손실 계산 (MSE)
    total_loss /= len(x_data)  # 전체 개수로 나눠 평균 손실 계산

    # (6) 평균 기울기 계산
    gradient = gradient_sum / len(x_data)

    # (7) 가중치(weight) 업데이트 (경사 하강법 적용)
    weight = weight - learning_rate * gradient

    # (8) 손실 값 저장 (시각화를 위해)
    loss_history.append(total_loss)
