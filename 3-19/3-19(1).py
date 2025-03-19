import random
import matplotlib.pyplot as plt

# 학습 데이터
x_train = [1, 2, 3]
y_train = [6, 12, 18]

# 학습률 및 반복 횟수 설정
learning_rate = 0.01 # 안정적인 학습을 위해 작은 값 사용
epochs = 100 # 전체 데이터를 100번 학습

# 초기 가중치 설정 (무작위 작은 값)
w  = random.uniform(-1, 1)

# 기록 저장 리스트
weights_history = []
loss_history = []

# Stochastic Gradient Descent (SGD) 구현
for epoch in range(epochs):
    data = list(zip(x_train, y_train))
    random.shuffle(data)  # 데이터 섞기

    for x, y in data:
        # 1. 개별 데이터 샘플에 대한 기울기 계산 (누적 X)
        gradient = x * (w * x - y)

        # 2. 가중치 업데이트
        w -= learning_rate * gradient

        # 3. 새로운 가중치로 예측 후 손실 계산 (MSE)
        loss = (w * x - y) ** 2

        # 4. 값 저장
        weights_history.append(w)
        loss_history.append(loss)

    # 5. Epoch 단위 출력 (마지막 샘플 기준)
    print(f"SGD Epoch {epoch + 1}: W = {w:.5f}, Loss = {loss:.5f}")

print(f"최종 W (SGD): {w:.5f}")

# 그래프 시각화
plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(weights_history, label="SGD W 변화", linestyle='dashed')
plt.xlabel("Iteration")
plt.ylabel("W 값")
plt.title("Gradient Descent 과정에서 W 값 변화")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_history, label="SGD Loss 변화", linestyle='dashed', color='blue')
plt.xlabel("Iteration")
plt.ylabel("Loss 값")
plt.title("Gradient Descent 과정에서 Loss 변화")
plt.legend()

plt.show()
