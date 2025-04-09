import numpy as np # numpy
import matplotlib.pyplot as plt

# ==== 1. 데이터 생성 ====
num_of_samples = 10
num_of_features = 2

np.random.seed(42) # 재현성을 위한 시드 고정

# 입력 데이터 X: 0~10 사이의 실수값으로 구성된 (10, 2) 배열
X = np.random.rand(num_of_samples, num_of_features) * 10

# 실제 정답을 생성하기 위한 파라미터
true_w = [3, 5]
true_b = 10
noise = np.random.randn(num_of_samples)

# 정답값 y 계산
y = X[:, 0] * true_w[0] + X[:, 1] * true_w[1] + true_b + noise

# ==== 2. 초기 파라미터 설정 ====
w = np.random.rand(num_of_features)
b = np.random.rand()

# ==== 3. 하이퍼파라미터 설정 ====
learning_rate = 0.01
epochs = 500
print_interval = 20