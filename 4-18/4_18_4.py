import numpy as np

num_features = 4
num_samples = 1000

np.random.seed(5)

X = np.random.rand(num_samples, num_features) * 2
w_true = np.random.randint(1, 11, (num_features, 1))
b_true = np.random.rand() * 0.5 # -1.5 ~ 1.5

y = X @ w_true + b_true # 매트릭스 * 매트릭스 + 스칼라

########################################################

w = np.random.rand(num_features, 1) # 4행 1열
b = np.random.rand()
learn_rate = 0.01

gradient = np.zeros(num_features)

for _ in range(10000):
    # 예측 값
    predict_y = X @ w + b # 내적을 사용해서 y값을 구함함

    # 오차
    error = predict_y - y

    # 기울기
    gradient_w = X.T @ error / num_samples # 기울기 구하는 것을 내적으로 구함함
    gradient_b = error.mean()

    # w, b 값 업데이트
    w = w - learn_rate * gradient_w
    b = b - learn_rate * gradient_b

print(f"W_true: {w_true}, \nb_true: {b_true}")    
print(f"W: {w}, \nb: {b}")