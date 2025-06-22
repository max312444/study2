from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터셋 로딩 및 분할
digits = load_digits() # 사이킷런에 저장된 data set
features = digits.data                    # (1797, 64): 8x8 이미지 벡터
labels = digits.target                    # (1797,): 0~9 클래스 정수

# print(features.shape)
# print(labels.shape)
# print(labels[:100])

# 2. 학습/테스트 셋 분할
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels # 비율를 0~9 동일한 비율로 유지
)

# 3. 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

np.set_printoptions(suppress=True)

# 4. 기본 설정
num_featuers = X_train_std.shape[1] # 특성의 개수 64개
num_samples = X_train_std.shape[0]  # 샘플의 개수 1437개
num_class = 10
learging_rate = 0.01
epochs = 50000


W = np.random.randn(num_featuers, num_class) # col 64, row 10
b = np.zeros(num_class) #10, -> 행 벡터

for i in range(epochs):
    # X (1437, 64) @ w(64, 10) + b(10,)
    logit = X_train_std @ W + b # 1437, 10 (1437, (w*x)*64) -> 브로드캐스팅이 적용됨 (자동으로 크기 적용)
    logit_max = np.max(logit, axis=1, keepdims=True) # 기존 값 유지 keepdims
    logit -= logit_max # Softmax의 수치 안정화, 너무 클 경우 최대값을 기준으로 빼준다. (음수로 되면 0에 가까워 짐)
    exp_logit = np.exp(logit) # 각 클래스 값을 지수로 집어 넣어 0 이상의 값 적용 (1437, 10)
    exp_logit_sum = np.sum(exp_logit, axis=1, keepdims=True) # 각 클래스 별 합 (1437, 1)
    softmax = exp_logit / exp_logit_sum

    i_matrix = np.eye(num_class) # 해당 수에 맞는 단위행렬 생성
    one_hot = i_matrix[y_train] # 각 레이블에 맞는 정답 지정

    #error = softmax(1437, 10) - one_hot (1437, 10)

    error = softmax - one_hot # 예측(확률 그 자체) - 정답
    gradient_w = X_train_std.T @ error / num_samples
    gradient_b = np.sum(error, axis=0) / num_samples # 열 별로 계산

    W -= learging_rate * gradient_w
    b -= learging_rate * gradient_b

    # loss
    loss = -np.sum(np.log(softmax + 1e-15) * one_hot) / num_samples

    if i % 1000 == 0:
        print(f"Train Loss f{loss:.4f}")

# print(logit.shape)
# print(logit[0])
# print(logit_max[0])
# print(logit_max.shape)

# for idx in range(5): # 음수값이 존재하지 않음
#     print(exp_logit[idx])
#     print(np.sum(exp_logit[idx]))

# print(np.sum(exp_logit_sum[0]))
# print(np.sum(exp_logit_sum[10]))

# print(softmax[0])
# print(np.sum(softmax[0]))
# print(np.sum(softmax[10]))

# print(y_train[0])
# print(one_hot[0])

# print(gradient_w.shape)
# print(gradient_b.shape)
# print(loss.shape)


def predict(arg_X, arg_label):
    # X (1437, 64) @ w(64, 10) + b(10,)
    logit = arg_X @ W + b # 1437, 10 (1437, (w*x)*64) -> 브로드캐스팅이 적용됨 (자동으로 크기 적용)
    logit_max = np.max(logit, keepdims=True) # 기존 값 유지 keepdims
    logit -= logit_max # Softmax의 수치 안정화, 너무 클 경우 최대값을 기준으로 빼준다. (음수로 되면 0에 가까워 짐)
    exp_logit = np.exp(logit) # 각 클래스 값을 지수로 집어 넣어 0 이상의 값 적용 (1437, 10)
    exp_logit_sum = np.sum(exp_logit, keepdims=True) # 각 클래스 별 합 (1437, 1)
    softmax = exp_logit / exp_logit_sum

    predict = np.argmax(softmax)

    print(f"lable: {arg_label}, predict: {predict}")

for idx in range(0, 10):
    predict(X_test_std[idx], y_test[idx])