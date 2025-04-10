import numpy as np

num_of_samples = 5
num_of_features = 2

# data set
np.random.seed(1)
X = np.random.rand(num_of_samples, num_of_features) * 10 # 10이 각가의 메트릭스 원소들에 들어가서 곱해진다.
x_true = [5, 3]
b_true = 4

# 출력 값
print(X)
print(X[0, 1])
print(X[1, 0])
print(X[:, 0] * 4)
print(X[:, 0] * 5 + X[:, 1] * 3 + b_true)