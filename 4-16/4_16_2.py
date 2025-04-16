import numpy as np

num_features = 3
num_samples = 2

np.random.seed(1)
np.set_printoptions(suppress=True, precision=3)
X = np.random.rand(num_samples, num_features)

print(X)

# h(x) = wx1 + wx2 + wx3 + b
w_true = np.random.randint(1, 10, num_features)
b_true = np.random.randn() * 0.5

y = X[:, 0] * w_true[0] + X[:, 1] * w_true[1] + X[:, 2] * w_true[2] + b_true
print(f"{w_true}, \n{b_true}, \n{y}")