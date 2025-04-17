import numpy as np

num_features = 3
num_samples = 1000

np.random.seed(1)
np.set_printoptions(suppress=True, precision=3)
X = np.random.rand(num_samples, num_features)
# h(x) = wx1 + wx2 + wx3 + b
w_true = np.random.randint(1, 10, num_features)
b_true = np.random.randn() * 0.5

y = X @ w_true + b_true

print(y)

# Learning
w = np.random.rand(num_features)
b = np.random.randn()

print(w, b)

# prediction
prediction = X @ w + b
print(prediction)

# error
error = prediction - y
print(error)