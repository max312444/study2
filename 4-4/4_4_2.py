import numpy as np

# bar = np.zeros((2))
# foo = np.zeros((3, 2))
# pos = np.zeros((2, 3, 2))

# print(f"bar.shape: {bar.shape}")
# print(f"foo.shape: {foo.shape}")
# print(f"pos.shape: {pos.shape}")

# print(bar)
# print("---------------------------")
# print(foo)
# print("---------------------------")
# print(pos)

# np.set_printoptions(suppress=True, precision=2)
# bar = np.random.rand(2, 3)

# print(bar)
# print("--" * 10)

# bar = bar * 10
# print(bar)

np.set_printoptions(suppress=True, precision=2)
X = np.random.rand(3, 1) * 10
# H(x) = w * x + b
# y = 2.5 * X + np.random.rand(100, 1)

pos = 2.5 * X
bar = np.random.randn(3, 1) * 2
y = 2.5 * X + bar

print(X)
print("--" * 10)
print(pos)
print("--" * 10)
print(bar)
print("--" * 10)
print(y)
print("--" * 10)
