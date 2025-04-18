import numpy as np

np.random.seed(5)

# 이해를 돕기 위해 랜덤 말고 수 지정
x = np.array([[1, 2],[3, 4]])
y = np.array([[2], [3]])

print(f"{x}\n{y}\n{x@y}")
