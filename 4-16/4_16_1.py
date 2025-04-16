import numpy as np

# h(x) = wx1 + wx2 + wx3 + b
bar = np.ones((5, 2))
kin = np.zeros((2, 3, 4))

print(f"{bar.shape}, {kin.shape}")
print(f"(bar), (kin)")