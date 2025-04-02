from sklearn.model_selection import train_test_split
import numpy as np

# X : 입력값
X = np.random.rand(10, 2) * 5

# y : 출력값 (정답)
y = np.random.randint(0, 2, size=10)


# Sample : 100
# train : 80
# test : 20 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(f"X_train.shape: {X_train.shape}")
print(X_train)