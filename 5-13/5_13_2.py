import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 및 분할
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# 2. 훈련/테스트 셋 분리 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 3. 특성 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_features = X_train.shape[1] #30

w = np.random.randn(num_features,1)
b = np.random.rand()
learning_rate = 0.01
np.set_printoptions(suppress=True, precision=5)



#1차원을 2차원으로 바꿔줌줌
y_train = y_train.reshape(-1, 1)

#z = wx +B 가설
z = X_train @ w + b


#prediction = 1 / (1 + e^ ( -2 )) 시그모이드 함수 
prediction = 1 / (1 + np.exp(-z))
print(prediction)
print(prediction.shape)
#error = prediction - y_train
error = prediction - y_train

#gradient_w , gradient_b 
gradient_W = X_train @ error / len(X_train)
gradient_b = error.mean()

#updata parameters:w , b
w = w -learning_rate * gradient_W
b = b - learning_rate * gradient_b

#calculate loss
-y_train*np.log(prediction) - (1 - y)*np.log(1 - prediction)