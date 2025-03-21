import numpy as np
import random
import matplotlib.pyplot as plt

# Data set
# input
# input data, features
# H(x) -> input data : x1 -> xn
x_train = [ np.random.rand() * 10 for _ in range(50)]
y_train = [val + np.random.rand() * 5 for val in x_train]


# BGC (Batch Gradient Descent) 배치경사하강법을 
# 이용하여 Linear Rergeresion 적용
# loss 값 구할 필요는 없음 안구해도됨 -> 값이 어떻게 변하는지 확인할 때만 추가
w = 0.0 # 초기 가중치
learning_rate = 0.008 # 학습률 -> 이건 조금씩 올리면 됨 한번에 크게 올리면 확 튀어 올라서 학습이 안될 수 있음
epoch = 100 # 학습 반복 횟수 -> 데이터 수가 적을 때, 정답에 잘못 찾아갈때 올리면 됨
loss_history = [] # 손실 함수 값 저장 리스트

# H(w) -> w * x + b
for num_of_epoch in range(epoch):
    data = list(zip(x_train, y_train))
    random.shuffle(data)

    # GD 수행 후 최적의 w 값 도출
    for x, y in data: # 데이터의 샘플 갯수 만큼 반복
    # w 값 업데이트
        w = w - learning_rate * (x * (x * w - y))
        pass
    
x_test = [val for val in range(10)]
y_test = [w * val for val in x_test]

plt.scatter(x_train,y_train, color ="blue")
plt.plot(x_test, y_test)
plt.show()

# output
# label
# f(x1) -> f(x2)