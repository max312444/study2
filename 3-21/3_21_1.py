import numpy as np
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
b = 0.0 # 초기 바이어스 값
learning_rate = 0.001 # 학습률 -> 이건 조금씩 올리면 됨 한번에 크게 올리면 확 튀어 올라서 학습이 안될 수 있음
epoch = 100 # 학습 반복 횟수 -> 데이터 수가 적을 때, 정답에 잘못 찾아갈때 올리면 됨
loss_history = [] # 손실 함수 값 저장 리스트

# H(w) -> w * x + b
for num_of_epoch in range(epoch):
    gradient_w_sum = 0.0 # epoch 를 돌 때 마다 초기화
    gradient_b_sum = 0.0
    loss = 0.0
    # zip은 나온 원소들을 튜플로 묶어버림
    # GD 수행 후 최적의 w 값 도출
    for x, y in zip(x_train, y_train): # 데이터의 샘플 갯수 만큼 반복
        gradient_w_sum += x * (w * x + b - y)
        gradient_b_sum += (w * x + b - y)
        loss += (w * x + b - y) ** 2
        
    # w, b 값 업데이트
    w = w - learning_rate * (gradient_w_sum / len(x_train))
    b = b - learning_rate * (gradient_b_sum / len(x_train))
    # w = w - Learning rate * gredient_sum
    
    loss_history.append(loss / len(x_train))
    print(f"loss : {loss / len(x_train)}")

x_test = [val for val in range(10)]
y_test = [w * val + b for val in x_test]

plt.scatter(x_train,y_train, color ="blue")
plt.plot(x_test, y_test)
plt.show()

# output
# label
# f(x1) -> f(x2)