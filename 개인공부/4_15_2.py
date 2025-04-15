import numpy as np

# ----1. 데이터 생성 -----
num_samples = 10
num_features = 2

np.random.seed(42) # 결과 재현성을 위한 시드 고정

# 입력 데이터 (10, 2) 크기의 배열
X = np.random.rand(num_samples, num_features) * 10

# 실제 가중치와 편향
true_weights = [3, 5]
true_bias = 10

# 노이즈 생성
noise = np.random.randn(num_samples) * 0.5

# 정답값 y 계산: y = 3*x1 + 5*x2 + 10 + noise
y = X[:, 0] * true_weights[0] + X[:, 1] * true_weights[1] + true_bias + noise

# ----2. 초기 파라미터 설정 ----
weights = np.random.rand(num_features) # 학습할 가중치
bias = np.random.rand()                # 학습할 편향

# ----3. 하이퍼파라미터 설정 ----
learning_rate = 0.03
epochs = 10000
print_interval = 500 # 주기적으로 손실 출력

# ----4. 학습 시작 ----
for epoch in range(epochs):
    total_loss = 0.0
    total_weight_grad = np.zeros(num_features)
    total_bias_error = 0.0
    
    # 각 데이터 샘플에 대해 예측 및 기울기 계산
    for x_sample, y_true in zip(X, y):
        predction = 0.0
        
        # 예측값 계산: w1*x1 + w2*x2 + ... + b
        for i in range(num_features):
            predction += weights[i] * x_sample[i]
        predction += bias
        
        # 오차 계산
        error = predction - y_true
        
        # 가중치에 대한 gradient 누적
        for i in range(num_features):
            total_weight_grad[i] += x_sample[i] * error
            
        # 편향 오차 누적
        total_bias_error += error
        
        # 손실 누적
        total_loss += error ** 2
        
        # 평균 gradient로 파라미터 업데이트
        for i in range(num_features):
            weights[i] -= learning_rate * (total_weight_grad[i] / num_samples)
        bias -= learning_rate * (total_bias_error / num_samples)
        
        # 일정 주기로 평균 손실 출력
        average_loss = total_loss / num_samples
        if epoch % print_interval == 0:
            print(f"[Epoch {epoch}] Average Loss: {average_loss:.4f}")
            
# ----5. 학습 결과 출력 ----
print("\n Training Complete")
print("Learned Weights:", weights)
print("Learned Bias :", bias)

# ----6. 예측 결과 비교 ----
print("\nPrediction vs Actual (first 5 samples)")
for i in range(5):
    predction = 0.0
    for j in range(num_features):
        predction += weights[j] * X[i][j]
    predction += bias
    print(f"Predicted: {predction:.2f} | Acyual: {y[i]:.2f}")
    