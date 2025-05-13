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

# 4. 초기 설정
num_features = X_train.shape[1]
epochs = 100000
learning_rate = 0.01

# 가중치, 편향 초기화 (정규분포)
weights = np.random.randn(num_features, 1)
bias = np.random.randn()

# 정답 레이블 reshape : (n_samples, 1)로 변환
y_train = y_train.reshape(-1, 1)

# 5. 경사하강법 학습 루프
for epoch in range(epochs):
    # 예측값 계산: z = Xw + b
    z = X_train @ weights + bias
    predictions = 1 / (1 + np.exp(-z))
    
    # 오차 계산
    errors = predictions - y_train
    
    # 그래디언트 계산
    grad_weights = X_train.T @ errors / len(X_train)
    grad_bias = np.mean(errors)
    
    # 파라미터 업데이트
    weights -= learning_rate * grad_weights
    bias -= learning_rate * grad_bias
    
    # 손실 함수 계산 (로그 손실)
    loss = -np.mean(
        y_train * np.log(predictions + 1e-15) + 
        (1 - y_train) * np.log(1 - predictions + 1e-15)
    )
    
    # 학습 상황 출력 (옵션: 조절 가능)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
# 6. 테스트 세트에 대한 예측 및 정확도 평가ㅣ
z_test = X_test @ weights + bias
y_prob_test = 1 / (1 + np.exp(-z_test))
y_preb_test = (y_prob_test >= 0.5).astype(int)

# 정확도 평가
test_accuracy = np.mean(y_preb_test.reshape(-1) == y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
