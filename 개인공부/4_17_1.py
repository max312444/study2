import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. 데이터 준비
# 공부시간(hours_study), 수면시간(hours_sleep) → 시험점수(score)
data = {
    'hours_study': [1, 2, 3, 4, 5, 6, 7, 8],
    'hours_sleep': [8, 7, 6, 5, 5, 6, 7, 8],
    'score': [50, 55, 65, 70, 75, 80, 88, 95]
}

df = pd.DataFrame(data)

# 2. 입력(X), 출력(y) 분리
X = df[['hours_study', 'hours_sleep']]
y = df['score']

# 3. 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 예측 및 평가
predictions = model.predict(X_test)
print("예측값:", predictions)
print("실제값:", y_test.values)

# 6. 회귀계수 확인
print("기울기 (weights):", model.coef_)  # 각 피처의 가중치 (w1, w2)
print("절편 (bias):", model.intercept_)  # 절편 b
