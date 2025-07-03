import numpy as np
from sklearn.linear_model import LinearRegression

# 입력 데이터: [기온, 광고비]
X = np.array([
    [25, 1.0],
    [30, 2.0],
    [35, 3.0],
    [28, 1.5],
    [33, 2.5]
])

# 출력 데이터: 매출
y = np.array([60, 80, 100, 75, 95])

# 모델 생성 & 학습
model = LinearRegression()
model.fit(X, y)

# 결과 출력
print("가중치 (w):", model.coef_)      # [기온 계수, 광고비 계수]
print("절편 (b):", model.intercept_)

# 새로운 데이터 예측 (ex. 기온 31도, 광고비 1.8만원)
new_data = np.array([[31, 1.8]])
pred = model.predict(new_data)
print("예측된 매출:", pred[0])
