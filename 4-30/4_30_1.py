from sklearn.datasets import fetch_california_housing

# 1. 데이터 로드
data = fetch_california_housing()

# 2. 주요 속성 확인
X = data.data
y = data.target
feature_names = data.feature_names

print(type(data), type(X), type(feature_names))
print("입력 X shape:", X.shape)
print("입력 y shape:", y.shape)
print("특성 이름:", feature_names)
print("설명:", data.DESCR[:1000])