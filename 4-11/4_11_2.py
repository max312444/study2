
samples = []
y = []
# 기본 값 설정
w = [0.2, 0.3]
b = 0.1
# w 개수 2개 
gradient_w = [0.0, 0.0]
gradient_b = 0.0

# 모든 샘플 순회 : 1 epoch
for dp, y in zip(samples, y):
    #  예측값
    predict_y = w[0] * dp[0] + w[1] * dp[1] + b # w 값을 각각 구해서 다 더함 
    
    # 오차 : 예측값 - 정답
    error = predict_y - y
    
    # 경사값(기울기 값) 누적
    gradient_w[0] += dp[0] * error
    gradient_w[1] += dp[1] * error
    gradient_b += error

# update gradient of each W
w[0] = w[0] - gradient_w[0] / len(samples)
w[1] = w[1] - gradient_w[1] / len(samples)

# update gradient of each b
b = b - gradient_b / len(samples)
