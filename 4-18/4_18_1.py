import numpy as np

np.random.seed(3)

x = np.random.randint(1, 4, (2,2)) # 1번쨰 2번쨰는 범위 1에서4미만
y = np.random.randint(1, 4, (2,2)) # 3번째는 원소수 및 차원

# 벡터끼리 연산할 때는 좌항과 우항의 백터의 자릿수(원소의 개수)가 같아야한다.
print(f"{x}\n{y}") # 벡터끼리의 내적을하면 스칼라가 나온다