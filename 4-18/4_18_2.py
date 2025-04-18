import numpy as np

np.random.seed(5)

x = np.random.randint(1, 4, (2,2))
y = np.random.randint(1, 4, (1,2)) 

# 벡터끼리 연산할 때는 좌항과 우항의 백터의 자릿수(원소의 개수)가 같아야한다.
print(f"{x}\n{y}\n{x+y}") # 벡터끼리의 내적을하면 스칼라가 나온다
# 지금 여기서 브로드캐스팅이 일어나 1이 복제되서 연산이 일어난다.