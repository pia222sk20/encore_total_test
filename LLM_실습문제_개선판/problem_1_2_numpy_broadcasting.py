"""
문제 1.2: NumPy 브로드캐스팅 및 벡터화

요구사항:
1. 1D와 2D 배열 간 브로드캐스팅
2. 행렬의 각 행/열에 벡터 연산
3. 벡터화를 이용한 반복문 없애기
4. 성능 비교
"""

import numpy as np
import time

print("="*60)
print("문제 1.2: NumPy 브로드캐스팅 및 벡터화")
print("="*60)

# 1. 기본 브로드캐스팅
print("\n[1] 기본 브로드캐스팅")
print("-" * 60)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
vector = np.array([10, 20, 30])

print(f"행렬 (3×3):\n{matrix}")
print(f"\n벡터 (3,): {vector}")

# 브로드캐스팅: 벡터가 각 행에 자동으로 확장
result = matrix + vector
print(f"\n행렬 + 벡터 (브로드캐스팅):\n{result}")
print(f"설명: 벡터가 자동으로 행마다 더해짐")


# 2. 행/열 별 연산
print("\n[2] 행/열 별 연산")
print("-" * 60)
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

print(f"원본 데이터:\n{data}")

# 각 행의 합
row_sums = data.sum(axis=1, keepdims=True)
print(f"\n각 행의 합 (shape {row_sums.shape}):\n{row_sums}")

# 각 행으로 나누기 (정규화)
normalized = data / row_sums
print(f"\n각 행으로 정규화:\n{normalized}")
print(f"검증: 첫 번째 행의 합 = {normalized[0].sum():.6f} (1에 가까워야 함)")

# 각 열의 평균
col_means = data.mean(axis=0, keepdims=True)
print(f"\n각 열의 평균 (shape {col_means.shape}):\n{col_means}")

# 각 열에서 평균 뺄셈 (중앙화)
centered = data - col_means
print(f"\n각 열의 평균을 뺀 후:\n{centered}")


# 3. 벡터화를 이용한 반복문 제거
print("\n[3] 반복문 제거 (벡터화)")
print("-" * 60)

# 작은 예시로 동일한 결과 확인
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# 방법 1: 반복문 (느림)
def method_loop(x, y):
    result = []
    for i in range(len(x)):
        result.append(x[i] * y[i] + x[i]**2)
    return result

# 방법 2: 벡터화 (빠름)
def method_vectorized(x, y):
    return x * y + x**2

result_loop = method_loop(x, y)
result_vec = method_vectorized(x, y)

print(f"x = {x}")
print(f"y = {y}")
print(f"x*y + x² 계산:")
print(f"  반복문 결과: {result_loop}")
print(f"  벡터화 결과: {result_vec}")
print(f"  동일한가? {np.allclose(result_loop, result_vec)}")


# 4. 성능 비교
print("\n[4] 성능 비교 (큰 배열)")
print("-" * 60)
large_x = np.arange(1000000)
large_y = np.arange(1000000, 2000000)

# 반복문 버전
start = time.time()
for _ in range(10):
    result_loop = []
    for i in range(len(large_x)):
        result_loop.append(large_x[i] + large_y[i])
time_loop = time.time() - start

# 벡터화 버전
start = time.time()
for _ in range(10):
    result_vec = large_x + large_y
time_vec = time.time() - start

print(f"배열 크기: {len(large_x):,}")
print(f"반복 횟수: 10회")
print(f"반복문 소요 시간: {time_loop:.6f}초")
print(f"벡터화 소요 시간: {time_vec:.6f}초")
print(f"속도 향상: {time_loop/time_vec:.1f}배")

print("\n" + "="*60)
print("모든 계산 완료")
print("="*60)
