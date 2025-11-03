"""
문제 1.1: NumPy 벡터 및 행렬 연산

요구사항:
1. 두 벡터의 내적 계산
2. 행렬 곱셈 수행
3. 벡터 노름 계산 (L1, L2, L∞)
4. 조건부 인덱싱 적용
"""

import numpy as np

print("="*60)
print("문제 1.1: NumPy 벡터 및 행렬 연산")
print("="*60)

# 1. 벡터 내적
print("\n[1] 벡터 내적 계산")
print("-" * 60)
v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 8])

dot_product = np.dot(v1, v2)
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"내적 (np.dot) = {dot_product}")
print(f"내적 (@ 연산자) = {v1 @ v2}")
print(f"수동 계산: 1×5 + 2×6 + 3×7 + 4×8 = {1*5 + 2*6 + 3*7 + 4*8}")


# 2. 행렬 곱셈
print("\n[2] 행렬 곱셈")
print("-" * 60)
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2×3
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])   # 3×2

C = A @ B  # 또는 np.matmul(A, B)
print(f"행렬 A (2×3):\n{A}")
print(f"\n행렬 B (3×2):\n{B}")
print(f"\nA @ B = C (2×2):\n{C}")
print(f"\n첫 번째 행 계산 검증:")
print(f"C[0,0] = 1×7 + 2×9 + 3×11 = {1*7 + 2*9 + 3*11} (실제: {C[0,0]})")
print(f"C[0,1] = 1×8 + 2×10 + 3×12 = {1*8 + 2*10 + 3*12} (실제: {C[0,1]})")


# 3. 벡터 노름
print("\n[3] 벡터 노름 계산")
print("-" * 60)
c = np.array([3, 4, 5])
print(f"벡터 c = {c}")

# L1 노름 (맨하탄 거리)
l1_norm = np.linalg.norm(c, ord=1)
print(f"L1 노름 = |3| + |4| + |5| = {l1_norm}")

# L2 노름 (유클리드 거리)
l2_norm = np.linalg.norm(c, ord=2)
l2_manual = np.sqrt(3**2 + 4**2 + 5**2)
print(f"L2 노름 = √(3² + 4² + 5²) = {l2_norm:.4f} (수동: {l2_manual:.4f})")

# L∞ 노름 (최댓값)
linf_norm = np.linalg.norm(c, ord=np.inf)
print(f"L∞ 노름 = max(|3|, |4|, |5|) = {linf_norm}")


# 4. 조건부 인덱싱
print("\n[4] 조건부 인덱싱")
print("-" * 60)
data = np.array([1, 5, 3, 8, 2, 9, 4, 7, 6])
print(f"원본 배열: {data}")

# 5보다 큰 값
greater_than_5 = data[data > 5]
print(f"\n5보다 큰 값: {greater_than_5}")

# 3 이상 7 이하 값
between_3_7 = data[(data >= 3) & (data <= 7)]
print(f"3 이상 7 이하 값: {between_3_7}")

# 짝수
even = data[data % 2 == 0]
print(f"짝수: {even}")

print("\n" + "="*60)
print("모든 계산 완료")
print("="*60)
