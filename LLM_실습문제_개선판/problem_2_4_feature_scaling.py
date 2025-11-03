"""
문제 2.4: 특성 정규화 및 표준화 (Feature Scaling)

요구사항:
1. Min-Max 정규화 (Normalization)
2. 표준화 (Standardization)
3. 로버스트 스케일링 (이상치 처리)
4. 다양한 방법 비교 및 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer

print("="*60)
print("문제 2.4: 특성 정규화 및 표준화")
print("="*60)

# 1. 데이터 생성 (이상치 포함)
print("\n[1] 데이터 생성")
print("-" * 60)

np.random.seed(42)
# 정상 데이터
normal_data = np.random.normal(loc=100, scale=15, size=95)
# 이상치 추가
outliers = np.array([200, 250, 300, 350, 400])
data = np.concatenate([normal_data, outliers]).reshape(-1, 1)

print(f"데이터 크기: {len(data)}")
print(f"최소값: {data.min():.2f}")
print(f"최대값: {data.max():.2f}")
print(f"평균: {data.mean():.2f}")
print(f"표준편차: {data.std():.2f}")


# 2. Min-Max 정규화
print("\n[2] Min-Max 정규화 (0 ~ 1 범위)")
print("-" * 60)

scaler_minmax = MinMaxScaler(feature_range=(0, 1))
data_minmax = scaler_minmax.fit_transform(data)

print("정규화 후:")
print(f"  최소값: {data_minmax.min():.4f}")
print(f"  최대값: {data_minmax.max():.4f}")
print(f"  평균: {data_minmax.mean():.4f}")
print(f"  표준편차: {data_minmax.std():.4f}")
print(f"  처음 5개: {data_minmax[:5].ravel()}")


# 3. 표준화 (Z-score)
print("\n[3] 표준화 (Z-score 정규화)")
print("-" * 60)

scaler_standard = StandardScaler()
data_standard = scaler_standard.fit_transform(data)

print("표준화 후:")
print(f"  최소값: {data_standard.min():.4f}")
print(f"  최대값: {data_standard.max():.4f}")
print(f"  평균: {data_standard.mean():.4f}")
print(f"  표준편차: {data_standard.std():.4f}")
print(f"  처음 5개: {data_standard[:5].ravel()}")


# 4. 로버스트 스케일링
print("\n[4] 로버스트 스케일링 (이상치 영향 최소화)")
print("-" * 60)

scaler_robust = RobustScaler()
data_robust = scaler_robust.fit_transform(data)

print("로버스트 스케일링 후:")
print(f"  최소값: {data_robust.min():.4f}")
print(f"  최대값: {data_robust.max():.4f}")
print(f"  평균: {data_robust.mean():.4f}")
print(f"  표준편차: {data_robust.std():.4f}")


# 5. 비교 테이블
print("\n[5] 스케일링 방법 비교")
print("-" * 60)

comparison = pd.DataFrame({
    '원본': [data.min(), data.max(), data.mean(), data.std()],
    'Min-Max': [data_minmax.min(), data_minmax.max(), data_minmax.mean(), data_minmax.std()],
    '표준화': [data_standard.min(), data_standard.max(), data_standard.mean(), data_standard.std()],
    '로버스트': [data_robust.min(), data_robust.max(), data_robust.mean(), data_robust.std()]
}, index=['최소값', '최대값', '평균', '표준편차'])

print(comparison.to_string())


# 6. 시각화
print("\n[6] 스케일링 방법 시각화")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 원본
ax = axes[0, 0]
ax.hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('값', fontsize=11)
ax.set_ylabel('빈도', fontsize=11)
ax.set_title('원본 데이터', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Min-Max
ax = axes[0, 1]
ax.hist(data_minmax, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
ax.set_xlabel('값', fontsize=11)
ax.set_ylabel('빈도', fontsize=11)
ax.set_title('Min-Max 정규화 (0 ~ 1)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 표준화
ax = axes[1, 0]
ax.hist(data_standard, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
ax.set_xlabel('값', fontsize=11)
ax.set_ylabel('빈도', fontsize=11)
ax.set_title('표준화 (Z-score)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 로버스트
ax = axes[1, 1]
ax.hist(data_robust, bins=20, color='lightyellow', edgecolor='black', alpha=0.7)
ax.set_xlabel('값', fontsize=11)
ax.set_ylabel('빈도', fontsize=11)
ax.set_title('로버스트 스케일링', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('다양한 스케일링 방법 비교', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('problem_2_4_feature_scaling.png', dpi=100)
print("✓ 시각화 저장: problem_2_4_feature_scaling.png")
plt.close()


# 7. 역변환
print("\n[7] 역변환 (원본으로 복원)")
print("-" * 60)

data_minmax_inverse = scaler_minmax.inverse_transform(data_minmax)
data_standard_inverse = scaler_standard.inverse_transform(data_standard)

print(f"Min-Max 역변환 에러: {np.mean(np.abs(data - data_minmax_inverse)):.6f}")
print(f"표준화 역변환 에러: {np.mean(np.abs(data - data_standard_inverse)):.6f}")

print("\n" + "="*60)
print("특성 정규화 및 표준화 완료")
print("="*60)
