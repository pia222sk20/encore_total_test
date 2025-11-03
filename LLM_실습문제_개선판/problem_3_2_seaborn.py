"""
문제 3.2: Seaborn을 이용한 고급 시각화

요구사항:
1. Seaborn 스타일 설정
2. Heatmap 그리기
3. Pairplot 생성
4. Boxplot으로 분포 비교
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("문제 3.2: Seaborn 고급 시각화")
print("="*60)

# 샘플 데이터 생성
np.random.seed(42)
data = pd.DataFrame({
    '나이': np.random.randint(20, 60, 100),
    '급여': np.random.randint(2000, 6000, 100),
    '근무년수': np.random.randint(0, 30, 100),
    '만족도': np.random.randint(1, 6, 100)
})

# 1. Heatmap
print("\n[1] 상관계수 Heatmap")
print("-" * 60)

fig, ax = plt.subplots(figsize=(8, 6))
correlation = data.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar_kws={'label': '상관계수'})
ax.set_title('변수 간 상관계수', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_seaborn_01_heatmap.png', dpi=100)
print("✓ Heatmap 저장: plot_seaborn_01_heatmap.png")
plt.close()

print("\n상관계수 행렬:")
print(correlation)


# 2. Boxplot
print("\n[2] Boxplot으로 분포 비교")
print("-" * 60)

# 직급별 급여 분포를 보기 위해 카테고리 추가
data['직급'] = pd.cut(data['근무년수'], bins=3, labels=['사원', '대리', '과장'])

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='직급', y='급여', data=data, ax=ax, palette='Set2')
ax.set_xlabel('직급', fontsize=12)
ax.set_ylabel('급여 (만원)', fontsize=12)
ax.set_title('직급별 급여 분포', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_seaborn_02_boxplot.png', dpi=100)
print("✓ Boxplot 저장: plot_seaborn_02_boxplot.png")
plt.close()


# 3. Scatterplot with Hue
print("\n[3] Hue를 이용한 분류 시각화")
print("-" * 60)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='근무년수', y='급여', hue='직급', size='만족도', data=data, ax=ax, s=100, alpha=0.7)
ax.set_xlabel('근무년수', fontsize=12)
ax.set_ylabel('급여 (만원)', fontsize=12)
ax.set_title('근무년수 vs 급여 (직급별 색상, 만족도별 크기)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_seaborn_03_scatterplot.png', dpi=100)
print("✓ Scatterplot 저장: plot_seaborn_03_scatterplot.png")
plt.close()


# 4. 분포 비교
print("\n[4] 분포 비교 (여러 종류)")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram + KDE
ax = axes[0]
sns.histplot(data, x='급여', kde=True, ax=ax, color='skyblue')
ax.set_xlabel('급여 (만원)', fontsize=11)
ax.set_ylabel('빈도', fontsize=11)
ax.set_title('급여 분포 (히스토그램 + KDE)', fontweight='bold')

# Violin plot
ax = axes[1]
data_melted = data[['직급', '급여']].copy()
sns.violinplot(x='직급', y='급여', data=data_melted, ax=ax, palette='Set1')
ax.set_xlabel('직급', fontsize=11)
ax.set_ylabel('급여 (만원)', fontsize=11)
ax.set_title('직급별 급여 바이올린 플롯', fontweight='bold')

plt.tight_layout()
plt.savefig('plot_seaborn_04_distribution.png', dpi=100)
print("✓ 분포 비교 저장: plot_seaborn_04_distribution.png")
plt.close()

print("\n" + "="*60)
print("Seaborn 시각화 완료")
print("="*60)
print("\n생성된 파일:")
print("  - plot_seaborn_01_heatmap.png (상관계수 히트맵)")
print("  - plot_seaborn_02_boxplot.png (박스플롯)")
print("  - plot_seaborn_03_scatterplot.png (산점도)")
print("  - plot_seaborn_04_distribution.png (분포 비교)")
