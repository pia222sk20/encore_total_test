"""
문제 3.1: Matplotlib을 이용한 기본 차트 그리기

요구사항:
1. 선 그래프, 산점도, 막대 그래프, 히스토그램
2. 서브플롯 사용
3. 축 레이블, 제목, 범례 추가
4. 여러 라인/그룹 표현
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("="*60)
print("문제 3.1: Matplotlib 기본 차트 그리기")
print("="*60)

# 1. 선 그래프
print("\n[1] 선 그래프")
print("-" * 60)

# 데이터 준비
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y1, label='sin(x)', marker='o', linestyle='-', linewidth=2)
ax.plot(x, y2, label='cos(x)', marker='s', linestyle='--', linewidth=2)
ax.set_xlabel('x 값', fontsize=12)
ax.set_ylabel('y 값', fontsize=12)
ax.set_title('삼각함수 그래프', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_01_line.png', dpi=100)
print("✓ 선 그래프 저장: plot_01_line.png")
plt.close()


# 2. 산점도
print("\n[2] 산점도")
print("-" * 60)

np.random.seed(42)
n_points = 100
x_scatter = np.random.randn(n_points)
y_scatter = 2 * x_scatter + np.random.randn(n_points) * 0.5

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x_scatter, y_scatter, s=50, alpha=0.6, c='blue')
ax.plot([-3, 3], [-6, 6], 'r--', label='y=2x', linewidth=2)
ax.set_xlabel('X 변수', fontsize=12)
ax.set_ylabel('Y 변수', fontsize=12)
ax.set_title('산점도: X와 Y의 관계', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_02_scatter.png', dpi=100)
print("✓ 산점도 저장: plot_02_scatter.png")
plt.close()


# 3. 막대 그래프
print("\n[3] 막대 그래프")
print("-" * 60)

categories = ['2020년', '2021년', '2022년', '2023년', '2024년']
sales = [1200, 1450, 1800, 2100, 2500]
profit = [300, 400, 550, 700, 900]

x_pos = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x_pos - width/2, sales, width, label='매출액', color='skyblue')
bars2 = ax.bar(x_pos + width/2, profit, width, label='순이익', color='orange')

ax.set_xlabel('연도', fontsize=12)
ax.set_ylabel('금액 (백만원)', fontsize=12)
ax.set_title('연도별 매출액 및 순이익', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories)
ax.legend()

# 값 표시
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('plot_03_bar.png', dpi=100)
print("✓ 막대 그래프 저장: plot_03_bar.png")
plt.close()


# 4. 히스토그램
print("\n[4] 히스토그램")
print("-" * 60)

# 정규분포 데이터
data = np.random.normal(loc=100, scale=15, size=1000)

fig, ax = plt.subplots(figsize=(10, 6))
counts, bins, patches = ax.hist(data, bins=30, color='green', alpha=0.7, edgecolor='black')
ax.axvline(float(data.mean()), color='red', linestyle='--', linewidth=2, label=f'평균: {data.mean():.1f}')
ax.axvline(float(np.median(data)), color='orange', linestyle='--', linewidth=2, label=f'중간값: {np.median(data):.1f}')
ax.set_xlabel('값', fontsize=12)
ax.set_ylabel('빈도', fontsize=12)
ax.set_title('정규분포 히스토그램', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('plot_04_hist.png', dpi=100)
print("✓ 히스토그램 저장: plot_04_hist.png")
plt.close()


# 5. 서브플롯
print("\n[5] 서브플롯 (2×2 그리드)")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 서브플롯 1: 선 그래프
x = np.linspace(0, 2*np.pi, 100)
axes[0, 0].plot(x, np.sin(x), 'b-', linewidth=2)
axes[0, 0].set_title('sin(x)', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 서브플롯 2: 산점도
axes[0, 1].scatter(np.random.randn(50), np.random.randn(50), s=50, alpha=0.6)
axes[0, 1].set_title('산점도', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 서브플롯 3: 막대
axes[1, 0].bar(['A', 'B', 'C', 'D'], [10, 15, 7, 12], color=['red', 'blue', 'green', 'orange'])
axes[1, 0].set_title('막대 그래프', fontweight='bold')

# 서브플롯 4: 히스토그램
axes[1, 1].hist(np.random.randn(500), bins=20, color='purple', alpha=0.7)
axes[1, 1].set_title('히스토그램', fontweight='bold')

plt.suptitle('Matplotlib 서브플롯 예제', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('plot_05_subplots.png', dpi=100)
print("✓ 서브플롯 저장: plot_05_subplots.png")
plt.close()

print("\n" + "="*60)
print("모든 차트 그리기 완료")
print("="*60)
print("\n생성된 파일:")
print("  - plot_01_line.png (선 그래프)")
print("  - plot_02_scatter.png (산점도)")
print("  - plot_03_bar.png (막대 그래프)")
print("  - plot_04_hist.png (히스토그램)")
print("  - plot_05_subplots.png (서브플롯)")
