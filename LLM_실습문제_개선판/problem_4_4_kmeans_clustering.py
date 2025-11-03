"""
문제 4.4: K-Means 클러스터링

요구사항:
1. K-Means 클러스터링 수행
2. 최적의 클러스터 수 결정 (엘보우 방법)
3. 클러스터 시각화
4. 클러스터 내 거리 계산
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("="*60)
print("문제 4.4: K-Means 클러스터링")
print("="*60)

# 1. 데이터 생성
print("\n[1] 클러스터링 데이터 생성")
print("-" * 60)

X, y_true = make_blobs(n_samples=300, n_features=2, centers=3, random_state=42)[:2]

print(f"데이터 형태: {X.shape}")
print(f"클러스터 수: 3개")


# 2. 엘보우 방법으로 최적의 k 찾기
print("\n[2] 최적의 클러스터 수 결정 (엘보우 방법)")
print("-" * 60)

inertias = []
silhouette_scores = []
K_range = range(1, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

print("클러스터 수에 따른 inertia:")
for k, inertia in zip(K_range, inertias):
    print(f"  K={k}: inertia={inertia:.2f}")


# 3. 최적 k로 K-Means 수행
print("\n[3] K=3으로 K-Means 수행")
print("-" * 60)

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

print(f"클러스터 중심:")
for i, center in enumerate(centers):
    print(f"  클러스터 {i}: {center}")

print(f"\n각 클러스터의 샘플 수:")
for i in range(optimal_k):
    print(f"  클러스터 {i}: {(labels == i).sum()}")


# 4. 시각화
print("\n[4] 결과 시각화")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 엘보우 곡선
ax = axes[0, 0]
ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax.axvline(x=3, color='red', linestyle='--', label='최적 K=3')
ax.set_xlabel('클러스터 수 (K)')
ax.set_ylabel('Inertia')
ax.set_title('엘보우 방법')
ax.legend()
ax.grid(True, alpha=0.3)

# 실루엣 점수
ax = axes[0, 1]
ax.plot(range(2, 10), silhouette_scores, 'go-', linewidth=2, markersize=8)
ax.set_xlabel('클러스터 수 (K)')
ax.set_ylabel('실루엣 점수')
ax.set_title('실루엣 점수')
ax.grid(True, alpha=0.3)

# 클러스터 분포
ax = axes[1, 0]
colors = ['red', 'blue', 'green']
for i in range(optimal_k):
    mask = labels == i
    ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=f'클러스터 {i}', s=50, alpha=0.6)
ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='*', s=500, label='중심', edgecolor='yellow', linewidth=2)
ax.set_xlabel('특성 1')
ax.set_ylabel('특성 2')
ax.set_title('K-Means 클러스터링 결과 (K=3)')
ax.legend()
ax.grid(True, alpha=0.3)

# 다양한 K에 대한 비교
ax = axes[1, 1]
test_k_values = [2, 3, 4, 5]
for k in test_k_values:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_test = kmeans_test.fit_predict(X)
    score = silhouette_score(X, labels_test) if k > 1 else 0
    ax.text(0.1, 1.0 - test_k_values.index(k) * 0.2, 
            f'K={k}: silhouette={score:.3f}', fontsize=12, transform=ax.transAxes)
ax.axis('off')
ax.set_title('클러스터 수 비교')

plt.tight_layout()
plt.savefig('problem_4_4_kmeans_clustering.png', dpi=100)
print("✓ 시각화 저장: problem_4_4_kmeans_clustering.png")
plt.close()

print("\n" + "="*60)
print("K-Means 클러스터링 완료")
print("="*60)
