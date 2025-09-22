"""
문제 4.5: 클러스터링 (K-means, 계층적 클러스터링)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

def problem_4_5():
    print("=== 문제 4.5: 클러스터링 (K-means, 계층적 클러스터링) ===")
    
    # 1. 합성 데이터 생성
    print("=== 1. 합성 데이터로 클러스터링 ===")
    X_synthetic, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, 
                                   random_state=0, n_features=2)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X_synthetic[:, 0], X_synthetic[:, 1], c=y_true, cmap='viridis')
    plt.title('True Clusters (Synthetic Data)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # 2. K-means 클러스터링
    kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
    y_kmeans = kmeans.fit_predict(X_synthetic)
    
    plt.subplot(1, 3, 2)
    plt.scatter(X_synthetic[:, 0], X_synthetic[:, 1], c=y_kmeans, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
               s=300, c='red', marker='x', linewidths=3, label='Centroids')
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # 3. 계층적 클러스터링
    hierarchical = AgglomerativeClustering(n_clusters=4)
    y_hierarchical = hierarchical.fit_predict(X_synthetic)
    
    plt.subplot(1, 3, 3)
    plt.scatter(X_synthetic[:, 0], X_synthetic[:, 1], c=y_hierarchical, cmap='viridis')
    plt.title('Hierarchical Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    # 성능 평가
    kmeans_ari = adjusted_rand_score(y_true, y_kmeans)
    hierarchical_ari = adjusted_rand_score(y_true, y_hierarchical)
    kmeans_silhouette = silhouette_score(X_synthetic, y_kmeans)
    hierarchical_silhouette = silhouette_score(X_synthetic, y_hierarchical)
    
    print(f"K-means ARI: {kmeans_ari:.4f}")
    print(f"Hierarchical ARI: {hierarchical_ari:.4f}")
    print(f"K-means Silhouette: {kmeans_silhouette:.4f}")
    print(f"Hierarchical Silhouette: {hierarchical_silhouette:.4f}")
    
    # 4. 실제 데이터셋 (Iris) 클러스터링
    print(f"\n=== 2. Iris 데이터셋 클러스터링 ===")
    iris = load_iris()
    X_iris = iris.data
    y_iris_true = iris.target
    
    # 데이터 정규화
    scaler = StandardScaler()
    X_iris_scaled = scaler.fit_transform(X_iris)
    
    print(f"Iris 데이터 형태: {X_iris.shape}")
    print(f"클래스: {iris.target_names}")
    
    # 최적 클러스터 수 찾기 (Elbow Method)
    print(f"\n=== Elbow Method for Optimal K ===")
    K_range = range(1, 11)
    inertias = []
    silhouette_scores = []
    
    for k in K_range:
        if k == 1:
            inertias.append(0)
            silhouette_scores.append(0)
        else:
            kmeans_temp = KMeans(n_clusters=k, random_state=0, n_init=10)
            kmeans_temp.fit(X_iris_scaled)
            inertias.append(kmeans_temp.inertia_)
            silhouette_scores.append(silhouette_score(X_iris_scaled, kmeans_temp.labels_))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(K_range[1:], silhouette_scores[1:], 'ro-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    optimal_k = K_range[1:][np.argmax(silhouette_scores[1:])]
    print(f"최적 클러스터 수 (Silhouette 기준): {optimal_k}")
    
    # 5. Iris 데이터 클러스터링 결과
    kmeans_iris = KMeans(n_clusters=3, random_state=0, n_init=10)
    y_kmeans_iris = kmeans_iris.fit_predict(X_iris_scaled)
    
    hierarchical_iris = AgglomerativeClustering(n_clusters=3)
    y_hierarchical_iris = hierarchical_iris.fit_predict(X_iris_scaled)
    
    # 결과 시각화 (첫 2개 특성만 사용)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris_true, cmap='viridis')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('True Classes')
    
    plt.subplot(1, 3, 2)
    plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_kmeans_iris, cmap='viridis')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('K-means Clustering')
    
    plt.subplot(1, 3, 3)
    plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_hierarchical_iris, cmap='viridis')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Hierarchical Clustering')
    
    plt.tight_layout()
    plt.show()
    
    # 6. 계층적 클러스터링 덴드로그램
    plt.figure(figsize=(12, 6))
    
    # 샘플 일부만 사용 (가독성을 위해)
    sample_indices = np.random.choice(len(X_iris_scaled), 30, replace=False)
    X_sample = X_iris_scaled[sample_indices]
    
    linkage_matrix = linkage(X_sample, method='ward')
    dendrogram(linkage_matrix, labels=[f'Sample {i}' for i in sample_indices])
    plt.title('Hierarchical Clustering Dendrogram (Sample)')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 7. 성능 평가
    iris_kmeans_ari = adjusted_rand_score(y_iris_true, y_kmeans_iris)
    iris_hierarchical_ari = adjusted_rand_score(y_iris_true, y_hierarchical_iris)
    iris_kmeans_silhouette = silhouette_score(X_iris_scaled, y_kmeans_iris)
    iris_hierarchical_silhouette = silhouette_score(X_iris_scaled, y_hierarchical_iris)
    
    print(f"\n=== Iris 클러스터링 성능 ===")
    print(f"K-means ARI: {iris_kmeans_ari:.4f}")
    print(f"Hierarchical ARI: {iris_hierarchical_ari:.4f}")
    print(f"K-means Silhouette: {iris_kmeans_silhouette:.4f}")
    print(f"Hierarchical Silhouette: {iris_hierarchical_silhouette:.4f}")
    
    # 8. 클러스터별 특성 분석
    iris_df = pd.DataFrame(X_iris, columns=iris.feature_names)
    iris_df['true_class'] = y_iris_true
    iris_df['kmeans_cluster'] = y_kmeans_iris
    iris_df['hierarchical_cluster'] = y_hierarchical_iris
    
    print(f"\n=== 클러스터별 평균 특성 (K-means) ===")
    cluster_means = iris_df.groupby('kmeans_cluster')[iris.feature_names].mean()
    print(cluster_means)
    
    # 9. 혼동 행렬 스타일 비교
    from sklearn.metrics import confusion_matrix
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # K-means vs True
    cm_kmeans = confusion_matrix(y_iris_true, y_kmeans_iris)
    sns.heatmap(cm_kmeans, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_xlabel('K-means Cluster')
    axes[0].set_ylabel('True Class')
    axes[0].set_title('K-means vs True Classes')
    
    # Hierarchical vs True
    cm_hierarchical = confusion_matrix(y_iris_true, y_hierarchical_iris)
    sns.heatmap(cm_hierarchical, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_xlabel('Hierarchical Cluster')
    axes[1].set_ylabel('True Class')
    axes[1].set_title('Hierarchical vs True Classes')
    
    plt.tight_layout()
    plt.show()
    
    # 10. 다른 연결 방법 비교
    print(f"\n=== 계층적 클러스터링 연결 방법 비교 ===")
    linkage_methods = ['ward', 'complete', 'average', 'single']
    linkage_scores = {}
    
    for method in linkage_methods:
        if method == 'ward':
            hc = AgglomerativeClustering(n_clusters=3, linkage=method)
        else:
            hc = AgglomerativeClustering(n_clusters=3, linkage=method)
        
        labels = hc.fit_predict(X_iris_scaled)
        ari = adjusted_rand_score(y_iris_true, labels)
        silhouette = silhouette_score(X_iris_scaled, labels)
        linkage_scores[method] = {'ARI': ari, 'Silhouette': silhouette}
        print(f"{method.upper()}: ARI={ari:.4f}, Silhouette={silhouette:.4f}")
    
    return kmeans_iris, hierarchical_iris, linkage_scores

if __name__ == "__main__":
    kmeans_model, hierarchical_model, scores = problem_4_5()