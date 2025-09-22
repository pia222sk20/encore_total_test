"""
문제 4.6: K-means 군집화를 이용한 고객 세분화
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def problem_4_6():
    print("=== 문제 4.6: K-means 군집화를 이용한 고객 세분화 ===")
    
    # 1. 고객 데이터 생성
    np.random.seed(42)
    
    # 고객 유형별 데이터 생성
    # 그룹 1: 고소득, 고지출 (VIP 고객)
    vip_customers = np.random.multivariate_normal([80, 85], [[100, 50], [50, 100]], 50)
    
    # 그룹 2: 중소득, 중지출 (일반 고객)
    regular_customers = np.random.multivariate_normal([50, 50], [[80, 20], [20, 80]], 100)
    
    # 그룹 3: 고소득, 저지출 (절약형 고객)
    saving_customers = np.random.multivariate_normal([75, 25], [[90, -30], [-30, 60]], 75)
    
    # 데이터 결합
    all_data = np.vstack([vip_customers, regular_customers, saving_customers])
    
    # DataFrame 생성
    df = pd.DataFrame(all_data, columns=['연소득', '연간지출'])
    df['고객ID'] = [f'C{i+1:03d}' for i in range(len(df))]
    
    # 추가 특성 생성
    df['지출비율'] = df['연간지출'] / df['연소득'] * 100
    df['절약성향'] = df['연소득'] - df['연간지출']
    
    print(f"생성된 고객 데이터 개수: {len(df)}")
    print("\n기본 통계:")
    print(df[['연소득', '연간지출', '지출비율', '절약성향']].describe())
    print("-" * 60)
    
    # 2. 데이터 탐색 및 시각화
    print("=== 데이터 탐색 ===")
    
    # 기본 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 산점도
    axes[0, 0].scatter(df['연소득'], df['연간지출'], alpha=0.6, color='blue')
    axes[0, 0].set_xlabel('연소득')
    axes[0, 0].set_ylabel('연간지출')
    axes[0, 0].set_title('연소득 vs 연간지출')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 히스토그램
    axes[0, 1].hist(df['연소득'], bins=20, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('연소득')
    axes[0, 1].set_ylabel('빈도')
    axes[0, 1].set_title('연소득 분포')
    
    axes[1, 0].hist(df['연간지출'], bins=20, alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('연간지출')
    axes[1, 0].set_ylabel('빈도')
    axes[1, 0].set_title('연간지출 분포')
    
    # 지출비율 vs 절약성향
    axes[1, 1].scatter(df['지출비율'], df['절약성향'], alpha=0.6, color='red')
    axes[1, 1].set_xlabel('지출비율(%)')
    axes[1, 1].set_ylabel('절약성향')
    axes[1, 1].set_title('지출비율 vs 절약성향')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 3. 데이터 전처리
    print("=== 데이터 전처리 ===")
    
    # 클러스터링에 사용할 특성 선택
    features = ['연소득', '연간지출', '지출비율', '절약성향']
    X = df[features].values
    
    print(f"클러스터링 특성: {features}")
    print(f"데이터 형태: {X.shape}")
    
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"표준화 전 범위:")
    for i, feature in enumerate(features):
        print(f"  {feature}: {X[:, i].min():.1f} ~ {X[:, i].max():.1f}")
    
    print(f"표준화 후 범위:")
    for i, feature in enumerate(features):
        print(f"  {feature}: {X_scaled[:, i].min():.1f} ~ {X_scaled[:, i].max():.1f}")
    print("-" * 60)
    
    # 4. 최적 클러스터 수 결정 (Elbow Method)
    print("=== 최적 클러스터 수 결정 ===")
    
    K_range = range(1, 11)
    inertias = []
    silhouette_scores = []
    
    for k in K_range:
        if k == 1:
            inertias.append(0)
            silhouette_scores.append(0)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # 최적 K 선택 (Silhouette Score 기준)
    optimal_k = K_range[1:][np.argmax(silhouette_scores[1:])]
    print(f"최적 클러스터 수 (Silhouette 기준): {optimal_k}")
    
    # Elbow Method 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].set_xlabel('클러스터 수 (K)')
    axes[0].set_ylabel('Inertia (WCSS)')
    axes[0].set_title('Elbow Method')
    axes[0].grid(True)
    
    axes[1].plot(K_range[1:], silhouette_scores[1:], 'ro-')
    axes[1].axvline(optimal_k, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('클러스터 수 (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    print("-" * 60)
    
    # 5. K-means 클러스터링 수행
    print("=== K-means 클러스터링 수행 ===")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # 결과를 데이터프레임에 추가
    df['클러스터'] = cluster_labels
    
    print(f"사용된 클러스터 수: {optimal_k}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}")
    
    # 클러스터별 고객 수
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    print(f"\n클러스터별 고객 수:")
    for cluster, count in cluster_counts.items():
        percentage = count / len(df) * 100
        print(f"  클러스터 {cluster}: {count}명 ({percentage:.1f}%)")
    print("-" * 60)
    
    # 6. 클러스터별 특성 분석
    print("=== 클러스터별 특성 분석 ===")
    
    cluster_analysis = df.groupby('클러스터')[features].agg(['mean', 'std']).round(2)
    print("클러스터별 평균 및 표준편차:")
    print(cluster_analysis)
    
    # 클러스터별 요약 통계
    print(f"\n클러스터별 상세 분석:")
    for cluster in sorted(df['클러스터'].unique()):
        cluster_data = df[df['클러스터'] == cluster]
        print(f"\n=== 클러스터 {cluster} ===")
        print(f"고객 수: {len(cluster_data)}명")
        print(f"평균 연소득: {cluster_data['연소득'].mean():.1f}")
        print(f"평균 연간지출: {cluster_data['연간지출'].mean():.1f}")
        print(f"평균 지출비율: {cluster_data['지출비율'].mean():.1f}%")
        print(f"평균 절약성향: {cluster_data['절약성향'].mean():.1f}")
    print("-" * 60)
    
    # 7. 클러스터 시각화
    print("=== 클러스터 시각화 ===")
    
    # 색상 설정
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    cluster_colors = [colors[i] for i in cluster_labels]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 원본 특성 공간에서의 클러스터
    scatter = axes[0, 0].scatter(df['연소득'], df['연간지출'], c=cluster_labels, cmap='viridis', alpha=0.6)
    axes[0, 0].set_xlabel('연소득')
    axes[0, 0].set_ylabel('연간지출')
    axes[0, 0].set_title('클러스터 결과 (연소득 vs 연간지출)')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # 클러스터 중심점 표시
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    axes[0, 0].scatter(centroids_original[:, 0], centroids_original[:, 1], 
                      c='red', marker='x', s=200, linewidths=3, label='중심점')
    axes[0, 0].legend()
    
    # 지출비율 vs 절약성향
    scatter2 = axes[0, 1].scatter(df['지출비율'], df['절약성향'], c=cluster_labels, cmap='viridis', alpha=0.6)
    axes[0, 1].set_xlabel('지출비율(%)')
    axes[0, 1].set_ylabel('절약성향')
    axes[0, 1].set_title('클러스터 결과 (지출비율 vs 절약성향)')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # 클러스터별 특성 비교 (평행좌표)
    cluster_means = df.groupby('클러스터')[features].mean()
    cluster_means_scaled = pd.DataFrame(scaler.transform(cluster_means), 
                                      index=cluster_means.index, 
                                      columns=cluster_means.columns)
    
    for cluster in cluster_means_scaled.index:
        axes[0, 2].plot(range(len(features)), cluster_means_scaled.loc[cluster], 
                       'o-', label=f'클러스터 {cluster}', linewidth=2)
    axes[0, 2].set_xticks(range(len(features)))
    axes[0, 2].set_xticklabels(features, rotation=45)
    axes[0, 2].set_ylabel('표준화된 값')
    axes[0, 2].set_title('클러스터별 특성 프로필')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 클러스터별 박스플롯
    for i, feature in enumerate(['연소득', '연간지출']):
        ax = axes[1, i]
        df.boxplot(column=feature, by='클러스터', ax=ax)
        ax.set_title(f'클러스터별 {feature} 분포')
        ax.set_xlabel('클러스터')
        ax.set_ylabel(feature)
    
    # 클러스터 크기 파이차트
    axes[1, 2].pie(cluster_counts.values, labels=[f'클러스터 {i}' for i in cluster_counts.index], 
                  autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title('클러스터별 고객 분포')
    
    plt.tight_layout()
    plt.show()
    
    # 8. PCA를 이용한 2차원 시각화
    print("=== PCA를 이용한 차원 축소 시각화 ===")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA 설명 분산 비율: {pca.explained_variance_ratio_}")
    print(f"누적 설명 분산: {pca.explained_variance_ratio_.sum():.3f}")
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.xlabel(f'PC1 (설명분산: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 (설명분산: {pca.explained_variance_ratio_[1]:.3f})')
    plt.title('PCA 차원 축소 후 클러스터 시각화')
    plt.colorbar(scatter)
    
    # PCA 공간에서의 클러스터 중심점
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='중심점')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 9. 고객 세분화 전략 수립
    print("=== 고객 세분화 전략 ===")
    
    # 클러스터별 특성을 바탕으로 고객 유형 명명
    cluster_profiles = {}
    for cluster in sorted(df['클러스터'].unique()):
        cluster_data = df[df['클러스터'] == cluster]
        avg_income = cluster_data['연소득'].mean()
        avg_spending = cluster_data['연간지출'].mean()
        avg_ratio = cluster_data['지출비율'].mean()
        avg_saving = cluster_data['절약성향'].mean()
        
        # 고객 유형 분류
        if avg_income > 70 and avg_spending > 70:
            customer_type = "프리미엄 고객"
            strategy = "고급 서비스 제공, VIP 혜택"
        elif avg_income > 60 and avg_ratio < 40:
            customer_type = "절약형 고객"
            strategy = "투자 상품 추천, 적금 상품"
        elif avg_spending > 60:
            customer_type = "소비형 고객"
            strategy = "할인 혜택, 포인트 적립"
        else:
            customer_type = "일반 고객"
            strategy = "기본 서비스, 맞춤형 상품"
        
        cluster_profiles[cluster] = {
            '고객유형': customer_type,
            '마케팅전략': strategy,
            '고객수': len(cluster_data),
            '평균소득': avg_income,
            '평균지출': avg_spending,
            '지출비율': avg_ratio
        }
    
    print("클러스터별 고객 유형 및 마케팅 전략:")
    for cluster, profile in cluster_profiles.items():
        print(f"\n클러스터 {cluster}: {profile['고객유형']}")
        print(f"  고객 수: {profile['고객수']}명")
        print(f"  평균 소득: {profile['평균소득']:.1f}")
        print(f"  평균 지출: {profile['평균지출']:.1f}")
        print(f"  지출 비율: {profile['지출비율']:.1f}%")
        print(f"  마케팅 전략: {profile['마케팅전략']}")
    
    return {
        'customer_data': df,
        'kmeans_model': kmeans,
        'scaler': scaler,
        'optimal_k': optimal_k,
        'cluster_profiles': cluster_profiles,
        'pca_data': X_pca,
        'silhouette_scores': silhouette_scores
    }

if __name__ == "__main__":
    results = problem_4_6()