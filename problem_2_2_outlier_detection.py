"""
문제 2.2: IQR을 이용한 이상값 탐지 및 처리
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def problem_2_2():
    print("=== 문제 2.2: IQR을 이용한 이상값 탐지 및 처리 ===")
    
    # 1. 데이터 생성 (정상 데이터 + 이상값)
    np.random.seed(42)
    
    # 정상 데이터
    normal_data = np.random.normal(50, 15, 200)
    
    # 이상값 추가
    outliers = np.array([120, 125, 130, -10, -15, 135])
    data = np.concatenate([normal_data, outliers])
    
    # DataFrame 생성
    df = pd.DataFrame({'value': data})
    df['index'] = range(len(df))
    
    print(f"전체 데이터 개수: {len(df)}")
    print(f"기본 통계:")
    print(df['value'].describe())
    print("-" * 60)
    
    # 2. IQR 방법으로 이상값 탐지
    print("=== IQR 방법으로 이상값 탐지 ===")
    
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    
    print(f"Q1 (25% 분위수): {Q1:.2f}")
    print(f"Q3 (75% 분위수): {Q3:.2f}")
    print(f"IQR (Q3 - Q1): {IQR:.2f}")
    
    # IQR 기준 이상값 경계 계산
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"하한 경계 (Q1 - 1.5*IQR): {lower_bound:.2f}")
    print(f"상한 경계 (Q3 + 1.5*IQR): {upper_bound:.2f}")
    
    # 이상값 식별
    outliers_mask = (df['value'] < lower_bound) | (df['value'] > upper_bound)
    outliers_data = df[outliers_mask]
    normal_data_filtered = df[~outliers_mask]
    
    print(f"\n탐지된 이상값 개수: {len(outliers_data)}")
    print("이상값들:")
    print(outliers_data)
    print(f"정상 데이터 개수: {len(normal_data_filtered)}")
    print("-" * 60)
    
    # 3. Z-score 방법으로 이상값 탐지 (비교용)
    print("=== Z-score 방법으로 이상값 탐지 (비교) ===")
    
    mean_val = df['value'].mean()
    std_val = df['value'].std()
    threshold = 3  # 3-sigma 규칙
    
    df['z_score'] = np.abs((df['value'] - mean_val) / std_val)
    z_outliers_mask = df['z_score'] > threshold
    z_outliers_data = df[z_outliers_mask]
    
    print(f"평균: {mean_val:.2f}")
    print(f"표준편차: {std_val:.2f}")
    print(f"Z-score 임계값: {threshold}")
    print(f"Z-score 방법으로 탐지된 이상값 개수: {len(z_outliers_data)}")
    print("Z-score 이상값들:")
    print(z_outliers_data[['value', 'z_score']])
    print("-" * 60)
    
    # 4. 이상값 처리 전략들
    print("=== 이상값 처리 전략 ===")
    
    # 전략 1: 삭제
    df_removed = df[~outliers_mask].copy()
    print(f"전략 1 - 삭제 후 데이터 개수: {len(df_removed)}")
    
    # 전략 2: 경계값으로 대체 (Winsorizing)
    df_winsorized = df.copy()
    df_winsorized.loc[df_winsorized['value'] < lower_bound, 'value'] = lower_bound
    df_winsorized.loc[df_winsorized['value'] > upper_bound, 'value'] = upper_bound
    print(f"전략 2 - 경계값 대체: 하한 {lower_bound:.2f}, 상한 {upper_bound:.2f}")
    
    # 전략 3: 평균/중앙값으로 대체
    df_mean_replaced = df.copy()
    median_val = normal_data_filtered['value'].median()
    df_mean_replaced.loc[outliers_mask, 'value'] = median_val
    print(f"전략 3 - 중앙값({median_val:.2f})으로 대체")
    
    # 전략 4: 로그 변환
    df_log = df.copy()
    df_log['value_log'] = np.log1p(df_log['value'] - df_log['value'].min() + 1)
    print("전략 4 - 로그 변환 적용")
    print("-" * 60)
    
    # 5. 각 전략의 통계적 비교
    print("=== 각 전략의 통계적 비교 ===")
    
    strategies = {
        '원본': df['value'],
        '이상값 삭제': df_removed['value'],
        '경계값 대체': df_winsorized['value'],
        '중앙값 대체': df_mean_replaced['value']
    }
    
    comparison_stats = {}
    for name, data in strategies.items():
        comparison_stats[name] = {
            '평균': data.mean(),
            '중앙값': data.median(),
            '표준편차': data.std(),
            '최솟값': data.min(),
            '최댓값': data.max(),
            '왜도': data.skew(),
            '첨도': data.kurtosis()
        }
    
    stats_df = pd.DataFrame(comparison_stats).T.round(3)
    print(stats_df)
    print("-" * 60)
    
    # 6. 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 원본 데이터 박스플롯
    axes[0, 0].boxplot(df['value'])
    axes[0, 0].set_title('원본 데이터')
    axes[0, 0].set_ylabel('값')
    
    # 원본 데이터 히스토그램
    axes[0, 1].hist(df['value'], bins=30, alpha=0.7, color='blue')
    axes[0, 1].axvline(lower_bound, color='red', linestyle='--', label='하한')
    axes[0, 1].axvline(upper_bound, color='red', linestyle='--', label='상한')
    axes[0, 1].set_title('원본 데이터 분포')
    axes[0, 1].set_xlabel('값')
    axes[0, 1].set_ylabel('빈도')
    axes[0, 1].legend()
    
    # 이상값 제거 후
    axes[0, 2].hist(df_removed['value'], bins=30, alpha=0.7, color='green')
    axes[0, 2].set_title('이상값 삭제 후')
    axes[0, 2].set_xlabel('값')
    axes[0, 2].set_ylabel('빈도')
    
    # 경계값 대체
    axes[1, 0].hist(df_winsorized['value'], bins=30, alpha=0.7, color='orange')
    axes[1, 0].set_title('경계값 대체 후')
    axes[1, 0].set_xlabel('값')
    axes[1, 0].set_ylabel('빈도')
    
    # 중앙값 대체
    axes[1, 1].hist(df_mean_replaced['value'], bins=30, alpha=0.7, color='purple')
    axes[1, 1].set_title('중앙값 대체 후')
    axes[1, 1].set_xlabel('값')
    axes[1, 1].set_ylabel('빈도')
    
    # 모든 전략 박스플롯 비교
    box_data = [df['value'], df_removed['value'], df_winsorized['value'], df_mean_replaced['value']]
    box_labels = ['원본', '삭제', '경계값', '중앙값']
    axes[1, 2].boxplot(box_data, labels=box_labels)
    axes[1, 2].set_title('전략별 분포 비교')
    axes[1, 2].set_ylabel('값')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 7. 산점도로 이상값 시각화
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(df['index'], df['value'], c='blue', alpha=0.6)
    plt.scatter(outliers_data['index'], outliers_data['value'], c='red', s=100, alpha=0.8, label='IQR 이상값')
    plt.axhline(lower_bound, color='red', linestyle='--', alpha=0.7)
    plt.axhline(upper_bound, color='red', linestyle='--', alpha=0.7)
    plt.title('IQR 방법으로 탐지된 이상값')
    plt.xlabel('인덱스')
    plt.ylabel('값')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.scatter(df['index'], df['value'], c='blue', alpha=0.6)
    plt.scatter(z_outliers_data['index'], z_outliers_data['value'], c='orange', s=100, alpha=0.8, label='Z-score 이상값')
    plt.title('Z-score 방법으로 탐지된 이상값')
    plt.xlabel('인덱스')
    plt.ylabel('값')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.scatter(df['value'], df['z_score'], alpha=0.6)
    plt.axhline(threshold, color='red', linestyle='--', label=f'임계값: {threshold}')
    plt.title('값 vs Z-score')
    plt.xlabel('값')
    plt.ylabel('Z-score')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    # Q-Q plot (정규분포와의 비교)
    from scipy import stats
    stats.probplot(normal_data_filtered['value'], dist="norm", plot=plt.gca())
    plt.title('Q-Q Plot (이상값 제거 후)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 8. 실제 데이터 예제 (Boston Housing Dataset 대신 생성)
    print("=== 실제 데이터 예제 ===")
    
    # 주택 가격 데이터 시뮬레이션
    np.random.seed(123)
    prices = np.random.lognormal(mean=5, sigma=0.5, size=500) * 100000
    # 일부 극단값 추가
    extreme_prices = np.array([2000000, 2500000, 3000000])  # 매우 비싼 집들
    prices = np.concatenate([prices, extreme_prices])
    
    housing_df = pd.DataFrame({'price': prices})
    
    # IQR 분석
    Q1_price = housing_df['price'].quantile(0.25)
    Q3_price = housing_df['price'].quantile(0.75)
    IQR_price = Q3_price - Q1_price
    
    lower_price = Q1_price - 1.5 * IQR_price
    upper_price = Q3_price + 1.5 * IQR_price
    
    price_outliers = housing_df[(housing_df['price'] < lower_price) | (housing_df['price'] > upper_price)]
    
    print(f"주택 가격 데이터 분석:")
    print(f"전체 주택 수: {len(housing_df)}")
    print(f"가격 이상값 개수: {len(price_outliers)}")
    print(f"이상값 비율: {len(price_outliers)/len(housing_df)*100:.1f}%")
    print(f"가격 범위: ${housing_df['price'].min():,.0f} - ${housing_df['price'].max():,.0f}")
    print(f"이상값 상한선: ${upper_price:,.0f}")
    
    return {
        'original_data': df,
        'outliers_iqr': outliers_data,
        'outliers_zscore': z_outliers_data,
        'processed_data': {
            'removed': df_removed,
            'winsorized': df_winsorized,
            'mean_replaced': df_mean_replaced
        },
        'stats_comparison': stats_df,
        'bounds': {'lower': lower_bound, 'upper': upper_bound}
    }

if __name__ == "__main__":
    results = problem_2_2()