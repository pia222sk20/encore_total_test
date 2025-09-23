"""
문제 2.1: 결측값 식별 및 처리 전략 비교
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def problem_2_1():
    print("=== 문제 2.1: 결측값 식별 및 처리 전략 비교 ===")
    
    # 1. Titanic 데이터 로드 및 결측값 분석
    titanic = sns.load_dataset('titanic')
    
    print("=== 결측값 분석 ===")
    print("데이터 형태:", titanic.shape)
    print("\n각 열의 결측값 개수:")
    missing_counts = titanic.isnull().sum()
    missing_ratios = (titanic.isnull().sum() / len(titanic)) * 100
    
    missing_info = pd.DataFrame({
        '결측값 개수': missing_counts,
        '결측값 비율(%)': missing_ratios.round(2)
    })
    print(missing_info[missing_info['결측값 개수'] > 0])
    
    # age 열 집중 분석
    age_missing_count = titanic['age'].isnull().sum()
    age_missing_ratio = (age_missing_count / len(titanic)) * 100
    print(f"\nage 열 결측값 분석:")
    print(f"- 결측값 개수: {age_missing_count}개")
    print(f"- 전체 데이터 대비 비율: {age_missing_ratio:.2f}%")
    print("-" * 60)
    
    # 2. 전략 1: 삭제 (Deletion)
    print("=== 전략 1: 삭제 ===")
    
    # age 열에 결측값이 있는 행 삭제
    titanic_dropped = titanic.dropna(subset=['age'])
    print(f"원본 데이터 행 개수: {len(titanic)}")
    print(f"age 결측값 삭제 후 행 개수: {len(titanic_dropped)}")
    print(f"삭제된 행 개수: {len(titanic) - len(titanic_dropped)}")
    print(f"데이터 손실률: {((len(titanic) - len(titanic_dropped)) / len(titanic)) * 100:.2f}%")
    print("-" * 60)
    
    # 3. 전략 2: 대체 (Imputation)
    print("=== 전략 2: 대체 ===")
    
    # 평균값으로 대체
    age_mean = titanic['age'].mean()
    age_filled_mean = titanic['age'].fillna(age_mean)
    print(f"age 열 평균값: {age_mean:.2f}")
    print(f"평균값 대체 후 결측값 개수: {age_filled_mean.isnull().sum()}")
    
    # 중앙값으로 대체
    age_median = titanic['age'].median()
    age_filled_median = titanic['age'].fillna(age_median)
    print(f"age 열 중앙값: {age_median:.2f}")
    print(f"중앙값 대체 후 결측값 개수: {age_filled_median.isnull().sum()}")
    
    # 최빈값으로 대체
    age_mode = titanic['age'].mode()[0]
    age_filled_mode = titanic['age'].fillna(age_mode)
    print(f"age 열 최빈값: {age_mode:.2f}")
    print(f"최빈값 대체 후 결측값 개수: {age_filled_mode.isnull().sum()}")
    print("-" * 60)
    
    # 4. 고급 대체 전략
    print("=== 고급 대체 전략 ===")
    
    # 그룹별 대체 (성별과 좌석등급별 평균값)
    titanic_group_filled = titanic.copy()
    titanic_group_filled['age'] = titanic_group_filled.groupby(['sex', 'pclass'])['age'].transform(
        lambda x: x.fillna(x.mean())
    )
    
    print("성별-좌석등급별 평균값으로 대체:")
    group_means = titanic.groupby(['sex', 'pclass'])['age'].mean().round(2)
    print(group_means)
    print(f"그룹별 대체 후 결측값 개수: {titanic_group_filled['age'].isnull().sum()}")
    
    # 전진 채우기 (Forward Fill)
    age_ffill = titanic['age'].fillna(method='ffill')
    print(f"전진 채우기 후 결측값 개수: {age_ffill.isnull().sum()}")
    
    # 후진 채우기 (Backward Fill)
    age_bfill = titanic['age'].fillna(method='bfill')
    print(f"후진 채우기 후 결측값 개수: {age_bfill.isnull().sum()}")
    print("-" * 60)
    
    # 5. 각 전략의 통계적 비교
    print("=== 통계적 비교 ===")
    
    strategies = {
        '원본 (결측값 제외)': titanic['age'].dropna(),
        '삭제 후': titanic_dropped['age'],
        '평균값 대체': age_filled_mean,
        '중앙값 대체': age_filled_median,
        '최빈값 대체': age_filled_mode,
        '그룹별 대체': titanic_group_filled['age']
    }
    
    comparison_stats = {}
    for name, data in strategies.items():
        comparison_stats[name] = {
            '평균': data.mean(),
            '중앙값': data.median(),
            '표준편차': data.std(),
            '최솟값': data.min(),
            '최댓값': data.max(),
            '데이터 수': len(data)
        }
    
    stats_df = pd.DataFrame(comparison_stats).T.round(2)
    print("각 전략별 통계:")
    print(stats_df)
    print("-" * 60)
    
    # 6. 시각화 비교
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 원본 데이터 분포
    axes[0, 0].hist(titanic['age'].dropna(), bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('원본 데이터 (결측값 제외)')
    axes[0, 0].set_xlabel('나이')
    axes[0, 0].set_ylabel('빈도')
    
    # 평균값 대체
    axes[0, 1].hist(age_filled_mean, bins=30, alpha=0.7, color='green')
    axes[0, 1].axvline(age_mean, color='red', linestyle='--', label=f'대체값: {age_mean:.1f}')
    axes[0, 1].set_title('평균값 대체')
    axes[0, 1].set_xlabel('나이')
    axes[0, 1].set_ylabel('빈도')
    axes[0, 1].legend()
    
    # 중앙값 대체
    axes[0, 2].hist(age_filled_median, bins=30, alpha=0.7, color='orange')
    axes[0, 2].axvline(age_median, color='red', linestyle='--', label=f'대체값: {age_median:.1f}')
    axes[0, 2].set_title('중앙값 대체')
    axes[0, 2].set_xlabel('나이')
    axes[0, 2].set_ylabel('빈도')
    axes[0, 2].legend()
    
    # 최빈값 대체
    axes[1, 0].hist(age_filled_mode, bins=30, alpha=0.7, color='purple')
    axes[1, 0].axvline(age_mode, color='red', linestyle='--', label=f'대체값: {age_mode:.1f}')
    axes[1, 0].set_title('최빈값 대체')
    axes[1, 0].set_xlabel('나이')
    axes[1, 0].set_ylabel('빈도')
    axes[1, 0].legend()
    
    # 그룹별 대체
    axes[1, 1].hist(titanic_group_filled['age'], bins=30, alpha=0.7, color='red')
    axes[1, 1].set_title('그룹별 평균 대체')
    axes[1, 1].set_xlabel('나이')
    axes[1, 1].set_ylabel('빈도')
    
    # 박스플롯 비교
    box_data = [
        titanic['age'].dropna(),
        age_filled_mean,
        age_filled_median,
        titanic_group_filled['age']
    ]
    box_labels = ['원본', '평균', '중앙값', '그룹별']
    axes[1, 2].boxplot(box_data, labels=box_labels)
    axes[1, 2].set_title('분포 비교 (박스플롯)')
    axes[1, 2].set_ylabel('나이')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 7. 결측값 패턴 분석
    print("=== 결측값 패턴 분석 ===")
    
    # 결측값이 있는 열들만 선택
    cols_with_missing = ['age', 'embarked', 'deck']
    missing_pattern = titanic[cols_with_missing].isnull()
    
    # 결측값 패턴별 빈도
    pattern_counts = missing_pattern.value_counts()
    print("결측값 패턴 분석:")
    for pattern, count in pattern_counts.head(10).items():
        print(f"{pattern}: {count}명")
    
    # 결측값 히트맵
    plt.figure(figsize=(10, 8))
    sns.heatmap(titanic.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('결측값 패턴 히트맵')
    plt.xlabel('열')
    plt.ylabel('행 (샘플)')
    plt.show()
    
    # 8. 생존율에 미치는 영향 분석
    print("=== 결측값 처리가 생존율에 미치는 영향 ===")
    
    survival_comparison = {}
    
    # 원본 데이터 (결측값 있는 상태)
    survival_comparison['원본'] = titanic['survived'].mean()
    
    # 삭제 후
    survival_comparison['삭제 후'] = titanic_dropped['survived'].mean()
    
    # 각 대체 전략별 생존율 (age를 이용한 분석을 위해 새로운 DataFrame 생성)
    for strategy_name, age_data in [('평균값 대체', age_filled_mean), 
                                   ('중앙값 대체', age_filled_median),
                                   ('그룹별 대체', titanic_group_filled['age'])]:
        temp_df = titanic.copy()
        temp_df['age'] = age_data
        survival_comparison[strategy_name] = temp_df['survived'].mean()
    
    print("각 전략별 전체 생존율:")
    for strategy, survival_rate in survival_comparison.items():
        print(f"{strategy}: {survival_rate:.4f}")
    
    # 나이대별 생존율 비교 (그룹별 대체 vs 원본)
    age_groups = pd.cut(titanic_group_filled['age'], bins=[0, 12, 18, 35, 60, 100], 
                       labels=['어린이', '청소년', '청년', '중년', '노년'])
    age_survival = titanic_group_filled.groupby(age_groups)['survived'].mean()
    
    print(f"\n나이대별 생존율 (그룹별 대체 후):")
    for age_group, survival_rate in age_survival.items():
        print(f"{age_group}: {survival_rate:.4f}")
    
    return {
        'original_data': titanic,
        'dropped_data': titanic_dropped,
        'filled_strategies': {
            'mean': age_filled_mean,
            'median': age_filled_median,
            'mode': age_filled_mode,
            'group': titanic_group_filled['age']
        },
        'stats_comparison': stats_df,
        'survival_comparison': survival_comparison
    }

if __name__ == "__main__":
    results = problem_2_1()