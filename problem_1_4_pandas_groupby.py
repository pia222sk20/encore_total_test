"""
문제 1.4: Pandas GroupBy를 활용한 집계 데이터 분석
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def problem_1_4():
    print("=== 문제 1.4: Pandas GroupBy를 활용한 집계 데이터 분석 ===")
    
    # 1. Titanic 데이터 로드
    titanic = sns.load_dataset('titanic')
    print("Titanic 데이터 정보:")
    print(f"데이터 형태: {titanic.shape}")
    print(f"열 정보:")
    print(titanic.info())
    print("\n첫 5행:")
    print(titanic.head())
    print("-" * 60)
    
    # 2. 좌석 등급(pclass)별 생존율 계산
    print("=== 좌석 등급별 생존율 분석 ===")
    
    survival_rate_by_pclass = titanic.groupby('pclass')['survived'].mean()
    print("좌석 등급별 생존율:")
    for pclass, rate in survival_rate_by_pclass.items():
        print(f"{pclass}등석: {rate:.3f} ({rate*100:.1f}%)")
    
    print(f"\n결과 해석:")
    print(f"- 1등석의 생존율이 가장 높음: {survival_rate_by_pclass[1]:.3f}")
    print(f"- 3등석의 생존율이 가장 낮음: {survival_rate_by_pclass[3]:.3f}")
    print(f"- 1등석과 3등석의 생존율 차이: {survival_rate_by_pclass[1] - survival_rate_by_pclass[3]:.3f}")
    print("-" * 60)
    
    # 3. 성별과 좌석 등급별 다중 집계
    print("=== 성별 및 좌석 등급별 집계 분석 ===")
    
    # agg() 함수를 사용한 다중 집계
    agg_result = titanic.groupby(['sex', 'pclass']).agg({
        'age': ['mean', 'std', 'count'],
        'fare': ['max', 'min', 'mean'],
        'survived': 'mean'
    }).round(2)
    
    print("성별 및 좌석 등급별 집계 결과:")
    print(agg_result)
    print("-" * 60)
    
    # 4. 결과 분석
    print("=== 상세 분석 ===")
    
    # 평균 나이가 가장 많은 그룹
    age_mean_series = agg_result['age']['mean']
    max_age_group = age_mean_series.idxmax()
    max_age_value = age_mean_series.max()
    print(f"가장 평균 나이가 많은 그룹: {max_age_group} -> {max_age_value}세")
    
    # 최대 요금이 가장 높은 그룹
    fare_max_series = agg_result['fare']['max']
    max_fare_group = fare_max_series.idxmax()
    max_fare_value = fare_max_series.max()
    print(f"가장 최대 요금이 높은 그룹: {max_fare_group} -> ${max_fare_value}")
    
    # 생존율이 가장 높은 그룹
    survival_series = agg_result['survived']['mean']
    max_survival_group = survival_series.idxmax()
    max_survival_value = survival_series.max()
    print(f"가장 생존율이 높은 그룹: {max_survival_group} -> {max_survival_value:.3f}")
    
    # 생존율이 가장 낮은 그룹
    min_survival_group = survival_series.idxmin()
    min_survival_value = survival_series.min()
    print(f"가장 생존율이 낮은 그룹: {min_survival_group} -> {min_survival_value:.3f}")
    print("-" * 60)
    
    # 5. 추가 그룹별 분석
    print("=== 추가 그룹별 분석 ===")
    
    # 나이대별 분석 (연령대 구간 생성)
    titanic_copy = titanic.copy()
    titanic_copy['age_group'] = pd.cut(titanic_copy['age'], 
                                     bins=[0, 12, 18, 35, 60, 100], 
                                     labels=['어린이', '청소년', '청년', '중년', '노년'])
    
    age_group_analysis = titanic_copy.groupby('age_group').agg({
        'survived': ['count', 'sum', 'mean'],
        'fare': 'mean'
    }).round(3)
    
    print("연령대별 분석:")
    print(age_group_analysis)
    print()
    
    # 승선 항구별 분석
    embark_analysis = titanic.groupby('embark_town').agg({
        'survived': ['count', 'mean'],
        'fare': 'mean',
        'age': 'mean'
    }).round(3)
    
    print("승선 항구별 분석:")
    print(embark_analysis)
    print("-" * 60)
    
    # 6. 피벗 테이블 생성
    print("=== 피벗 테이블 분석 ===")
    
    # 성별과 좌석등급별 생존율 피벗 테이블
    survival_pivot = titanic.pivot_table(
        values='survived', 
        index='sex', 
        columns='pclass', 
        aggfunc='mean'
    ).round(3)
    
    print("성별-좌석등급별 생존율 피벗 테이블:")
    print(survival_pivot)
    print()
    
    # 연령대와 성별별 평균 요금 피벗 테이블
    fare_pivot = titanic_copy.pivot_table(
        values='fare',
        index='age_group',
        columns='sex',
        aggfunc='mean'
    ).round(2)
    
    print("연령대-성별별 평균 요금 피벗 테이블:")
    print(fare_pivot)
    print("-" * 60)
    
    # 7. 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 좌석등급별 생존율
    survival_rate_by_pclass.plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('좌석등급별 생존율')
    axes[0, 0].set_xlabel('좌석등급')
    axes[0, 0].set_ylabel('생존율')
    axes[0, 0].set_xticklabels(['1등석', '2등석', '3등석'], rotation=0)
    
    # 성별-좌석등급별 생존율 히트맵
    sns.heatmap(survival_pivot, annot=True, cmap='RdYlBu_r', ax=axes[0, 1])
    axes[0, 1].set_title('성별-좌석등급별 생존율')
    
    # 연령대별 생존자 수
    age_group_counts = titanic_copy.groupby('age_group')['survived'].sum()
    age_group_counts.plot(kind='bar', ax=axes[0, 2], color='lightgreen')
    axes[0, 2].set_title('연령대별 생존자 수')
    axes[0, 2].set_xlabel('연령대')
    axes[0, 2].set_ylabel('생존자 수')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 성별별 생존율
    sex_survival = titanic.groupby('sex')['survived'].mean()
    sex_survival.plot(kind='bar', ax=axes[1, 0], color=['pink', 'lightblue'])
    axes[1, 0].set_title('성별 생존율')
    axes[1, 0].set_xlabel('성별')
    axes[1, 0].set_ylabel('생존율')
    axes[1, 0].set_xticklabels(['여성', '남성'], rotation=0)
    
    # 승선 항구별 승객 수
    embark_counts = titanic['embark_town'].value_counts()
    embark_counts.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%')
    axes[1, 1].set_title('승선 항구별 승객 분포')
    axes[1, 1].set_ylabel('')
    
    # 좌석등급별 요금 분포
    titanic.boxplot(column='fare', by='pclass', ax=axes[1, 2])
    axes[1, 2].set_title('좌석등급별 요금 분포')
    axes[1, 2].set_xlabel('좌석등급')
    axes[1, 2].set_ylabel('요금')
    
    plt.tight_layout()
    plt.show()
    
    # 8. 고급 그룹 분석
    print("=== 고급 그룹 분석 ===")
    
    # 가족 크기 계산 (sibsp + parch + 1)
    titanic_copy['family_size'] = titanic_copy['sibsp'] + titanic_copy['parch'] + 1
    titanic_copy['family_type'] = pd.cut(titanic_copy['family_size'], 
                                       bins=[0, 1, 4, 11], 
                                       labels=['혼자', '소가족', '대가족'])
    
    # 가족 유형별 생존율
    family_survival = titanic_copy.groupby('family_type')['survived'].mean()
    print("가족 유형별 생존율:")
    for family_type, rate in family_survival.items():
        print(f"{family_type}: {rate:.3f}")
    print()
    
    # 요금 구간별 분석
    titanic_copy['fare_range'] = pd.qcut(titanic_copy['fare'], 
                                       q=4, 
                                       labels=['저가', '중하', '중상', '고가'])
    
    fare_range_analysis = titanic_copy.groupby('fare_range').agg({
        'survived': 'mean',
        'pclass': lambda x: x.mode()[0] if not x.empty else None,  # 최빈값
        'age': 'mean'
    }).round(3)
    
    print("요금 구간별 분석:")
    print(fare_range_analysis)
    print("-" * 60)
    
    # 9. 교차 분석 (복합 조건)
    print("=== 교차 분석 ===")
    
    # 여성, 1등석, 성인의 생존율
    complex_condition = (titanic['sex'] == 'female') & (titanic['pclass'] == 1) & (titanic['age'] >= 18)
    female_first_adult_survival = titanic[complex_condition]['survived'].mean()
    print(f"여성, 1등석, 성인의 생존율: {female_first_adult_survival:.3f}")
    
    # 남성, 3등석, 청년의 생존율
    complex_condition2 = (titanic['sex'] == 'male') & (titanic['pclass'] == 3) & (titanic['age'].between(18, 35))
    male_third_young_survival = titanic[complex_condition2]['survived'].mean()
    print(f"남성, 3등석, 청년(18-35세)의 생존율: {male_third_young_survival:.3f}")
    
    # 차이 분석
    survival_gap = female_first_adult_survival - male_third_young_survival
    print(f"두 그룹 간 생존율 차이: {survival_gap:.3f}")
    
    return {
        'survival_by_class': survival_rate_by_pclass,
        'multi_group_agg': agg_result,
        'survival_pivot': survival_pivot,
        'age_group_analysis': age_group_analysis,
        'family_survival': family_survival
    }

if __name__ == "__main__":
    results = problem_1_4()