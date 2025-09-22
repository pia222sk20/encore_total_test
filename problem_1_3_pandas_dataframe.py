"""
문제 1.3: Pandas DataFrame 생성 및 조건부 인덱싱
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def problem_1_3():
    print("=== 문제 1.3: Pandas DataFrame 생성 및 조건부 인덱싱 ===")
    
    # 1. 딕셔너리를 이용한 DataFrame 생성
    np.random.seed(42)
    
    data = {
        'name': ['김철수', '이영희', '박민수', '최지영', '정태현', '한소희', '윤상호', '배수진'],
        'age': [25, 30, 35, 28, 42, 31, 29, 27],
        'city': ['서울', '부산', '대구', '서울', '광주', '서울', '부산', '대구'],
        'salary': [3500, 4200, 3800, 4500, 5200, 3900, 3600, 4100],
        'department': ['개발', '마케팅', '개발', '기획', '개발', '마케팅', '기획', '개발'],
        'experience': [2, 5, 8, 3, 12, 6, 4, 3]
    }
    
    df = pd.DataFrame(data)
    print("생성된 DataFrame:")
    print(df)
    print(f"\nDataFrame 정보:")
    print(f"형태: {df.shape}")
    print(f"열 이름: {list(df.columns)}")
    print(f"데이터 타입:\n{df.dtypes}")
    print("-" * 60)
    
    # 2. 기본 정보 확인
    print("=== 기본 통계 정보 ===")
    print(df.describe())
    print(f"\n각 열의 고유값 개수:")
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"{col}: {df[col].nunique()}개 ({list(df[col].unique())})")
        else:
            print(f"{col}: 최소 {df[col].min()}, 최대 {df[col].max()}")
    print("-" * 60)
    
    # 3. 단일 조건 필터링
    print("=== 단일 조건 필터링 ===")
    
    # 나이가 30 이상인 직원
    age_filter = df['age'] >= 30
    print("나이가 30 이상인 직원:")
    print(df[age_filter])
    print(f"조건을 만족하는 직원 수: {age_filter.sum()}명")
    print()
    
    # 서울에 거주하는 직원
    seoul_filter = df['city'] == '서울'
    print("서울에 거주하는 직원:")
    print(df[seoul_filter])
    print(f"서울 거주 직원 수: {seoul_filter.sum()}명")
    print()
    
    # 급여가 4000 이상인 직원
    high_salary_filter = df['salary'] >= 4000
    print("급여가 4000 이상인 직원:")
    print(df[high_salary_filter])
    print("-" * 60)
    
    # 4. 복합 조건 필터링 (AND, OR, NOT)
    print("=== 복합 조건 필터링 ===")
    
    # AND 조건: 서울 거주 AND 급여 4000 이상
    and_condition = (df['city'] == '서울') & (df['salary'] >= 4000)
    print("서울 거주 AND 급여 4000 이상:")
    print(df[and_condition])
    print()
    
    # OR 조건: 개발팀 OR 급여 4500 이상
    or_condition = (df['department'] == '개발') | (df['salary'] >= 4500)
    print("개발팀 OR 급여 4500 이상:")
    print(df[or_condition])
    print()
    
    # NOT 조건: 서울이 아닌 지역
    not_condition = ~(df['city'] == '서울')
    print("서울이 아닌 지역 거주자:")
    print(df[not_condition])
    print()
    
    # 복잡한 조건: (개발팀 AND 경력 5년 이상) OR (급여 4500 이상)
    complex_condition = ((df['department'] == '개발') & (df['experience'] >= 5)) | (df['salary'] >= 4500)
    print("(개발팀 AND 경력 5년 이상) OR (급여 4500 이상):")
    print(df[complex_condition])
    print("-" * 60)
    
    # 5. isin() 메서드 활용
    print("=== isin() 메서드 활용 ===")
    
    # 특정 도시들에 거주하는 직원
    target_cities = ['서울', '부산']
    city_isin = df['city'].isin(target_cities)
    print(f"{target_cities}에 거주하는 직원:")
    print(df[city_isin])
    print()
    
    # 특정 부서들에 속한 직원
    target_departments = ['개발', '기획']
    dept_isin = df['department'].isin(target_departments)
    print(f"{target_departments} 부서 직원:")
    print(df[dept_isin])
    print("-" * 60)
    
    # 6. query() 메서드 활용
    print("=== query() 메서드 활용 ===")
    
    # 문자열 형태의 조건식 사용
    query_result1 = df.query('age >= 30 and salary >= 4000')
    print("age >= 30 and salary >= 4000:")
    print(query_result1)
    print()
    
    query_result2 = df.query('city == "서울" and department == "개발"')
    print('city == "서울" and department == "개발":')
    print(query_result2)
    print()
    
    query_result3 = df.query('salary > age * 100')
    print("salary > age * 100 (급여가 나이*100보다 큰 경우):")
    print(query_result3)
    print("-" * 60)
    
    # 7. loc과 iloc를 이용한 조건부 선택
    print("=== loc과 iloc를 이용한 조건부 선택 ===")
    
    # loc: 라벨 기반 인덱싱 + 조건
    loc_result = df.loc[df['age'] >= 30, ['name', 'age', 'salary']]
    print("나이 30 이상인 직원의 이름, 나이, 급여:")
    print(loc_result)
    print()
    
    # 특정 인덱스 범위와 열 선택
    iloc_result = df.iloc[2:5, [0, 1, 3]]  # 2~4번째 행, 0,1,3번째 열
    print("2~4번째 행의 이름, 나이, 급여:")
    print(iloc_result)
    print("-" * 60)
    
    # 8. 조건부 데이터 수정
    print("=== 조건부 데이터 수정 ===")
    
    # DataFrame 복사본 생성
    df_modified = df.copy()
    
    # 조건에 따른 새 열 추가
    df_modified['salary_grade'] = '보통'
    df_modified.loc[df_modified['salary'] >= 4500, 'salary_grade'] = '높음'
    df_modified.loc[df_modified['salary'] < 3700, 'salary_grade'] = '낮음'
    
    print("급여 등급이 추가된 DataFrame:")
    print(df_modified[['name', 'salary', 'salary_grade']])
    print()
    
    # 나이대 그룹 추가
    conditions = [
        df_modified['age'] < 30,
        (df_modified['age'] >= 30) & (df_modified['age'] < 40),
        df_modified['age'] >= 40
    ]
    choices = ['20대', '30대', '40대+']
    df_modified['age_group'] = np.select(conditions, choices, default='기타')
    
    print("나이대 그룹이 추가된 DataFrame:")
    print(df_modified[['name', 'age', 'age_group']])
    print("-" * 60)
    
    # 9. 그룹별 통계 및 시각화
    print("=== 그룹별 통계 ===")
    
    # 부서별 통계
    dept_stats = df.groupby('department').agg({
        'salary': ['mean', 'max', 'min'],
        'age': 'mean',
        'experience': 'mean'
    }).round(2)
    print("부서별 통계:")
    print(dept_stats)
    print()
    
    # 도시별 통계
    city_stats = df.groupby('city').agg({
        'salary': 'mean',
        'age': 'mean'
    }).round(2)
    print("도시별 통계:")
    print(city_stats)
    print("-" * 60)
    
    # 10. 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 부서별 급여 분포
    df.boxplot(column='salary', by='department', ax=axes[0, 0])
    axes[0, 0].set_title('부서별 급여 분포')
    axes[0, 0].set_xlabel('부서')
    axes[0, 0].set_ylabel('급여')
    
    # 도시별 직원 수
    city_counts = df['city'].value_counts()
    axes[0, 1].bar(city_counts.index, city_counts.values)
    axes[0, 1].set_title('도시별 직원 수')
    axes[0, 1].set_xlabel('도시')
    axes[0, 1].set_ylabel('직원 수')
    
    # 나이와 급여의 관계
    axes[1, 0].scatter(df['age'], df['salary'], c=df['experience'], cmap='viridis')
    axes[1, 0].set_xlabel('나이')
    axes[1, 0].set_ylabel('급여')
    axes[1, 0].set_title('나이 vs 급여 (색상: 경력)')
    
    # 경력과 급여의 관계
    axes[1, 1].scatter(df['experience'], df['salary'])
    axes[1, 1].set_xlabel('경력 (년)')
    axes[1, 1].set_ylabel('급여')
    axes[1, 1].set_title('경력 vs 급여')
    
    plt.tight_layout()
    plt.show()
    
    # 11. 조건별 요약 정보
    print("=== 조건별 요약 정보 ===")
    conditions_summary = {
        '전체': len(df),
        '서울 거주': len(df[df['city'] == '서울']),
        '개발팀': len(df[df['department'] == '개발']),
        '급여 4000 이상': len(df[df['salary'] >= 4000]),
        '경력 5년 이상': len(df[df['experience'] >= 5]),
        '30대': len(df[(df['age'] >= 30) & (df['age'] < 40)])
    }
    
    for condition, count in conditions_summary.items():
        percentage = (count / len(df)) * 100
        print(f"{condition}: {count}명 ({percentage:.1f}%)")
    
    return {
        'original_df': df,
        'modified_df': df_modified,
        'dept_stats': dept_stats,
        'city_stats': city_stats,
        'conditions_summary': conditions_summary
    }

if __name__ == "__main__":
    results = problem_1_3()