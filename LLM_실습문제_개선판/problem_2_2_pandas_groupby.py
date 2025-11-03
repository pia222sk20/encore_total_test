"""
문제 2.2: Pandas GroupBy 및 데이터 집계

요구사항:
1. 그룹별 통계 계산
2. 다중 열 그룹화
3. 커스텀 집계 함수
4. 그룹별 정렬 및 순위 매기기
"""

import pandas as pd
import numpy as np

print("="*60)
print("문제 2.2: Pandas GroupBy 및 데이터 집계")
print("="*60)

# 샘플 데이터 생성
data = {
    '부서': ['영업', '영업', '개발', '개발', '개발', '인사', '인사', '영업'],
    '이름': ['김철수', '이영희', '박준호', '최수진', '정민준', '한미진', '이소영', '조준환'],
    '급여': [3000, 3500, 4000, 4200, 3800, 3000, 3200, 3600],
    '보너스': [500, 600, 800, 900, 700, 300, 400, 550],
    '경력년수': [3, 5, 7, 6, 4, 2, 3, 4]
}
df = pd.DataFrame(data)

print("원본 데이터:")
print(df)


# 1. 그룹별 단일 통계
print("\n[1] 그룹별 단일 통계")
print("-" * 60)

print("부서별 평균 급여:")
avg_salary = df.groupby('부서')['급여'].mean()
print(avg_salary)
print(f"\n타입: {type(avg_salary)}")

print("\n부서별 급여 합계:")
print(df.groupby('부서')['급여'].sum())

print("\n부서별 인원 수:")
print(df.groupby('부서').size())


# 2. 그룹별 다중 통계
print("\n[2] 그룹별 다중 통계")
print("-" * 60)

print("부서별 급여 통계 (평균, 합계, 최대, 최소):")
group_stats = df.groupby('부서')['급여'].agg(['mean', 'sum', 'max', 'min', 'count'])
print(group_stats)

print("\n부서별 급여, 보너스 통계:")
stats = df.groupby('부서')[['급여', '보너스']].agg(['mean', 'std'])
print(stats)


# 3. 커스텀 집계 함수
print("\n[3] 커스텀 집계 함수")
print("-" * 60)

def range_func(x):
    """최댓값 - 최솟값"""
    return x.max() - x.min()

print("부서별 급여 범위 (최대 - 최소):")
salary_range = df.groupby('부서')['급여'].agg(range_func)
print(salary_range)

print("\n람다 함수로 계산 - 급여 표준편차 / 평균:")
cv = df.groupby('부서')['급여'].agg(lambda x: x.std() / x.mean())
print(cv)

print("\n여러 커스텀 함수 적용:")
multi_agg = df.groupby('부서')['급여'].agg(['mean', 'median', range_func, 'std'])
print(multi_agg)
print("컬럼명: mean, median, range_func, std")


# 4. 그룹별 정렬 및 순위
print("\n[4] 그룹별 정렬 및 순위")
print("-" * 60)

print("각 부서 내 급여 기준 상위 2명:")
top2 = df.groupby('부서', group_keys=False).apply(
    lambda x: x.nlargest(2, '급여')
)
print(top2[['부서', '이름', '급여']])

print("\n각 부서별 급여 순위:")
df['부서내순위'] = df.groupby('부서')['급여'].rank(method='min', ascending=False)
print(df[['부서', '이름', '급여', '부서내순위']])

print("\n부서별 평균 급여 상위 정렬:")
dept_avg = df.groupby('부서')['급여'].mean().sort_values(ascending=False)
print(dept_avg)

print("\n" + "="*60)
print("모든 작업 완료")
print("="*60)
