"""
문제 2.3: 결측치 처리 및 데이터 정제

요구사항:
1. 결측치 확인 및 제거
2. 결측치 채우기 (평균, 이전값, 다음값)
3. 중복값 찾기 및 제거
4. 데이터 타입 변환
"""

import pandas as pd
import numpy as np

print("="*60)
print("문제 2.3: 결측치 처리 및 데이터 정제")
print("="*60)

# 결측치가 있는 샘플 데이터 생성
data = {
    '이름': ['김철수', '이영희', None, '최수진', '정민준', '한미진', '이소영', '조준환'],
    '나이': [25, np.nan, 28, 32, np.nan, 26, 29, 31],
    '급여': [3000, 3500, 4000, np.nan, 3800, 3000, 3200, 3600],
    '입사일': ['2020-01-15', '2019-03-20', '2021-05-10', '2020-07-22', 
              '2021-02-14', '2022-06-01', '2020-11-30', '2019-12-05']
}
df = pd.DataFrame(data)

print("원본 데이터 (결측치 포함):")
print(df)


# 1. 결측치 확인
print("\n[1] 결측치 확인")
print("-" * 60)

print("각 열의 결측치 개수:")
print(df.isnull().sum())

print("\n결측치 비율 (%):")
missing_ratio = (df.isnull().sum() / len(df)) * 100
print(missing_ratio)

print("\n결측치 존재 위치:")
print(df.isnull())

print("\n전체 결측치 개수:", df.isnull().sum().sum())


# 2. 결측치 제거
print("\n[2] 결측치 제거")
print("-" * 60)

print("결측치 있는 행 모두 제거 (dropna):")
df_dropped = df.dropna()
print(df_dropped)
print(f"원본: {len(df)}행 -> 제거 후: {len(df_dropped)}행")

print("\n특정 열의 결측치만 기준으로 제거 (subset):")
df_dropped2 = df.dropna(subset=['나이', '급여'])
print(df_dropped2)

print("\n어느 한 열이라도 결측치 있으면 제거 (any):")
df_dropped3 = df.dropna(how='any')
print(f"남은 행 수: {len(df_dropped3)}")

print("\n모든 열이 결측치면 제거 (all):")
df_dropped4 = df.dropna(how='all')
print(f"남은 행 수: {len(df_dropped4)}")


# 3. 결측치 채우기
print("\n[3] 결측치 채우기")
print("-" * 60)

print("3-1. 고정값으로 채우기:")
df_filled1 = df.copy()
df_filled1['이름'] = df_filled1['이름'].fillna('정보없음')
df_filled1['급여'] = df_filled1['급여'].fillna(0)
print(df_filled1)

print("\n3-2. 평균으로 채우기:")
df_filled2 = df.copy()
avg_age = df_filled2['나이'].mean()
print(f"나이 평균: {avg_age:.1f}")
df_filled2['나이'] = df_filled2['나이'].fillna(avg_age)
print(df_filled2[['이름', '나이']])

print("\n3-3. 이전값(forward fill)으로 채우기:")
df_filled3 = df.copy()
df_filled3['급여'] = df_filled3['급여'].ffill()
print(df_filled3[['이름', '급여']])

print("\n3-4. 다음값(backward fill)으로 채우기:")
df_filled4 = df.copy()
df_filled4['급여'] = df_filled4['급여'].bfill()
print(df_filled4[['이름', '급여']])


# 4. 중복값 처리
print("\n[4] 중복값 처리")
print("-" * 60)

# 중복값 있는 데이터
data_dup = {
    'A': [1, 2, 2, 3, 3, 3],
    'B': ['x', 'y', 'y', 'z', 'z', 'z']
}
df_dup = pd.DataFrame(data_dup)
print("중복값 있는 데이터:")
print(df_dup)

print("\n중복값 확인:")
print(df_dup.duplicated())

print("\n중복값 개수:", df_dup.duplicated().sum())

print("\n중복값 행 보기:")
print(df_dup[df_dup.duplicated(keep=False)])

print("\n중복값 제거 (첫 번째 유지):")
df_unique = df_dup.drop_duplicates()
print(df_unique)


# 5. 데이터 타입 변환
print("\n[5] 데이터 타입 변환")
print("-" * 60)

df_convert = df.copy()
print("원본 타입:")
print(df_convert.dtypes)

print("\n날짜 타입으로 변환:")
df_convert['입사일'] = pd.to_datetime(df_convert['입사일'])
print(df_convert.dtypes)
print(f"첫 번째 입사일: {df_convert['입사일'].iloc[0]}")
print(f"연도 추출: {df_convert['입사일'].dt.year.iloc[0]}")

print("\n" + "="*60)
print("모든 작업 완료")
print("="*60)
