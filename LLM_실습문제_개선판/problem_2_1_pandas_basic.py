"""
문제 2.1: Pandas DataFrame 생성 및 데이터 조회

요구사항:
1. 딕셔너리, CSV, 리스트로 DataFrame 생성
2. head, tail, info, describe 활용
3. 특정 행/열 조회
4. 조건부 필터링
"""

import pandas as pd
import numpy as np

print("="*60)
print("문제 2.1: Pandas DataFrame 생성 및 데이터 조회")
print("="*60)

# 1. 다양한 방법으로 DataFrame 생성
print("\n[1] DataFrame 생성 방법")
print("-" * 60)

# 방법 1: 딕셔너리
data_dict = {
    '이름': ['김철수', '이영희', '박준호', '최수진'],
    '나이': [25, 30, 28, 32],
    '점수': [85, 92, 88, 95],
    '직급': ['사원', '대리', '사원', '과장']
}
df1 = pd.DataFrame(data_dict)
print("방법 1: 딕셔너리로 생성")
print(df1)

# 방법 2: 리스트
data_list = [
    ['김철수', 25, 85],
    ['이영희', 30, 92],
    ['박준호', 28, 88],
    ['최수진', 32, 95]
]
df2 = pd.DataFrame(data_list, columns=['이름', '나이', '점수'])
print("\n방법 2: 리스트로 생성")
print(df2)

# 방법 3: NumPy 배열
data_array = np.random.rand(4, 3)
df3 = pd.DataFrame(data_array, columns=['A', 'B', 'C'])
print("\n방법 3: NumPy 배열로 생성")
print(df3)


# 2. 데이터 조회 함수들
print("\n[2] 데이터 조회")
print("-" * 60)
df = df1.copy()

print("전체 데이터:")
print(df)

print("\n앞 2행 (head):")
print(df.head(2))

print("\n뒤 2행 (tail):")
print(df.tail(2))

print("\nDataFrame 정보 (info):")
df.info()

print("\n기본 통계 (describe):")
print(df.describe())

print("\n데이터 타입 확인:")
print(df.dtypes)


# 3. 행과 열 선택
print("\n[3] 행과 열 선택")
print("-" * 60)

print("특정 열 선택 - '이름' 열:")
print(df['이름'])
print(f"타입: {type(df['이름'])}")

print("\n여러 열 선택 - ['이름', '점수']:")
print(df[['이름', '점수']])

print("\n특정 행 선택 - 0번 행:")
print(df.iloc[0])

print("\n여러 행 선택 - 1~2번 행:")
print(df.iloc[1:3])

print("\n특정 위치 선택 - df.iloc[1, 2] (2번 행, 3번 열):")
print(f"값: {df.iloc[1, 2]}")


# 4. 조건부 필터링
print("\n[4] 조건부 필터링")
print("-" * 60)

print("나이 > 28인 행:")
print(df[df['나이'] > 28])

print("\n점수 >= 90인 행:")
print(df[df['점수'] >= 90])

print("\n나이 > 27 AND 점수 > 85인 행:")
print(df[(df['나이'] > 27) & (df['점수'] > 85)])

print("\n이름이 '김철수' 또는 '최수진'인 행:")
print(df[df['이름'].isin(['김철수', '최수진'])])

print("\n" + "="*60)
print("모든 작업 완료")
print("="*60)
