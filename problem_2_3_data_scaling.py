"""
문제 2.3: Scikit-learn을 활용한 데이터 스케일링 비교
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def problem_2_3():
    print("=== 문제 2.3: Scikit-learn을 활용한 데이터 스케일링 비교 ===")
    
    # 1. 와인 데이터셋 로드 및 특성 선택
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)[['alcohol', 'malic_acid']]
    
    print("원본 데이터:")
    print(df.head())
    print(f"데이터 형태: {df.shape}")
    print(f"Alcohol 범위: {df['alcohol'].min():.2f} ~ {df['alcohol'].max():.2f}")
    print(f"Malic Acid 범위: {df['malic_acid'].min():.2f} ~ {df['malic_acid'].max():.2f}")
    
    # 2. 스케일링 전 데이터 시각화
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(df['alcohol'], df['malic_acid'])
    plt.title('Original Data')
    plt.xlabel('Alcohol')
    plt.ylabel('Malic Acid')
    plt.grid(True)
    
    # 3. StandardScaler 적용 및 시각화
    scaler_std = StandardScaler()
    df_std = scaler_std.fit_transform(df)
    
    plt.subplot(1, 3, 2)
    plt.scatter(df_std[:, 0], df_std[:, 1])
    plt.title('StandardScaler Transformed Data')
    plt.xlabel('Alcohol (Standardized)')
    plt.ylabel('Malic Acid (Standardized)')
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.grid(True)
    
    # 4. MinMaxScaler 적용 및 시각화
    scaler_minmax = MinMaxScaler()
    df_minmax = scaler_minmax.fit_transform(df)
    
    plt.subplot(1, 3, 3)
    plt.scatter(df_minmax[:, 0], df_minmax[:, 1])
    plt.title('MinMaxScaler Transformed Data')
    plt.xlabel('Alcohol (Normalized)')
    plt.ylabel('Malic Acid (Normalized)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 5. 결과 비교 분석
    print("\n원본 데이터 기술 통계:")
    print(df.describe().loc[['mean', 'std', 'min', 'max']])
    
    print("\nStandardScaler 변환 후 기술 통계:")
    print(pd.DataFrame(df_std, columns=['alcohol', 'malic_acid']).describe().loc[['mean', 'std', 'min', 'max']])
    
    print("\nMinMaxScaler 변환 후 기술 통계:")
    print(pd.DataFrame(df_minmax, columns=['alcohol', 'malic_acid']).describe().loc[['mean', 'std', 'min', 'max']])
    
    print("\n=== 스케일링 방법 비교 ===")
    print("StandardScaler: 평균 0, 표준편차 1로 변환 (정규분포 가정)")
    print("- 장점: 정규분포를 따르는 데이터에 효과적")
    print("- 단점: 이상값에 민감함")
    print("\nMinMaxScaler: 최솟값 0, 최댓값 1로 변환 (범위 고정)")
    print("- 장점: 모든 특성이 동일한 범위로 변환")
    print("- 단점: 이상값에 매우 민감함")

if __name__ == "__main__":
    problem_2_3()