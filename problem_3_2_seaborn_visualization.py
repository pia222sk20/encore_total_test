"""
문제 3.2: Seaborn을 활용한 다변량 데이터 관계 시각화
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def problem_3_2():
    print("=== 문제 3.2: Seaborn을 활용한 다변량 데이터 관계 시각화 ===")
    
    # 1. 데이터 준비
    titanic = sns.load_dataset('titanic')
    df_numeric = titanic[['age', 'fare', 'pclass', 'survived']].copy()
    
    # 결측값을 중앙값으로 채우기
    df_numeric['age'].fillna(df_numeric['age'].median(), inplace=True)
    df_numeric['fare'].fillna(df_numeric['fare'].median(), inplace=True)
    
    print("선택된 수치형 변수:")
    print(df_numeric.info())
    print(f"\n결측값 확인:")
    print(df_numeric.isnull().sum())
    
    # 2. pairplot 시각화
    print("\nGenerating pairplot...")
    sns.pairplot(df_numeric, hue='survived', palette='viridis')
    plt.suptitle('Pair Plot of Titanic Numeric Features by Survival', y=1.02)
    plt.show()
    
    # 3. 상관계수 행렬 계산
    corr_matrix = df_numeric.corr()
    print("\n상관계수 행렬:")
    print(corr_matrix)
    
    # 4. heatmap 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show()
    
    print("\n=== 분석 결과 ===")
    print("1. Pairplot: 변수들 간의 관계와 생존 여부에 따른 분포 차이를 보여줌")
    print("2. 상관계수 행렬: 변수들 간의 선형 관계 강도를 수치로 표현")
    print("3. Heatmap: 상관계수를 색상으로 시각화하여 패턴을 쉽게 파악")
    
    # 주요 인사이트 출력
    print(f"\n주요 상관관계:")
    print(f"- survived와 pclass: {corr_matrix.loc['survived', 'pclass']:.3f} (음의 상관관계)")
    print(f"- survived와 fare: {corr_matrix.loc['survived', 'fare']:.3f} (양의 상관관계)")
    print(f"- pclass와 fare: {corr_matrix.loc['pclass', 'fare']:.3f} (음의 상관관계)")

if __name__ == "__main__":
    problem_3_2()