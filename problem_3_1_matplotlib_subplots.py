"""
문제 3.1: Matplotlib을 활용한 다중 플롯 생성
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def problem_3_1():
    print("=== 문제 3.1: Matplotlib을 활용한 다중 플롯 생성 ===")
    
    # 1. 데이터 준비
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    
    print("Iris 데이터셋 정보:")
    print(df.info())
    print(f"\n품종별 데이터 개수:")
    print(df['species'].value_counts())
    
    # 2. 2x2 subplot 그리드 생성
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Comprehensive Iris Dataset Visualization', fontsize=16)
    
    # 3 & 4. 각 subplot에 플롯 그리기
    
    # ax[0,0]: Sepal Length Histogram
    ax[0,0].hist(df['sepal length (cm)'], bins=20, color='skyblue', edgecolor='black')
    ax[0,0].set_title('Histogram of Sepal Length')
    ax[0,0].set_xlabel('Sepal Length (cm)')
    ax[0,0].set_ylabel('Frequency')
    ax[0,0].grid(True, alpha=0.3)
    
    # ax[0,1]: Sepal Width Box Plot
    ax[0,1].boxplot(df['sepal width (cm)'])
    ax[0,1].set_title('Box Plot of Sepal Width')
    ax[0,1].set_ylabel('Sepal Width (cm)')
    ax[0,1].set_xticks([])  # x축 눈금 제거
    ax[0,1].grid(True, alpha=0.3)
    
    # ax[1,0]: Petal Length vs Width Scatter Plot
    colors = ['red', 'green', 'blue']
    for i, species in enumerate(iris.target_names):
        species_data = df[df['species'] == i]
        ax[1,0].scatter(species_data['petal length (cm)'], species_data['petal width (cm)'], 
                       c=colors[i], label=species, alpha=0.7)
    ax[1,0].set_title('Scatter Plot of Petal Length vs Width')
    ax[1,0].set_xlabel('Petal Length (cm)')
    ax[1,0].set_ylabel('Petal Width (cm)')
    ax[1,0].legend()
    ax[1,0].grid(True, alpha=0.3)
    
    # ax[1,1]: Species Count Bar Plot
    species_counts = df['species'].value_counts().sort_index()
    ax[1,1].bar(iris.target_names, species_counts, color=['#ff9999','#66b3ff','#99ff99'])
    ax[1,1].set_title('Count of Each Species')
    ax[1,1].set_xlabel('Species')
    ax[1,1].set_ylabel('Count')
    ax[1,1].grid(True, alpha=0.3)
    
    # 각 막대 위에 숫자 표시
    for i, count in enumerate(species_counts):
        ax[1,1].text(i, count + 1, str(count), ha='center', va='bottom')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # suptitle과 겹치지 않도록 조정
    plt.show()
    
    print("\n=== 플롯 설명 ===")
    print("1. 히스토그램: Sepal Length의 분포를 보여줌")
    print("2. 박스플롯: Sepal Width의 중앙값, 사분위수, 이상값을 보여줌")
    print("3. 산점도: Petal Length와 Width의 관계를 종별로 색상 구분하여 표시")
    print("4. 막대그래프: 각 종의 데이터 개수를 보여줌 (각각 50개씩)")

if __name__ == "__main__":
    problem_3_1()