"""
문제 4.3: 결정 트리와 랜덤 포레스트를 이용한 분류
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

def problem_4_3():
    print("=== 문제 4.3: 결정 트리와 랜덤 포레스트를 이용한 분류 ===")
    
    # 1. 데이터 로드
    wine = load_wine()
    X, y = wine.data, wine.target
    
    print(f"데이터 형태: {X.shape}")
    print(f"특성 개수: {len(wine.feature_names)}")
    print(f"클래스: {wine.target_names}")
    print(f"클래스별 개수: {pd.Series(y).value_counts().sort_index()}")
    
    # 데이터프레임 생성
    df = pd.DataFrame(X, columns=wine.feature_names)
    df['target'] = y
    df['target_name'] = [wine.target_names[i] for i in y]
    
    # 2. 데이터 탐색
    print(f"\n=== 기본 통계 ===")
    print(df.describe())
    
    # 특성 간 상관관계
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[wine.feature_names].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Wine Dataset Feature Correlation')
    plt.tight_layout()
    plt.show()
    
    # 클래스별 주요 특성 분포
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    important_features = ['alcohol', 'total_phenols', 'flavanoids', 'color_intensity', 'proline', 'od280/od315_of_diluted_wines']
    
    for i, feature in enumerate(important_features):
        ax = axes[i//3, i%3]
        for target_class in range(3):
            class_data = df[df['target'] == target_class][feature]
            ax.hist(class_data, alpha=0.7, label=wine.target_names[target_class], bins=15)
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{feature} Distribution by Class')
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 3. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
    
    # 4. 결정 트리 모델
    print(f"\n=== 결정 트리 모델 ===")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    
    print(f"결정 트리 정확도: {dt_accuracy:.4f}")
    
    # 5. 랜덤 포레스트 모델
    print(f"\n=== 랜덤 포레스트 모델 ===")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"랜덤 포레스트 정확도: {rf_accuracy:.4f}")
    
    # 6. 모델 비교
    print(f"\n=== 모델 성능 비교 ===")
    print(f"결정 트리 정확도: {dt_accuracy:.4f}")
    print(f"랜덤 포레스트 정확도: {rf_accuracy:.4f}")
    print(f"성능 향상: {rf_accuracy - dt_accuracy:.4f}")
    
    # 7. 결정 트리 시각화
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, 
              feature_names=wine.feature_names,
              class_names=wine.target_names,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title('Decision Tree Visualization')
    plt.show()
    
    # 8. 특성 중요도 비교
    dt_importance = pd.DataFrame({
        'feature': wine.feature_names,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    rf_importance = pd.DataFrame({
        'feature': wine.feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    top_dt = dt_importance.head(10)
    plt.barh(range(len(top_dt)), top_dt['importance'], color='skyblue')
    plt.yticks(range(len(top_dt)), top_dt['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Decision Tree - Top 10 Features')
    plt.gca().invert_yaxis()
    
    plt.subplot(1, 2, 2)
    top_rf = rf_importance.head(10)
    plt.barh(range(len(top_rf)), top_rf['importance'], color='lightgreen')
    plt.yticks(range(len(top_rf)), top_rf['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest - Top 10 Features')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== 결정 트리 상위 5개 중요 특성 ===")
    for _, row in dt_importance.head(5).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    print(f"\n=== 랜덤 포레스트 상위 5개 중요 특성 ===")
    for _, row in rf_importance.head(5).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # 9. 혼동 행렬 비교
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 결정 트리 혼동 행렬
    dt_cm = confusion_matrix(y_test, dt_pred)
    sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names, yticklabels=wine.target_names,
                ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Decision Tree Confusion Matrix')
    
    # 랜덤 포레스트 혼동 행렬
    rf_cm = confusion_matrix(y_test, rf_pred)
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=wine.target_names, yticklabels=wine.target_names,
                ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Random Forest Confusion Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # 10. 상세 분류 보고서
    print(f"\n=== 결정 트리 Classification Report ===")
    print(classification_report(y_test, dt_pred, target_names=wine.target_names))
    
    print(f"\n=== 랜덤 포레스트 Classification Report ===")
    print(classification_report(y_test, rf_pred, target_names=wine.target_names))
    
    # 11. 하이퍼파라미터에 따른 성능 변화
    print(f"\n=== 하이퍼파라미터 영향 분석 ===")
    
    # 트리 깊이별 성능
    depths = range(1, 11)
    dt_scores = []
    rf_scores = []
    
    for depth in depths:
        # 결정 트리
        dt_temp = DecisionTreeClassifier(random_state=42, max_depth=depth)
        dt_temp.fit(X_train, y_train)
        dt_scores.append(dt_temp.score(X_test, y_test))
        
        # 랜덤 포레스트
        rf_temp = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=depth)
        rf_temp.fit(X_train, y_train)
        rf_scores.append(rf_temp.score(X_test, y_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(depths, dt_scores, 'o-', label='Decision Tree', linewidth=2)
    plt.plot(depths, rf_scores, 's-', label='Random Forest', linewidth=2)
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Model Performance vs Tree Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 최적 깊이 찾기
    best_dt_depth = depths[np.argmax(dt_scores)]
    best_rf_depth = depths[np.argmax(rf_scores)]
    
    print(f"최적 결정 트리 깊이: {best_dt_depth} (정확도: {max(dt_scores):.4f})")
    print(f"최적 랜덤 포레스트 깊이: {best_rf_depth} (정확도: {max(rf_scores):.4f})")
    
    return dt_model, rf_model

if __name__ == "__main__":
    dt_model, rf_model = problem_4_3()