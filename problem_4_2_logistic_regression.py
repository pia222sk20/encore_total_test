"""
문제 4.2: 로지스틱 회귀를 이용한 유방암 진단 분류
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def problem_4_2():
    print("=== 문제 4.2: 로지스틱 회귀를 이용한 유방암 진단 분류 ===")
    
    # 1. 데이터 로드
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    print(f"데이터 형태: {X.shape}")
    print(f"특성 개수: {len(cancer.feature_names)}")
    print(f"클래스: {cancer.target_names}")
    print(f"클래스별 개수: {pd.Series(y).value_counts()}")
    
    # 2. 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n스케일링 전 평균: {X.mean():.2f}, 표준편차: {X.std():.2f}")
    print(f"스케일링 후 평균: {X_scaled.mean():.2f}, 표준편차: {X_scaled.std():.2f}")
    
    # 3. 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=101
    )
    
    print(f"\n훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
    
    # 4. 모델 학습 및 예측
    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n모델 정확도: {accuracy:.4f}")
    
    # 5. 성능 평가
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 혼동 행렬 설명
    print("\n=== 혼동 행렬 해석 ===")
    tn, fp, fn, tp = cm.ravel()
    print(f"TN (True Negative): {tn} - 실제 악성을 악성으로 올바르게 예측")
    print(f"FP (False Positive): {fp} - 실제 악성을 양성으로 잘못 예측 (Type I Error)")
    print(f"FN (False Negative): {fn} - 실제 양성을 악성으로 잘못 예측 (Type II Error) - **가장 위험**")
    print(f"TP (True Positive): {tp} - 실제 양성을 양성으로 올바르게 예측")
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=cancer.target_names, yticklabels=cancer.target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Breast Cancer Classification')
    plt.show()
    
    # 특성 중요도 (로지스틱 회귀 계수)
    feature_importance = pd.DataFrame({
        'feature': cancer.feature_names,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\n=== 상위 10개 중요 특성 (절댓값 기준) ===")
    print(feature_importance.head(10))

if __name__ == "__main__":
    problem_4_2()