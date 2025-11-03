"""
문제 6.1: 교차 검증 (Cross Validation)

요구사항:
1. K-Fold 교차 검증 수행
2. Stratified K-Fold 사용
3. Leave-One-Out 교차 검증
4. 결과 비교 및 분석
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import (cross_val_score, KFold, StratifiedKFold,
                                     LeaveOneOut, cross_validate)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

print("="*60)
print("문제 6.1: 교차 검증 (Cross Validation)")
print("="*60)

# 데이터 로드 (간단한 버전)
print("\n[1] 데이터 로드")
print("-" * 60)

np.random.seed(42)
X = np.random.randn(150, 4)
y = np.hstack([np.zeros(50), np.ones(50), np.ones(50) * 2])

print(f"데이터 형태: {X.shape}")
print(f"클래스: 0={np.sum(y==0)}, 1={np.sum(y==1)}, 2={np.sum(y==2)}")


# 2. K-Fold 교차 검증
print("\n[2] K-Fold 교차 검증")
print("-" * 60)

model = LogisticRegression(max_iter=200)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores_kfold = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"K-Fold (k=5) 점수: {scores_kfold}")
print(f"평균: {scores_kfold.mean():.4f} ± {scores_kfold.std():.4f}")


# 3. Stratified K-Fold
print("\n[3] Stratified K-Fold 교차 검증")
print("-" * 60)

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_skfold = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')

print(f"Stratified K-Fold (k=5) 점수: {scores_skfold}")
print(f"평균: {scores_skfold.mean():.4f} ± {scores_skfold.std():.4f}")


# 4. Leave-One-Out (작은 데이터)
print("\n[4] Leave-One-Out 교차 검증 (소규모 데이터에서만 수행)")
print("-" * 60)

X_small = X[:50]
y_small = y[:50]

loo = LeaveOneOut()
scores_loo = cross_val_score(model, X_small, y_small, cv=loo, scoring='accuracy')

print(f"Leave-One-Out 점수: {scores_loo[:10]}... (처음 10개)")
print(f"평균: {scores_loo.mean():.4f}")


# 5. 다양한 메트릭으로 평가
print("\n[5] 다양한 메트릭으로 교차 검증")
print("-" * 60)

scoring = {'accuracy': 'accuracy', 'precision_macro': 'precision_macro',
           'recall_macro': 'recall_macro', 'f1_macro': 'f1_macro'}

cv_results = cross_validate(model, X, y, cv=skfold, scoring=scoring)

print("\n교차 검증 결과:")
results_df = pd.DataFrame({
    'Accuracy': cv_results['test_accuracy'],
    'Precision': cv_results['test_precision_macro'],
    'Recall': cv_results['test_recall_macro'],
    'F1-Score': cv_results['test_f1_macro']
})
print(results_df)

print("\n평균 점수:")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    mean = results_df[metric].mean()
    std = results_df[metric].std()
    print(f"  {metric}: {mean:.4f} ± {std:.4f}")


# 6. 모델 비교
print("\n[6] 여러 모델 비교")
print("-" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

comparison = {}
for name, clf in models.items():
    scores = cross_val_score(clf, X, y, cv=skfold, scoring='accuracy')
    comparison[name] = {
        'Mean': scores.mean(),
        'Std': scores.std(),
        'Scores': scores
    }
    print(f"{name}:")
    print(f"  평균: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  점수: {scores}")

print("\n" + "="*60)
print("교차 검증 완료")
print("="*60)
