"""
문제 4.2: 로지스틱 회귀 (Logistic Regression)

요구사항:
1. 이진 분류 데이터셋 준비
2. 로지스틱 회귀 모델 훈련
3. 정확도, 정밀도, 재현율 계산
4. 혼동행렬 및 ROC 곡선 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, classification_report, roc_curve, auc)

print("="*60)
print("문제 4.2: 로지스틱 회귀 (Logistic Regression)")
print("="*60)

# 1. 데이터셋 생성
print("\n[1] 이진 분류 데이터셋 생성")
print("-" * 60)

X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                          n_redundant=0, random_state=42)

print(f"데이터 크기: {X.shape}")
print(f"클래스 분포: 0 = {(y==0).sum()}, 1 = {(y==1).sum()}")


# 2. 데이터 분할
print("\n[2] 학습/테스트 데이터 분할")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"학습 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")


# 3. 모델 훈련
print("\n[3] 로지스틱 회귀 모델 훈련")
print("-" * 60)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

print(f"모델 계수: {model.coef_}")
print(f"모델 절편: {model.intercept_}")


# 4. 예측
print("\n[4] 예측 수행")
print("-" * 60)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"처음 10개의 예측값: {y_pred[:10]}")
print(f"처음 10개의 확률값:")
for i in range(10):
    print(f"  Sample {i}: Class 0={y_pred_proba[i, 0]:.4f}, Class 1={y_pred_proba[i, 1]:.4f}")


# 5. 모델 평가
print("\n[5] 모델 평가")
print("-" * 60)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"정확도 (Accuracy): {accuracy:.4f}")
print(f"정밀도 (Precision): {precision:.4f}")
print(f"재현율 (Recall): {recall:.4f}")

print("\n분류 보고서:")
print(classification_report(y_test, y_pred))

print("\n혼동행렬 (Confusion Matrix):")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")


# 6. ROC 곡선
print("\n[6] ROC 곡선 계산")
print("-" * 60)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)
print(f"AUC 점수: {roc_auc:.4f}")


# 7. 시각화
print("\n[7] 결과 시각화")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 혼동행렬
ax = axes[0]
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set_title('혼동행렬', fontweight='bold', fontsize=12)
ax.set_ylabel('실제', fontsize=11)
ax.set_xlabel('예측', fontsize=11)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['0', '1'])
ax.set_yticklabels(['0', '1'])

# 수치 표시
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black',
                fontsize=14, fontweight='bold')

# ROC 곡선
ax = axes[1]
ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC 곡선 (AUC={roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='임의의 분류')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC 곡선', fontweight='bold', fontsize=12)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('problem_4_2_logistic_regression.png', dpi=100)
print("✓ 시각화 저장: problem_4_2_logistic_regression.png")
plt.close()

print("\n" + "="*60)
print("로지스틱 회귀 완료")
print("="*60)
