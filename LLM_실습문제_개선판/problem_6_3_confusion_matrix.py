"""
문제 6.3: 혼동행렬 (Confusion Matrix) 및 분류 지표

요구사항:
1. 혼동행렬 계산 및 해석
2. 정밀도, 재현율, F1 점수 계산
3. 클래스별 성능 평가
4. 다중 분류 문제 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, classification_report, roc_auc_score)

print("="*60)
print("문제 6.3: 혼동행렬 및 분류 지표")
print("="*60)

# 1. 불균형 이진 분류 데이터
print("\n[1] 불균형 이진 분류 데이터 생성")
print("-" * 60)

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          weights=[0.8, 0.2], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"클래스 분포: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
print(f"불균형 비율: 0={np.sum(y==0)/len(y):.1%}, 1={np.sum(y==1)/len(y):.1%}")


# 2. 모델 훈련
print("\n[2] 모델 훈련")
print("-" * 60)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("✓ 모델 훈련 완료")


# 3. 혼동행렬
print("\n[3] 혼동행렬 분석")
print("-" * 60)

cm = confusion_matrix(y_test, y_pred)
print("혼동행렬:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nTN (진음성): {tn}")
print(f"FP (거짓양성): {fp}")
print(f"FN (거짓음성): {fn}")
print(f"TP (진양성): {tp}")


# 4. 분류 지표 계산
print("\n[4] 분류 지표 계산")
print("-" * 60)

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = recall  # 재현율과 동일

print(f"정확도 (Accuracy): {accuracy:.4f}")
print(f"정밀도 (Precision): {precision:.4f} - TP / (TP + FP)")
print(f"재현율 (Recall): {recall:.4f} - TP / (TP + FN)")
print(f"특이도 (Specificity): {specificity:.4f} - TN / (TN + FP)")
print(f"F1 점수: {f1:.4f} - 2 * (정밀도 * 재현율) / (정밀도 + 재현율)")


# 5. 클래스별 성능
print("\n[5] 클래스별 성능")
print("-" * 60)

print("클래스 0:")
print(f"  정밀도: {precision_score(y_test, y_pred, pos_label=0, zero_division=0):.4f}")
print(f"  재현율: {recall_score(y_test, y_pred, pos_label=0, zero_division=0):.4f}")
print(f"  F1: {f1_score(y_test, y_pred, pos_label=0, zero_division=0):.4f}")

print("\n클래스 1:")
print(f"  정밀도: {precision_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
print(f"  재현율: {recall_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
print(f"  F1: {f1_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")

print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=['클래스 0', '클래스 1']))


# 6. 시각화
print("\n[6] 혼동행렬 시각화")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 혼동행렬 히트맵
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
                fontsize=16, fontweight='bold')

# 분류 지표 비교
ax = axes[1]
metrics = ['정확도', '정밀도', '재현율', '특이도', 'F1 점수']
values = [accuracy, precision, recall, specificity, f1]
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

bars = ax.bar(metrics, values, color=colors_bar, edgecolor='black', linewidth=1.5)
ax.set_ylabel('점수', fontsize=11)
ax.set_title('분류 지표', fontweight='bold', fontsize=12)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

# 값 표시
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('problem_6_3_confusion_matrix.png', dpi=100)
print("✓ 시각화 저장: problem_6_3_confusion_matrix.png")
plt.close()

print("\n" + "="*60)
print("혼동행렬 분석 완료")
print("="*60)
