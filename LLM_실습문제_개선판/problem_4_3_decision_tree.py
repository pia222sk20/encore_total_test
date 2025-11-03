"""
문제 4.3: 결정 트리 분류 (Decision Tree)

요구사항:
1. 결정 트리 모델 훈련
2. 특성 중요도 분석
3. 모델 성능 평가
4. 트리 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("="*60)
print("문제 4.3: 결정 트리 분류 (Decision Tree)")
print("="*60)

# 1. 데이터 생성
print("\n[1] 분류 데이터셋 생성")
print("-" * 60)

X, y = make_classification(n_samples=300, n_features=4, n_informative=3,
                          n_classes=3, n_clusters_per_class=1, random_state=42)
feature_names = ['특성 1', '특성 2', '특성 3', '특성 4']
target_names = ['클래스 0', '클래스 1', '클래스 2']

print(f"데이터 형태: {X.shape}")
print(f"특성명: {feature_names}")
print(f"클래스명: {target_names}")
print(f"클래스 분포: {np.bincount(y)}")


# 2. 데이터 분할
print("\n[2] 학습/테스트 데이터 분할")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"학습 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")


# 3. 모델 훈련
print("\n[3] 결정 트리 모델 훈련")
print("-" * 60)

tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)

print(f"모델 복잡도 (최대 깊이): {tree_model.get_depth()}")
print(f"노드 수: {tree_model.tree_.node_count}")


# 4. 예측
print("\n[4] 예측 수행")
print("-" * 60)

y_pred = tree_model.predict(X_test)
y_pred_proba = tree_model.predict_proba(X_test)

print("처음 10개 예측 결과:")
for i in range(min(10, len(y_test))):
    print(f"  실제: {target_names[y_test[i]]}, "
          f"예측: {target_names[y_pred[i]]}, "
          f"신뢰도: {y_pred_proba[i, int(y_pred[i])]:.4f}")


# 5. 모델 평가
print("\n[5] 모델 평가")
print("-" * 60)

accuracy = accuracy_score(y_test, y_pred)
print(f"정확도 (Accuracy): {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\n혼동행렬:")
print(cm)

print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


# 6. 특성 중요도
print("\n[6] 특성 중요도")
print("-" * 60)

importance = tree_model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]

print("특성 중요도 순위:")
for rank, idx in enumerate(sorted_idx, 1):
    print(f"  {rank}. {iris.feature_names[idx]}: {importance[idx]:.4f}")


# 7. 시각화
print("\n[7] 결과 시각화")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 특성 중요도
ax = axes[0]
indices = np.argsort(importance)
ax.barh(range(len(indices)), importance[indices])
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([iris.feature_names[i] for i in indices])
ax.set_xlabel('중요도')
ax.set_title('특성 중요도', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# 혼동행렬
ax = axes[1]
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set_title('혼동행렬', fontweight='bold')
ax.set_ylabel('실제')
ax.set_xlabel('예측')
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(iris.target_names)
ax.set_yticklabels(iris.target_names)

for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black',
                fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('problem_4_3_decision_tree.png', dpi=100)
print("✓ 시각화 저장: problem_4_3_decision_tree.png")
plt.close()


# 8. 트리 구조 시각화
print("\n[8] 트리 구조 시각화")
print("-" * 60)

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(tree_model, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True, ax=ax, fontsize=10)
plt.tight_layout()
plt.savefig('problem_4_3_tree_structure.png', dpi=100, bbox_inches='tight')
print("✓ 트리 구조 저장: problem_4_3_tree_structure.png")
plt.close()

print("\n" + "="*60)
print("결정 트리 분류 완료")
print("="*60)
