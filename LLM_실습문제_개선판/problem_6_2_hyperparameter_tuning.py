"""
문제 6.2: 하이퍼파라미터 튜닝 (Hyperparameter Tuning)

요구사항:
1. GridSearchCV로 최적 파라미터 탐색
2. RandomizedSearchCV로 무작위 탐색
3. 최적 파라미터 결과 분석
4. 성능 향상 비교
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("문제 6.2: 하이퍼파라미터 튜닝")
print("="*60)

# 1. 데이터 준비
print("\n[1] 데이터 준비")
print("-" * 60)

X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"학습 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")


# 2. 기본 모델 성능
print("\n[2] 기본 모델 성능")
print("-" * 60)

base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_train, y_train)
base_pred = base_model.predict(X_test)
base_accuracy = accuracy_score(y_test, base_pred)

print(f"기본 모델 정확도: {base_accuracy:.4f}")


# 3. GridSearchCV
print("\n[3] GridSearchCV로 최적 파라미터 탐색")
print("-" * 60)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=1
)

print("GridSearchCV 실행 중... (잠깐 시간이 걸릴 수 있습니다)")
grid_search.fit(X_train, y_train)

print(f"\n✓ GridSearchCV 완료")
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 교차검증 점수: {grid_search.best_score_:.4f}")

# 최적 모델로 예측
grid_pred = grid_search.predict(X_test)
grid_accuracy = accuracy_score(y_test, grid_pred)
print(f"테스트 정확도: {grid_accuracy:.4f}")
print(f"개선도: {(grid_accuracy - base_accuracy) * 100:.2f}%")


# 4. RandomizedSearchCV
print("\n[4] RandomizedSearchCV로 무작위 탐색")
print("-" * 60)

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("RandomizedSearchCV 실행 중...")
random_search.fit(X_train, y_train)

print(f"\n✓ RandomizedSearchCV 완료")
print(f"최적 파라미터: {random_search.best_params_}")
print(f"최고 교차검증 점수: {random_search.best_score_:.4f}")

random_pred = random_search.predict(X_test)
random_accuracy = accuracy_score(y_test, random_pred)
print(f"테스트 정확도: {random_accuracy:.4f}")
print(f"개선도: {(random_accuracy - base_accuracy) * 100:.2f}%")


# 5. 결과 비교
print("\n[5] 결과 비교")
print("-" * 60)

comparison_df = pd.DataFrame({
    '모델': ['기본 모델', 'GridSearchCV', 'RandomizedSearchCV'],
    '정확도': [base_accuracy, grid_accuracy, random_accuracy],
    '개선도 (%)': [0, (grid_accuracy - base_accuracy) * 100, (random_accuracy - base_accuracy) * 100]
})

print(comparison_df.to_string(index=False))


# 6. GridSearchCV 결과 분석
print("\n[6] 파라미터 영향도 분석")
print("-" * 60)

# 최상위 10개 결과
results_df = pd.DataFrame(grid_search.cv_results_)
top_results = results_df[['param_n_estimators', 'param_max_depth',
                          'param_min_samples_split', 'param_min_samples_leaf',
                          'mean_test_score']].head(10)

print("\n최상위 10개 결과:")
print(top_results.to_string(index=False))


# 7. 시각화
print("\n[7] 결과 시각화")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 모델 비교
ax = axes[0]
models = ['기본 모델', 'GridSearchCV', 'RandomizedSearchCV']
accuracies = [base_accuracy, grid_accuracy, random_accuracy]
colors = ['lightcoral', 'lightgreen', 'lightyellow']

bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('정확도', fontsize=11)
ax.set_title('모델 성능 비교', fontweight='bold', fontsize=12)
ax.set_ylim([min(accuracies) - 0.02, 1.0])
ax.grid(True, alpha=0.3, axis='y')

# 정확도 표시
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# max_depth에 따른 성능
ax = axes[1]
depth_performance = results_df.groupby('param_max_depth')['mean_test_score'].mean()
ax.plot(depth_performance.index, depth_performance.values, 'o-', linewidth=2, markersize=8, color='steelblue')
ax.set_xlabel('max_depth', fontsize=11)
ax.set_ylabel('평균 교차검증 점수', fontsize=11)
ax.set_title('max_depth에 따른 성능', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('problem_6_2_hyperparameter_tuning.png', dpi=100)
print("✓ 시각화 저장: problem_6_2_hyperparameter_tuning.png")
plt.close()

print("\n" + "="*60)
print("하이퍼파라미터 튜닝 완료")
print("="*60)
