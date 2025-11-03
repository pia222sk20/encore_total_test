"""
문제 4.5: SVM (Support Vector Machine)

요구사항:
1. 선형 SVM 훈련
2. 비선형 커널 (RBF) 사용
3. 모델 성능 평가
4. 결정 경계 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

print("="*60)
print("문제 4.5: SVM (Support Vector Machine)")
print("="*60)

# 1. 선형 분류 데이터
print("\n[1] 선형 분류 데이터 생성")
print("-" * 60)

X_linear, y_linear = make_classification(n_samples=200, n_features=2,
                                         n_redundant=0, random_state=42)
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_linear, y_linear, test_size=0.3, random_state=42
)

print(f"선형 데이터: {X_linear.shape}")


# 2. 비선형 분류 데이터
print("\n[2] 비선형 분류 데이터 생성")
print("-" * 60)

X_circle, y_circle = make_circles(n_samples=200, factor=0.5, noise=0.1, random_state=42)
X_train_cir, X_test_cir, y_train_cir, y_test_cir = train_test_split(
    X_circle, y_circle, test_size=0.3, random_state=42
)

print(f"비선형 데이터: {X_circle.shape}")


# 3. 선형 SVM
print("\n[3] 선형 SVM 훈련")
print("-" * 60)

svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_lin, y_train_lin)

y_pred_lin = svm_linear.predict(X_test_lin)
acc_lin = accuracy_score(y_test_lin, y_pred_lin)

print(f"선형 SVM 정확도: {acc_lin:.4f}")
print(f"지원 벡터 개수: {len(svm_linear.support_vectors_)}")


# 4. RBF 커널 SVM
print("\n[4] RBF 커널 SVM 훈련")
print("-" * 60)

svm_rbf_lin = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf_lin.fit(X_train_lin, y_train_lin)

y_pred_rbf_lin = svm_rbf_lin.predict(X_test_lin)
acc_rbf_lin = accuracy_score(y_test_lin, y_pred_rbf_lin)

print(f"RBF SVM (선형 데이터) 정확도: {acc_rbf_lin:.4f}")
print(f"지원 벡터 개수: {len(svm_rbf_lin.support_vectors_)}")

# 비선형 데이터에 RBF SVM
svm_rbf_cir = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf_cir.fit(X_train_cir, y_train_cir)

y_pred_rbf_cir = svm_rbf_cir.predict(X_test_cir)
acc_rbf_cir = accuracy_score(y_test_cir, y_pred_rbf_cir)

print(f"RBF SVM (비선형 데이터) 정확도: {acc_rbf_cir:.4f}")
print(f"지원 벡터 개수: {len(svm_rbf_cir.support_vectors_)}")


# 5. 결정 경계 시각화
print("\n[5] 결정 경계 시각화")
print("-" * 60)

def plot_decision_boundary(ax, svm, X, y, title):
    """결정 경계를 그리는 헬퍼 함수"""
    # 메시 생성
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # 예측
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 결정 경계
    ax.contourf(xx, yy, Z, alpha=0.3, levels=np.linspace(0, 1, 3), cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='black', linewidths=1, levels=[0.5])
    
    # 데이터
    ax.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', s=50, alpha=0.7, label='클래스 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='s', s=50, alpha=0.7, label='클래스 1')
    
    # 지원 벡터
    if len(svm.support_vectors_) > 0:
        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                  c='yellow', marker='*', s=300, edgecolors='black', linewidth=1.5,
                  label='지원 벡터')
    
    ax.set_xlabel('특성 1', fontsize=10)
    ax.set_ylabel('특성 2', fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 선형 데이터 + 선형 SVM
plot_decision_boundary(axes[0, 0], svm_linear, X_test_lin, y_test_lin,
                       f'선형 SVM (선형 데이터)\n정확도: {acc_lin:.4f}')

# 선형 데이터 + RBF SVM
plot_decision_boundary(axes[0, 1], svm_rbf_lin, X_test_lin, y_test_lin,
                       f'RBF SVM (선형 데이터)\n정확도: {acc_rbf_lin:.4f}')

# 비선형 데이터 + 선형 SVM
svm_linear_cir = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear_cir.fit(X_train_cir, y_train_cir)
y_pred_lin_cir = svm_linear_cir.predict(X_test_cir)
acc_lin_cir = accuracy_score(y_test_cir, y_pred_lin_cir)
plot_decision_boundary(axes[1, 0], svm_linear_cir, X_test_cir, y_test_cir,
                       f'선형 SVM (비선형 데이터)\n정확도: {acc_lin_cir:.4f}')

# 비선형 데이터 + RBF SVM
plot_decision_boundary(axes[1, 1], svm_rbf_cir, X_test_cir, y_test_cir,
                       f'RBF SVM (비선형 데이터)\n정확도: {acc_rbf_cir:.4f}')

plt.suptitle('SVM 커널 비교', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('problem_4_5_svm.png', dpi=100, bbox_inches='tight')
print("✓ 시각화 저장: problem_4_5_svm.png")
plt.close()

print("\n" + "="*60)
print("SVM 분석 완료")
print("="*60)
