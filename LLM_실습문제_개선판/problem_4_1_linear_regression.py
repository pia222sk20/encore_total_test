"""
문제 4.1: 선형 회귀 (Linear Regression)

요구사항:
1. 단순 선형 회귀 모델 훈련
2. 모델 성능 평가 (MSE, R² 점수)
3. 예측 수행
4. 결과 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("="*60)
print("문제 4.1: 선형 회귀 (Linear Regression)")
print("="*60)

# 1. 데이터 생성
print("\n[1] 데이터 생성")
print("-" * 60)

np.random.seed(42)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = 2.5 * X.ravel() + np.random.randn(10) * 2  # y = 2.5x + 잡음

print(f"X 형태: {X.shape}")
print(f"y 형태: {y.shape}")
print(f"X 데이터: {X.ravel()}")
print(f"y 데이터: {y}")


# 2. 모델 훈련
print("\n[2] 모델 훈련")
print("-" * 60)

model = LinearRegression()
model.fit(X, y)

print(f"기울기 (slope): {model.coef_[0]:.4f}")
print(f"절편 (intercept): {model.intercept_:.4f}")
print(f"회귀 방정식: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}")


# 3. 예측
print("\n[3] 예측")
print("-" * 60)

y_pred = model.predict(X)
print("실제값 vs 예측값:")
for i in range(len(X)):
    print(f"  X={X[i, 0]:.1f}: 실제={y[i]:.2f}, 예측={y_pred[i]:.2f}")


# 4. 모델 평가
print("\n[4] 모델 평가")
print("-" * 60)

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"평균제곱오차 (MSE): {mse:.4f}")
print(f"루트평균제곱오차 (RMSE): {rmse:.4f}")
print(f"결정계수 (R² Score): {r2:.4f}")


# 5. 새로운 데이터 예측
print("\n[5] 새로운 데이터 예측")
print("-" * 60)

X_new = np.array([0, 5.5, 11, 15]).reshape(-1, 1)
y_new_pred = model.predict(X_new)

print("새 입력값에 대한 예측:")
for i in range(len(X_new)):
    print(f"  X={X_new[i, 0]:.1f} -> 예측값: {y_new_pred[i]:.2f}")


# 6. 결과 시각화
print("\n[6] 결과 시각화")
print("-" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

# 학습 데이터
ax.scatter(X, y, s=100, color='blue', label='실제 데이터', alpha=0.7)

# 회귀선
X_plot = np.linspace(-1, 12, 100).reshape(-1, 1)
y_plot = model.predict(X_plot)
ax.plot(X_plot, y_plot, 'r-', linewidth=2, label='회귀선')

# 예측값
ax.scatter(X, y_pred, s=50, color='red', marker='x', linewidth=2, label='예측값')

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('선형 회귀 결과 (R²={:.4f})'.format(r2), fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('problem_4_1_linear_regression.png', dpi=100)
print("✓ 시각화 저장: problem_4_1_linear_regression.png")
plt.close()

print("\n" + "="*60)
print("선형 회귀 완료")
print("="*60)
