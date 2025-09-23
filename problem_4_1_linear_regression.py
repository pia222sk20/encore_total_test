"""
문제 4.1: 선형 회귀 모델 구현 및 성능 평가
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def problem_4_1():
    print("=== 문제 4.1: 선형 회귀 모델 구현 및 성능 평가 ===")
    
    # 1. 데이터 로드 및 분리
    print("California Housing 데이터셋 로딩...")
    housing = fetch_california_housing()
    
    print(f"데이터 형태: {housing.data.shape}")
    print(f"특성명: {housing.feature_names}")
    print(f"타겟 설명: {housing.DESCR[:200]}...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=0.2, random_state=42
    )
    
    print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
    
    # 2. 모델 생성 및 학습
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("선형 회귀 모델 학습 완료.")
    
    # 회귀 계수 출력
    print(f"\n회귀 계수:")
    for feature, coef in zip(housing.feature_names, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"절편(intercept): {model.intercept_:.4f}")
    
    # 3. 테스트 데이터 예측
    y_pred = model.predict(X_test)
    
    # 4. 모델 성능 평가
    # MSE (Mean Squared Error): 오차 제곱의 평균. 작을수록 좋습니다.
    # 예측값과 실제값의 차이를 나타내며, 단위가 원래 값의 제곱이 됩니다.
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nMean Squared Error (MSE): {mse:.4f}")
    
    # RMSE (Root Mean Squared Error): MSE의 제곱근, 원래 단위로 복원
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # R^2 (결정 계수): 모델이 데이터의 분산을 얼마나 잘 설명하는지를 나타내는 지표.
    # 1에 가까울수록 모델이 데이터를 잘 설명함을 의미합니다.
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared (R²): {r2:.4f}")
    
    # 5. 예측 결과 시각화
    plt.figure(figsize=(10, 8))
    
    # 실제값 vs 예측값 산점도
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs. Predicted Prices")
    plt.grid(True)
    
    # 잔차(residual) 플롯
    residuals = y_test - y_pred
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Prices")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True)
    
    # 잔차 히스토그램
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")
    plt.grid(True)
    
    # 실제값과 예측값의 분포 비교
    plt.subplot(2, 2, 4)
    plt.hist(y_test, bins=30, alpha=0.5, label='Actual', density=True)
    plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted', density=True)
    plt.xlabel("Price")
    plt.ylabel("Density")
    plt.title("Distribution Comparison")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== 모델 해석 ===")
    print(f"- MSE: {mse:.4f} (낮을수록 좋음)")
    print(f"- RMSE: {rmse:.4f} (실제 단위로 평균 오차)")
    print(f"- R²: {r2:.4f} (1에 가까울수록 좋음)")
    print(f"- 모델이 전체 분산의 {r2*100:.1f}%를 설명함")

if __name__ == "__main__":
    problem_4_1()