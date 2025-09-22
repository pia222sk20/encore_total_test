"""
문제 5.2: 손실 함수 및 역전파 구현

요구사항:
1. 간단한 신경망의 손실 함수(평균 제곱 오차) 구현
2. 역전파 알고리즘 수동 구현
3. 경사 하강법을 통한 가중치 업데이트
4. 학습 과정 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SimpleNeuralNetwork:
    """간단한 2층 신경망 클래스"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        신경망 초기화
        
        Parameters:
        - input_size: 입력층 크기
        - hidden_size: 은닉층 크기
        - output_size: 출력층 크기
        - learning_rate: 학습률
        """
        # 가중치 초기화 (He 초기화)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        self.loss_history = []
        
    def sigmoid(self, x):
        """시그모이드 활성화 함수"""
        # 수치적 안정성을 위한 클리핑
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """시그모이드 미분"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """순전파"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # 선형 출력 (회귀 문제)
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        """평균 제곱 오차 손실 함수"""
        m = y_true.shape[0]
        loss = np.sum((y_pred - y_true)**2) / (2 * m)
        return loss
    
    def backward(self, X, y_true, y_pred):
        """역전파 알고리즘"""
        m = X.shape[0]
        
        # 출력층 오차
        dz2 = y_pred - y_true  # MSE 손실의 미분
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # 은닉층 오차
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1, db1, dW2, db2):
        """경사 하강법으로 가중치 업데이트"""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X, y, epochs):
        """신경망 훈련"""
        for epoch in range(epochs):
            # 순전파
            y_pred = self.forward(X)
            
            # 손실 계산
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # 역전파
            dW1, db1, dW2, db2 = self.backward(X, y, y_pred)
            
            # 가중치 업데이트
            self.update_weights(dW1, db1, dW2, db2)
            
            # 주기적으로 손실 출력
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

def generate_data():
    """훈련 데이터 생성 (비선형 함수)"""
    np.random.seed(42)
    X = np.random.uniform(-2, 2, (200, 1))
    y = X**2 + 0.5*X + 0.1*np.random.randn(200, 1)  # 이차 함수 + 노이즈
    return X, y

def main():
    print("=== 문제 5.2: 손실 함수 및 역전파 구현 ===")
    
    # 데이터 생성
    X, y = generate_data()
    print(f"훈련 데이터 크기: X={X.shape}, y={y.shape}")
    
    # 신경망 생성
    nn = SimpleNeuralNetwork(input_size=1, hidden_size=10, output_size=1, learning_rate=0.01)
    print("신경망 구조: 1(입력) -> 10(은닉) -> 1(출력)")
    
    # 훈련 전 예측
    y_pred_before = nn.forward(X)
    loss_before = nn.compute_loss(y, y_pred_before)
    print(f"훈련 전 손실: {loss_before:.6f}")
    
    print("\n=== 신경망 훈련 시작 ===")
    nn.train(X, y, epochs=1000)
    
    # 훈련 후 예측
    y_pred_after = nn.forward(X)
    loss_after = nn.compute_loss(y, y_pred_after)
    print(f"\n훈련 후 손실: {loss_after:.6f}")
    print(f"손실 감소량: {loss_before - loss_after:.6f}")
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 1. 손실 함수 변화
    plt.subplot(2, 3, 1)
    plt.plot(nn.loss_history)
    plt.title('Loss Function During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    
    # 2. 훈련 전후 예측 비교
    plt.subplot(2, 3, 2)
    sorted_idx = np.argsort(X.flatten())
    plt.scatter(X, y, alpha=0.5, label='True Data')
    plt.plot(X[sorted_idx], y_pred_before[sorted_idx], 'r--', label='Before Training')
    plt.plot(X[sorted_idx], y_pred_after[sorted_idx], 'g-', label='After Training')
    plt.title('Predictions Before vs After Training')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # 3. 가중치 분포 (첫 번째 층)
    plt.subplot(2, 3, 3)
    plt.hist(nn.W1.flatten(), bins=20, alpha=0.7)
    plt.title('Weight Distribution (Layer 1)')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # 4. 활성화 함수 출력 분포
    plt.subplot(2, 3, 4)
    activation_output = nn.a1.flatten()
    plt.hist(activation_output, bins=20, alpha=0.7)
    plt.title('Hidden Layer Activation Distribution')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # 5. 오차 분포
    plt.subplot(2, 3, 5)
    errors = (y_pred_after - y).flatten()
    plt.hist(errors, bins=20, alpha=0.7)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error (Predicted - True)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # 6. 학습률 효과 비교
    plt.subplot(2, 3, 6)
    learning_rates = [0.001, 0.01, 0.1]
    for lr in learning_rates:
        nn_temp = SimpleNeuralNetwork(1, 10, 1, lr)
        nn_temp.train(X, y, epochs=500)
        plt.plot(nn_temp.loss_history, label=f'LR={lr}')
    plt.title('Learning Rate Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 역전파 상세 분석
    print("\n=== 역전파 상세 분석 ===")
    
    # 작은 배치로 역전파 계산 과정 보기
    X_small = X[:5]
    y_small = y[:5]
    
    # 순전파
    y_pred_small = nn.forward(X_small)
    loss_small = nn.compute_loss(y_small, y_pred_small)
    
    # 역전파
    dW1, db1, dW2, db2 = nn.backward(X_small, y_small, y_pred_small)
    
    print(f"작은 배치 크기: {X_small.shape[0]}")
    print(f"배치 손실: {loss_small:.6f}")
    print(f"dW1 크기: {dW1.shape}, 평균 기울기: {np.mean(np.abs(dW1)):.6f}")
    print(f"dW2 크기: {dW2.shape}, 평균 기울기: {np.mean(np.abs(dW2)):.6f}")
    print(f"db1 평균 기울기: {np.mean(np.abs(db1)):.6f}")
    print(f"db2 평균 기울기: {np.mean(np.abs(db2)):.6f}")
    
    # 수치적 기울기와 해석적 기울기 비교 (기울기 검증)
    print("\n=== 기울기 검증 ===")
    epsilon = 1e-7
    
    # W1의 첫 번째 원소에 대한 수치적 기울기
    original_w = nn.W1[0, 0]
    
    # +epsilon
    nn.W1[0, 0] = original_w + epsilon
    y_pred_plus = nn.forward(X_small)
    loss_plus = nn.compute_loss(y_small, y_pred_plus)
    
    # -epsilon
    nn.W1[0, 0] = original_w - epsilon
    y_pred_minus = nn.forward(X_small)
    loss_minus = nn.compute_loss(y_small, y_pred_minus)
    
    # 수치적 기울기
    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
    
    # 원래 값으로 복원
    nn.W1[0, 0] = original_w
    
    # 해석적 기울기
    y_pred_small = nn.forward(X_small)
    dW1, _, _, _ = nn.backward(X_small, y_small, y_pred_small)
    analytical_grad = dW1[0, 0]
    
    print(f"수치적 기울기: {numerical_grad:.8f}")
    print(f"해석적 기울기: {analytical_grad:.8f}")
    print(f"기울기 차이: {abs(numerical_grad - analytical_grad):.8f}")
    
    if abs(numerical_grad - analytical_grad) < 1e-5:
        print("✓ 기울기 검증 통과!")
    else:
        print("✗ 기울기 검증 실패!")
    
    print("------------------------------------------------------------")
    print("=== 주요 개념 정리 ===")
    print("1. 순전파: 입력 -> 은닉층 -> 출력층으로 계산 전파")
    print("2. 손실 함수: 예측값과 실제값의 차이를 측정 (MSE 사용)")
    print("3. 역전파: 연쇄 법칙을 사용해 기울기를 역방향으로 계산")
    print("4. 경사 하강법: 기울기의 반대 방향으로 가중치 업데이트")
    print("5. 학습률: 가중치 업데이트 크기를 조절하는 하이퍼파라미터")

if __name__ == "__main__":
    main()