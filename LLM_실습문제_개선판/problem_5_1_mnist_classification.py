"""
문제 5.1: 신경망을 이용한 MNIST 분류

요구사항:
1. MNIST 데이터셋 로드 및 전처리
2. 신경망 모델 구성
3. 모델 훈련
4. 모델 평가 및 예측
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.datasets import mnist

print("="*60)
print("문제 5.1: 신경망을 이용한 MNIST 분류")
print("="*60)

# 1. MNIST 데이터 로드
print("\n[1] MNIST 데이터 로드")
print("-" * 60)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"학습 데이터 형태: {X_train.shape}")
print(f"학습 라벨 형태: {y_train.shape}")
print(f"테스트 데이터 형태: {X_test.shape}")
print(f"테스트 라벨 형태: {y_test.shape}")
print(f"데이터 범위: {X_train.min()} ~ {X_train.max()}")


# 2. 데이터 전처리
print("\n[2] 데이터 전처리")
print("-" * 60)

# 정규화 (0~1 범위)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 평탄화 (28x28 -> 784)
X_train_flat = X_train.reshape(-1, 28*28)
X_test_flat = X_test.reshape(-1, 28*28)

print(f"정규화 후 범위: {X_train.min()} ~ {X_train.max()}")
print(f"평탄화 후 형태: {X_train_flat.shape}")

# 원핫 인코딩
y_train_encoded = keras.utils.to_categorical(y_train, 10)
y_test_encoded = keras.utils.to_categorical(y_test, 10)

print(f"원핫 인코딩 후 형태: {y_train_encoded.shape}")
print(f"첫 번째 샘플의 레이블: {y_train[0]} -> {y_train_encoded[0]}")


# 3. 모델 구성
print("\n[3] 신경망 모델 구성")
print("-" * 60)

model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

print("모델 구조:")
model.summary()


# 4. 모델 컴파일
print("\n[4] 모델 컴파일")
print("-" * 60)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("✓ 모델 컴파일 완료")


# 5. 모델 훈련
print("\n[5] 모델 훈련")
print("-" * 60)

history = model.fit(
    X_train_flat, y_train_encoded,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)


# 6. 모델 평가
print("\n[6] 모델 평가")
print("-" * 60)

test_loss, test_accuracy = model.evaluate(X_test_flat, y_test_encoded, verbose=0)
print(f"테스트 손실 (Loss): {test_loss:.4f}")
print(f"테스트 정확도 (Accuracy): {test_accuracy:.4f}")


# 7. 예측
print("\n[7] 예측")
print("-" * 60)

# 처음 10개 테스트 샘플 예측
predictions = model.predict(X_test_flat[:10], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

print("처음 10개 예측 결과:")
for i in range(10):
    print(f"  실제: {y_test[i]}, 예측: {predicted_labels[i]}, 신뢰도: {predictions[i, predicted_labels[i]]:.4f}")


# 8. 결과 시각화
print("\n[8] 결과 시각화")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 훈련/검증 손실
ax = axes[0, 0]
ax.plot(history.history['loss'], label='훈련 손실')
ax.plot(history.history['val_loss'], label='검증 손실')
ax.set_xlabel('Epoch')
ax.set_ylabel('손실 (Loss)')
ax.set_title('훈련/검증 손실')
ax.legend()
ax.grid(True, alpha=0.3)

# 훈련/검증 정확도
ax = axes[0, 1]
ax.plot(history.history['accuracy'], label='훈련 정확도')
ax.plot(history.history['val_accuracy'], label='검증 정확도')
ax.set_xlabel('Epoch')
ax.set_ylabel('정확도 (Accuracy)')
ax.set_title('훈련/검증 정확도')
ax.legend()
ax.grid(True, alpha=0.3)

# 샘플 이미지와 예측
ax = axes[1, 0]
sample_idx = 0
image = X_test[sample_idx]
ax.imshow(image, cmap='gray')
ax.set_title(f'실제: {y_test[sample_idx]}, 예측: {predicted_labels[sample_idx]}')
ax.axis('off')

# 예측 확률 분포
ax = axes[1, 1]
prob_dist = predictions[0]
ax.bar(range(10), prob_dist, color='steelblue')
ax.set_xlabel('숫자')
ax.set_ylabel('확률')
ax.set_title(f'첫 번째 샘플의 예측 확률 분포')
ax.set_xticks(range(10))

plt.tight_layout()
plt.savefig('problem_5_1_mnist_classification.png', dpi=100)
print("✓ 시각화 저장: problem_5_1_mnist_classification.png")
plt.close()

print("\n" + "="*60)
print("MNIST 분류 완료")
print("="*60)
