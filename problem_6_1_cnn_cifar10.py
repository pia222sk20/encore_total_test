"""
문제 6.1: CNN을 이용한 CIFAR-10 이미지 분류
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def problem_6_1():
    print("=== 문제 6.1: CNN을 이용한 CIFAR-10 이미지 분류 ===")
    
    # 1. 데이터 로드
    print("CIFAR-10 데이터셋 로딩 중...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # 클래스 이름 정의
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"훈련 데이터: {X_train.shape}")
    print(f"테스트 데이터: {X_test.shape}")
    print(f"클래스 수: {len(class_names)}")
    print(f"이미지 크기: {X_train.shape[1]}x{X_train.shape[2]}x{X_train.shape[3]}")
    
    # 2. 데이터 탐색
    print(f"\n=== 클래스별 분포 ===")
    unique, counts = np.unique(y_train, return_counts=True)
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        print(f"{class_names[class_idx]}: {count}개")
    
    # 샘플 이미지 시각화
    plt.figure(figsize=(15, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        # 각 클래스의 첫 번째 이미지 찾기
        class_idx = np.where(y_train.flatten() == i)[0][0]
        plt.imshow(X_train[class_idx])
        plt.title(f'{class_names[i]}')
        plt.axis('off')
    plt.suptitle('CIFAR-10 클래스별 샘플 이미지')
    plt.tight_layout()
    plt.show()
    
    # 무작위 샘플 이미지
    plt.figure(figsize=(12, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        idx = np.random.randint(0, len(X_train))
        plt.imshow(X_train[idx])
        plt.title(f'{class_names[y_train[idx][0]]}')
        plt.axis('off')
    plt.suptitle('무작위 샘플 이미지')
    plt.tight_layout()
    plt.show()
    
    # 3. 데이터 전처리
    print(f"\n=== 데이터 전처리 ===")
    
    # 픽셀 값 정규화 (0-1 범위)
    X_train_normalized = X_train.astype('float32') / 255.0
    X_test_normalized = X_test.astype('float32') / 255.0
    
    print(f"정규화 전 범위: [{X_train.min()}, {X_train.max()}]")
    print(f"정규화 후 범위: [{X_train_normalized.min():.1f}, {X_train_normalized.max():.1f}]")
    
    # 라벨 원-핫 인코딩
    y_train_categorical = to_categorical(y_train, 10)
    y_test_categorical = to_categorical(y_test, 10)
    
    print(f"원본 라벨 형태: {y_train.shape}")
    print(f"원-핫 인코딩 후: {y_train_categorical.shape}")
    print(f"샘플 라벨 변환: {y_train[0]} -> {y_train_categorical[0]}")
    
    # 4. CNN 모델 구성
    print(f"\n=== CNN 모델 구성 ===")
    model = Sequential([
        # 첫 번째 컨볼루션 블록
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # 두 번째 컨볼루션 블록
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # 세 번째 컨볼루션 블록
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        
        # 완전연결층
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # 5. 콜백 설정
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
    ]
    
    # 6. 모델 학습
    print(f"\n=== 모델 학습 ===")
    history = model.fit(
        X_train_normalized, y_train_categorical,
        batch_size=32,
        epochs=5,  # 실습용으로 짧게 설정
        validation_data=(X_test_normalized, y_test_categorical),
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. 학습 과정 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='훈련 정확도')
    plt.plot(history.history['val_accuracy'], label='검증 정확도')
    plt.xlabel('Epoch')
    plt.ylabel('정확도')
    plt.title('모델 정확도')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='훈련 손실')
    plt.plot(history.history['val_loss'], label='검증 손실')
    plt.xlabel('Epoch')
    plt.ylabel('손실')
    plt.title('모델 손실')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='학습률')
        plt.xlabel('Epoch')
        plt.ylabel('학습률')
        plt.title('학습률 변화')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 8. 모델 평가
    print(f"\n=== 모델 평가 ===")
    test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test_categorical, verbose=0)
    print(f"테스트 정확도: {test_accuracy:.4f}")
    print(f"테스트 손실: {test_loss:.4f}")
    
    # 9. 예측 및 상세 평가
    y_pred_prob = model.predict(X_test_normalized, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = y_test.flatten()
    
    print(f"\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 10. 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('예측값')
    plt.ylabel('실제값')
    plt.title('Confusion Matrix - CIFAR-10 분류')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # 11. 클래스별 정확도 분석
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    accuracy_df = pd.DataFrame({
        'Class': class_names,
        'Accuracy': class_accuracy
    }).sort_values('Accuracy', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=accuracy_df, x='Accuracy', y='Class', palette='viridis')
    plt.title('클래스별 정확도')
    plt.xlabel('정확도')
    for i, v in enumerate(accuracy_df['Accuracy']):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center')
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== 클래스별 정확도 ===")
    for class_name, acc in zip(accuracy_df['Class'], accuracy_df['Accuracy']):
        print(f"{class_name}: {acc:.4f}")
    
    # 12. 잘못 분류된 이미지 분석
    wrong_indices = np.where(y_pred != y_true)[0]
    print(f"\n잘못 분류된 이미지 수: {len(wrong_indices)} / {len(y_test)}")
    
    # 가장 확신 있게 틀린 예측들
    wrong_confidences = []
    for idx in wrong_indices:
        predicted_class = y_pred[idx]
        confidence = y_pred_prob[idx][predicted_class]
        wrong_confidences.append((idx, confidence))
    
    # 확신도 순으로 정렬
    wrong_confidences.sort(key=lambda x: x[1], reverse=True)
    
    plt.figure(figsize=(15, 8))
    for i in range(min(12, len(wrong_confidences))):
        idx, confidence = wrong_confidences[i]
        plt.subplot(3, 4, i + 1)
        plt.imshow(X_test[idx])
        actual = class_names[y_true[idx]]
        predicted = class_names[y_pred[idx]]
        plt.title(f'실제: {actual}\n예측: {predicted}\n확신도: {confidence:.3f}')
        plt.axis('off')
    plt.suptitle('가장 확신 있게 틀린 예측들')
    plt.tight_layout()
    plt.show()
    
    # 13. 예측 확률 분포 분석
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    max_probs = np.max(y_pred_prob, axis=1)
    correct_mask = (y_pred == y_true)
    
    plt.hist(max_probs[correct_mask], bins=30, alpha=0.7, label='올바른 예측', color='green')
    plt.hist(max_probs[~correct_mask], bins=30, alpha=0.7, label='잘못된 예측', color='red')
    plt.xlabel('최대 예측 확률')
    plt.ylabel('빈도')
    plt.title('예측 확률 분포')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    entropy = -np.sum(y_pred_prob * np.log(y_pred_prob + 1e-10), axis=1)
    plt.hist(entropy[correct_mask], bins=30, alpha=0.7, label='올바른 예측', color='green')
    plt.hist(entropy[~correct_mask], bins=30, alpha=0.7, label='잘못된 예측', color='red')
    plt.xlabel('예측 엔트로피')
    plt.ylabel('빈도')
    plt.title('예측 불확실성 분포')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, history

if __name__ == "__main__":
    import pandas as pd  # 지역 import로 lint 에러 방지
    model, history = problem_6_1()