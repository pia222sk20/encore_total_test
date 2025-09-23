"""
문제 5.3: 딥러닝을 이용한 IMDB 영화 리뷰 감정 분석
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def decode_review(encoded_review, word_index):
    """인코딩된 리뷰를 텍스트로 디코딩"""
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded = ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    return decoded

def problem_5_3():
    print("=== 문제 5.3: 딥러닝을 이용한 IMDB 영화 리뷰 감정 분석 ===")
    
    # 1. 데이터 로드
    print("IMDB 데이터셋 로딩 중...")
    max_features = 10000  # 상위 10,000개 단어만 사용
    maxlen = 500  # 리뷰 최대 길이
    
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
    
    print(f"훈련 데이터: {len(X_train)}개")
    print(f"테스트 데이터: {len(X_test)}개")
    print(f"클래스: 0(부정), 1(긍정)")
    print(f"훈련 데이터 클래스 분포: {np.bincount(y_train)}")
    
    # 2. 데이터 탐색
    word_index = imdb.get_word_index()
    print(f"\n어휘 사전 크기: {len(word_index)}")
    
    # 샘플 리뷰 확인
    print(f"\n=== 샘플 리뷰 (원본 인코딩) ===")
    print(f"첫 번째 리뷰 길이: {len(X_train[0])}")
    print(f"첫 번째 리뷰 (처음 20개 단어): {X_train[0][:20]}")
    
    # 디코딩된 리뷰
    sample_review = decode_review(X_train[0], word_index)
    print(f"\n=== 디코딩된 샘플 리뷰 ===")
    print(f"감정: {'긍정' if y_train[0] == 1 else '부정'}")
    print(f"리뷰: {sample_review[:200]}...")
    
    # 3. 리뷰 길이 분석
    review_lengths = [len(review) for review in X_train]
    print(f"\n=== 리뷰 길이 통계 ===")
    print(f"평균 길이: {np.mean(review_lengths):.1f}")
    print(f"중간값: {np.median(review_lengths):.1f}")
    print(f"최대 길이: {np.max(review_lengths)}")
    print(f"최소 길이: {np.min(review_lengths)}")
    
    # 리뷰 길이 분포 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(review_lengths, bins=50, alpha=0.7, color='skyblue')
    plt.axvline(np.mean(review_lengths), color='red', linestyle='--', label=f'평균: {np.mean(review_lengths):.1f}')
    plt.axvline(maxlen, color='green', linestyle='--', label=f'최대 길이: {maxlen}')
    plt.xlabel('리뷰 길이')
    plt.ylabel('빈도')
    plt.title('리뷰 길이 분포')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    lengths_pos = [len(X_train[i]) for i in range(len(X_train)) if y_train[i] == 1]
    lengths_neg = [len(X_train[i]) for i in range(len(X_train)) if y_train[i] == 0]
    
    plt.hist(lengths_pos, bins=30, alpha=0.7, label='긍정', color='blue')
    plt.hist(lengths_neg, bins=30, alpha=0.7, label='부정', color='red')
    plt.xlabel('리뷰 길이')
    plt.ylabel('빈도')
    plt.title('감정별 리뷰 길이 분포')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 4. 시퀀스 패딩
    print(f"\n=== 시퀀스 패딩 ===")
    X_train_padded = pad_sequences(X_train, maxlen=maxlen)
    X_test_padded = pad_sequences(X_test, maxlen=maxlen)
    
    print(f"패딩 후 훈련 데이터 형태: {X_train_padded.shape}")
    print(f"패딩 후 테스트 데이터 형태: {X_test_padded.shape}")
    
    # 5. LSTM 모델 구성
    print(f"\n=== LSTM 모델 구성 ===")
    model = Sequential([
        Embedding(max_features, 128, input_length=maxlen),
        LSTM(64, dropout=0.5, recurrent_dropout=0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # 6. 모델 학습
    print(f"\n=== 모델 학습 ===")
    history = model.fit(
        X_train_padded, y_train,
        batch_size=32,
        epochs=3,  # 실습용으로 짧게 설정
        validation_data=(X_test_padded, y_test),
        verbose=1
    )
    
    # 7. 학습 과정 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='훈련 정확도')
    plt.plot(history.history['val_accuracy'], label='검증 정확도')
    plt.xlabel('Epoch')
    plt.ylabel('정확도')
    plt.title('모델 정확도')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='훈련 손실')
    plt.plot(history.history['val_loss'], label='검증 손실')
    plt.xlabel('Epoch')
    plt.ylabel('손실')
    plt.title('모델 손실')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 8. 모델 평가
    print(f"\n=== 모델 평가 ===")
    test_loss, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    print(f"테스트 정확도: {test_accuracy:.4f}")
    print(f"테스트 손실: {test_loss:.4f}")
    
    # 9. 예측 및 상세 평가
    y_pred_prob = model.predict(X_test_padded, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    print(f"\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['부정', '긍정']))
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['부정', '긍정'], yticklabels=['부정', '긍정'])
    plt.xlabel('예측값')
    plt.ylabel('실제값')
    plt.title('Confusion Matrix - IMDB 감정 분석')
    plt.show()
    
    # 10. 예측 확률 분포 분석
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    prob_pos = y_pred_prob[y_test == 1].flatten()
    prob_neg = y_pred_prob[y_test == 0].flatten()
    
    plt.hist(prob_pos, bins=30, alpha=0.7, label='실제 긍정', color='blue')
    plt.hist(prob_neg, bins=30, alpha=0.7, label='실제 부정', color='red')
    plt.xlabel('예측 확률')
    plt.ylabel('빈도')
    plt.title('실제 라벨별 예측 확률 분포')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    correct_mask = (y_pred == y_test)
    prob_correct = y_pred_prob[correct_mask].flatten()
    prob_incorrect = y_pred_prob[~correct_mask].flatten()
    
    plt.hist(prob_correct, bins=30, alpha=0.7, label='올바른 예측', color='green')
    plt.hist(prob_incorrect, bins=30, alpha=0.7, label='잘못된 예측', color='orange')
    plt.xlabel('예측 확률')
    plt.ylabel('빈도')
    plt.title('예측 정확성별 확률 분포')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 11. 샘플 예측 결과
    print(f"\n=== 샘플 예측 결과 ===")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    for i, idx in enumerate(sample_indices):
        actual = y_test[idx]
        predicted_prob = y_pred_prob[idx][0]
        predicted = int(predicted_prob > 0.5)
        
        # 원본 리뷰 일부 디코딩
        review_text = decode_review(X_test[idx], word_index)
        
        print(f"\n--- 샘플 {i+1} ---")
        print(f"실제 감정: {'긍정' if actual == 1 else '부정'}")
        print(f"예측 감정: {'긍정' if predicted == 1 else '부정'}")
        print(f"예측 확률: {predicted_prob:.4f}")
        print(f"리뷰 (처음 100자): {review_text[:100]}...")
        print(f"정답 여부: {'✓' if actual == predicted else '✗'}")
    
    return model, history

if __name__ == "__main__":
    model, history = problem_5_3()