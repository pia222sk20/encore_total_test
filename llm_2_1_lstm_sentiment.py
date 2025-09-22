"""
문제 2.1: LSTM을 이용한 순차 데이터 모델링 및 감성 분석

지시사항:
PyTorch를 사용하여 간단한 LSTM 기반의 신경망을 구축하고, 주어진 영화 리뷰 데이터셋으로 
감성 분류(긍정/부정) 모델을 훈련시키세요. 모델은 단어 인덱스의 시퀀스를 입력받아 
이진 분류 결과를 출력해야 합니다.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# 대체 구현을 위한 가중치
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False
    print("PyTorch가 설치되지 않았습니다. 개념적 구현으로 진행합니다.")

class SimpleLSTMWithoutTorch:
    """PyTorch 없이 LSTM 개념을 시연하는 클래스"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        print(f"LSTM 모델 생성 - 어휘크기: {vocab_size}, 임베딩차원: {embedding_dim}, 은닉차원: {hidden_dim}")
    
    def forward_simulation(self, text_sequence):
        """LSTM 순전파 시뮬레이션"""
        print(f"입력 시퀀스: {text_sequence}")
        print("LSTM 처리 과정:")
        
        # 가상의 은닉 상태 변화
        hidden_states = []
        for i, word_idx in enumerate(text_sequence):
            # 실제로는 복잡한 게이트 연산이 일어남
            simulated_hidden = np.random.rand(self.hidden_dim) * 0.1
            hidden_states.append(simulated_hidden)
            print(f"  시점 {i+1}: 단어 {word_idx} -> 은닉상태 크기 {len(simulated_hidden)}")
        
        # 최종 은닉 상태를 기반으로 감성 예측
        final_output = np.mean(hidden_states[-1])  # 단순화된 예측
        sentiment = "긍정" if final_output > 0.05 else "부정"
        
        print(f"최종 예측: {sentiment} (점수: {final_output:.4f})")
        return sentiment, final_output

class SentimentLSTM(nn.Module):
    """PyTorch LSTM 감성 분석 모델"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # 마지막 타임스텝의 출력만 사용
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sigmoid(out)
        return sig_out

def prepare_data():
    """데이터 준비 및 전처리"""
    
    # 제공된 훈련 데이터
    train_data = {
        "This movie was fantastic and amazing": 1,
        "The acting was terrible and the story was boring": 0,
        "I really enjoyed the plot and the characters": 1,
        "A complete waste of time and money": 0,
        "The visuals were stunning, a true masterpiece": 1,
        "I would not recommend this film to anyone": 0
    }
    
    print("=== 훈련 데이터 ===")
    for text, label in train_data.items():
        sentiment = "긍정" if label == 1 else "부정"
        print(f"{sentiment}: {text}")
    
    texts = list(train_data.keys())
    labels = list(train_data.values())
    
    # 단어 사전 구축
    words = ' '.join(texts).lower().split()
    word_counts = Counter(words)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    word_to_int = {word: i+1 for i, word in enumerate(vocab)}  # 0은 패딩용
    
    print(f"\n=== 어휘 사전 ===")
    print(f"총 고유 단어 수: {len(vocab)}")
    print(f"단어 빈도: {dict(word_counts.most_common(10))}")
    print(f"단어->인덱스 매핑 (일부): {dict(list(word_to_int.items())[:10])}")
    
    # 텍스트를 정수 시퀀스로 변환
    text_ints = []
    for text in texts:
        text_ints.append([word_to_int[word] for word in text.lower().split()])
    
    print(f"\n=== 정수 시퀀스 변환 ===")
    for i, (text, sequence) in enumerate(zip(texts, text_ints)):
        print(f"'{text}' -> {sequence}")
    
    return texts, labels, text_ints, word_to_int, vocab

def train_lstm_model(text_ints, labels, word_to_int):
    """LSTM 모델 훈련"""
    
    if not torch_available:
        print("\nPyTorch가 없어 개념적 시뮬레이션을 실행합니다.")
        
        # 개념적 구현
        vocab_size = len(word_to_int) + 1
        model_sim = SimpleLSTMWithoutTorch(vocab_size, 50, 64)
        
        print("\n=== 모델 훈련 시뮬레이션 ===")
        for epoch in range(5):
            total_loss = 0
            for sequence, label in zip(text_ints, labels):
                # 가상의 손실 계산
                predicted_sentiment, score = model_sim.forward_simulation(sequence)
                actual_sentiment = "긍정" if label == 1 else "부정"
                
                # 간단한 손실 계산
                loss = abs(score - label) 
                total_loss += loss
            
            avg_loss = total_loss / len(text_ints)
            print(f"Epoch {epoch+1}: 평균 손실 = {avg_loss:.4f}")
        
        return model_sim
    
    # 실제 PyTorch 구현
    print("\n=== 실제 PyTorch 모델 훈련 ===")
    
    # 패딩 처리 (모든 시퀀스 길이를 통일)
    seq_length = max(len(x) for x in text_ints)
    features = torch.zeros((len(text_ints), seq_length), dtype=torch.long)
    
    for i, row in enumerate(text_ints):
        features[i, -len(row):] = torch.tensor(row)[:seq_length]
    
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    
    print(f"패딩된 특성 행렬 크기: {features.shape}")
    print(f"레이블 텐서 크기: {labels_tensor.shape}")
    
    # 모델 초기화
    vocab_size = len(word_to_int) + 1
    embedding_dim = 50
    hidden_dim = 64
    output_dim = 1
    n_layers = 2
    
    model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"모델 구조:")
    print(f"  어휘 크기: {vocab_size}")
    print(f"  임베딩 차원: {embedding_dim}")
    print(f"  LSTM 은닉 차원: {hidden_dim}")
    print(f"  LSTM 레이어 수: {n_layers}")
    
    # 훈련 루프
    train_losses = []
    epochs = 30
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(features)
        loss = criterion(output, labels_tensor)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if (epoch+1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    # 훈련 손실 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_losses, 'b-', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model, features, seq_length

def test_model(model, word_to_int, seq_length):
    """모델 테스트"""
    
    test_texts = [
        "The movie was good and enjoyable",
        "This film is absolutely terrible",
        "I loved every moment of it",
        "Boring and disappointing experience"
    ]
    
    print("\n=== 모델 테스트 ===")
    
    if not torch_available:
        print("개념적 테스트:")
        for text in test_texts:
            words = text.lower().split()
            # 사전에 있는 단어만 사용
            sequence = [word_to_int.get(word, 0) for word in words]
            sentiment, score = model.forward_simulation(sequence)
            print(f"'{text}' -> 예측: {sentiment}")
        return
    
    # 실제 PyTorch 테스트
    model.eval()
    with torch.no_grad():
        for text in test_texts:
            # 텍스트를 정수 시퀀스로 변환
            test_int = [word_to_int.get(word, 0) for word in text.lower().split()]
            test_feature = torch.zeros((1, seq_length), dtype=torch.long)
            test_feature[0, -len(test_int):] = torch.tensor(test_int)
            
            # 예측
            prediction = model(test_feature)
            predicted_sentiment = 'Positive' if prediction.item() > 0.5 else 'Negative'
            
            print(f"Text: '{text}'")
            print(f"  Prediction Score: {prediction.item():.4f}")
            print(f"  Predicted Sentiment: {predicted_sentiment}")
            print()

def analyze_lstm_architecture():
    """LSTM 아키텍처 분석"""
    
    print("\n=== LSTM 아키텍처 분석 ===")
    print("LSTM의 핵심 구성 요소:")
    print("1. 망각 게이트 (Forget Gate): 이전 정보 중 버릴 것을 결정")
    print("2. 입력 게이트 (Input Gate): 새로운 정보 중 저장할 것을 결정") 
    print("3. 출력 게이트 (Output Gate): 현재 상태에서 출력할 것을 결정")
    print("4. 셀 상태 (Cell State): 장기 기억을 담당")
    print("5. 은닉 상태 (Hidden State): 단기 기억을 담당")
    
    print("\n감성 분석에서 LSTM의 장점:")
    print("- 순차적 정보 처리: 단어의 순서를 고려한 문맥 이해")
    print("- 장기 의존성: 긴 문장에서도 앞부분의 중요 정보를 기억")
    print("- 가변 길이 입력: 다양한 길이의 문장을 처리 가능")

def main():
    print("=== 문제 2.1: LSTM을 이용한 순차 데이터 모델링 및 감성 분석 ===")
    
    # 1. 데이터 준비
    texts, labels, text_ints, word_to_int, vocab = prepare_data()
    
    # 2. 모델 훈련
    if torch_available:
        model, features, seq_length = train_lstm_model(text_ints, labels, word_to_int)
    else:
        model = train_lstm_model(text_ints, labels, word_to_int)
        seq_length = max(len(x) for x in text_ints)
    
    # 3. 모델 테스트
    test_model(model, word_to_int, seq_length)
    
    # 4. 아키텍처 분석
    analyze_lstm_architecture()
    
    if not torch_available:
        print("\n=== 실제 구현을 위한 설치 안내 ===")
        print("실제 LSTM 모델을 훈련하려면:")
        print("pip install torch matplotlib")

if __name__ == "__main__":
    main()