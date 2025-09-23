"""
문제 5.1: 소셜 미디어 데이터를 이용한 감정 분석
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# NLTK 데이터 다운로드
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def create_sample_data():
    """소셜 미디어 감정 분석용 샘플 데이터 생성"""
    positive_texts = [
        "I love this product! It's amazing and works perfectly.",
        "Great service! Very satisfied with my purchase.",
        "Excellent quality and fast delivery. Highly recommend!",
        "This is the best thing I've ever bought. So happy!",
        "Outstanding customer support. They really care about customers.",
        "Perfect! Exactly what I was looking for.",
        "Brilliant design and functionality. Love it!",
        "Fantastic experience. Will definitely buy again.",
        "Amazing value for money. Couldn't be happier!",
        "Superb quality and excellent service.",
        "Wonderful product! Exceeded my expectations.",
        "Great company with excellent products.",
        "Love the new features! Very impressive.",
        "Awesome customer service and quick response.",
        "Perfect solution to my problem. Thank you!",
        "Incredible performance and reliability.",
        "Beautiful design and great functionality.",
        "Excellent product at a great price.",
        "Very pleased with this purchase. Recommended!",
        "Outstanding quality and craftsmanship."
    ]
    
    negative_texts = [
        "This product is terrible. Don't waste your money.",
        "Worst purchase ever. Complete disappointment.",
        "Poor quality and bad customer service.",
        "Broken after one day. Very disappointed.",
        "Overpriced and doesn't work as advertised.",
        "Terrible experience. Would not recommend.",
        "Cheap quality and poor design. Avoid!",
        "Useless product. Total waste of money.",
        "Horrible customer support. Very frustrated.",
        "Defective product and no proper refund.",
        "Misleading description. Product is awful.",
        "Poor build quality and unreliable.",
        "Expensive for such poor quality.",
        "Disappointing performance. Expected better.",
        "Bad experience with delivery and product.",
        "Faulty product and unhelpful support.",
        "Not worth the money. Very unsatisfied.",
        "Terrible design and poor functionality.",
        "Worst company ever. Avoid at all costs.",
        "Completely broken and unusable."
    ]
    
    neutral_texts = [
        "The product arrived on time as expected.",
        "Standard quality product. Nothing special.",
        "It works as described. Average performance.",
        "Received the order. It's okay, nothing more.",
        "Product is fine. Meets basic requirements.",
        "Decent quality for the price. Acceptable.",
        "Normal shipping time. Product is alright.",
        "It's an average product. Does the job.",
        "Standard features and typical performance.",
        "Product is as expected. No surprises.",
        "Delivery was on schedule. Product is okay.",
        "Basic functionality works fine.",
        "Regular quality product. Standard service.",
        "It's acceptable for the price point.",
        "Average experience overall. Nothing outstanding.",
        "Product functions as intended. Standard quality.",
        "Typical performance for this type of product.",
        "Standard packaging and delivery.",
        "Product meets basic expectations.",
        "Average customer service experience."
    ]
    
    # 데이터프레임 생성
    texts = positive_texts + negative_texts + neutral_texts
    labels = ['positive'] * len(positive_texts) + ['negative'] * len(negative_texts) + ['neutral'] * len(neutral_texts)
    
    df = pd.DataFrame({
        'text': texts,
        'sentiment': labels
    })
    
    return df.sample(frac=1).reset_index(drop=True)  # 셔플

def preprocess_text(text):
    """텍스트 전처리 함수"""
    # 소문자 변환
    text = text.lower()
    
    # 특수문자 제거 (알파벳과 공백만 유지)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 토큰화
    tokens = word_tokenize(text)
    
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # 어간 추출
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)

def problem_5_1():
    print("=== 문제 5.1: 소셜 미디어 데이터를 이용한 감정 분석 ===")
    
    # NLTK 데이터 다운로드
    download_nltk_data()
    
    # 1. 샘플 데이터 생성
    df = create_sample_data()
    print(f"데이터 형태: {df.shape}")
    print(f"\n감정별 분포:")
    print(df['sentiment'].value_counts())
    
    # 2. 데이터 탐색
    print("\n=== 샘플 텍스트 ===")
    for sentiment in df['sentiment'].unique():
        sample = df[df['sentiment'] == sentiment]['text'].iloc[0]
        print(f"{sentiment.upper()}: {sample}")
    
    # 3. 텍스트 전처리
    print("\n=== 텍스트 전처리 ===")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # 전처리 전후 비교
    print("전처리 전:", df['text'].iloc[0])
    print("전처리 후:", df['processed_text'].iloc[0])
    
    # 4. 텍스트 길이 분석
    df['text_length'] = df['processed_text'].apply(len)
    df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))
    
    print(f"\n=== 텍스트 통계 ===")
    print(f"평균 문자 수: {df['text_length'].mean():.1f}")
    print(f"평균 단어 수: {df['word_count'].mean():.1f}")
    
    # 감정별 텍스트 길이 분포
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    df.boxplot(column='word_count', by='sentiment', ax=plt.gca())
    plt.title('감정별 단어 수 분포')
    plt.suptitle('')
    
    plt.subplot(1, 2, 2)
    for sentiment in df['sentiment'].unique():
        data = df[df['sentiment'] == sentiment]['word_count']
        plt.hist(data, alpha=0.7, label=sentiment, bins=10)
    plt.xlabel('단어 수')
    plt.ylabel('빈도')
    plt.title('감정별 단어 수 히스토그램')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 5. 가장 빈번한 단어 분석
    all_words = ' '.join(df['processed_text']).split()
    word_freq = Counter(all_words)
    
    print(f"\n=== 상위 10개 빈발 단어 ===")
    for word, freq in word_freq.most_common(10):
        print(f"{word}: {freq}")
    
    # 6. TF-IDF 벡터화
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['sentiment']
    
    print(f"\n벡터화 결과: {X.shape}")
    
    # 7. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
    
    # 8. 모델 학습
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # 9. 예측 및 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== 모델 성능 ===")
    print(f"정확도: {accuracy:.4f}")
    
    print(f"\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    # 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Sentiment Analysis')
    plt.show()
    
    # 10. 특성 중요도 (TF-IDF 가중치)
    feature_names = vectorizer.get_feature_names_out()
    
    # 각 클래스별 상위 특성
    print(f"\n=== 클래스별 중요 특성 (상위 5개) ===")
    for i, class_name in enumerate(model.classes_):
        coef = model.coef_[i]
        top_features = np.argsort(coef)[-5:][::-1]
        print(f"\n{class_name.upper()}:")
        for feature_idx in top_features:
            print(f"  {feature_names[feature_idx]}: {coef[feature_idx]:.4f}")
    
    # 11. 새로운 텍스트 예측 예시
    print(f"\n=== 새로운 텍스트 예측 예시 ===")
    new_texts = [
        "This product is absolutely amazing! I love it!",
        "Terrible quality. Very disappointed with this purchase.",
        "The product works fine. Nothing special."
    ]
    
    for text in new_texts:
        processed = preprocess_text(text)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        
        print(f"\n원본: {text}")
        print(f"예측: {prediction}")
        print(f"확률: {dict(zip(model.classes_, probabilities))}")

if __name__ == "__main__":
    problem_5_1()