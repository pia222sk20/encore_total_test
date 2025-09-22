"""
문제 1.2: 텍스트에서 벡터로 - TF-IDF

지시사항:
문제 1.1에서 전처리된 텍스트를 사용하여, scikit-learn의 TfidfVectorizer를 적용해 
문장들을 TF-IDF 행렬로 변환하세요. 벡터라이저가 학습한 피처(단어) 이름들과 
변환된 TF-IDF 행렬을 출력하여 텍스트가 어떻게 수치 데이터로 표현되는지 확인하세요.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def download_nltk_data():
    """NLTK 데이터 다운로드"""
    try:
        stopwords.words('english')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('stopwords')
        nltk.download('punkt')

def preprocess_text(text_list):
    """문제 1.1의 전처리 함수"""
    stop_words = set(stopwords.words('english'))
    
    processed_texts = []
    for text in text_list:
        text = text.lower()
        tokens = word_tokenize(text)
        processed_tokens = [
            word for word in tokens 
            if word.isalpha() and word not in stop_words
        ]
        processed_texts.append(processed_tokens)
        
    return processed_texts

def analyze_tfidf_manually(corpus):
    """TF-IDF 계산 과정을 수동으로 분석"""
    print("\n=== TF-IDF 계산 과정 수동 분석 ===")
    
    # 전체 어휘 구축
    all_words = set()
    for doc in corpus:
        all_words.update(doc.split())
    vocab = sorted(list(all_words))
    
    print(f"전체 어휘: {vocab}")
    print(f"어휘 크기: {len(vocab)}")
    
    # 각 문서별 단어 빈도 계산 (TF)
    tf_matrix = []
    for i, doc in enumerate(corpus):
        words = doc.split()
        tf_row = []
        for word in vocab:
            tf = words.count(word) / len(words)  # 정규화된 TF
            tf_row.append(tf)
        tf_matrix.append(tf_row)
        print(f"문서 {i+1} TF: {dict(zip(vocab, tf_row))}")
    
    # IDF 계산
    num_docs = len(corpus)
    idf_values = []
    for word in vocab:
        df = sum(1 for doc in corpus if word in doc.split())  # 단어가 포함된 문서 수
        idf = np.log(num_docs / df)
        idf_values.append(idf)
    
    print(f"\nIDF 값: {dict(zip(vocab, idf_values))}")
    
    # TF-IDF 계산
    tfidf_matrix = []
    for tf_row in tf_matrix:
        tfidf_row = [tf * idf for tf, idf in zip(tf_row, idf_values)]
        tfidf_matrix.append(tfidf_row)
    
    return vocab, tfidf_matrix

def main():
    print("=== 문제 1.2: 텍스트에서 벡터로 - TF-IDF ===")
    
    # NLTK 데이터 다운로드
    download_nltk_data()
    
    # 제공 데이터 (문제 1.1과 동일)
    raw_texts = [
        "Hello everyone, this is the first document for our NLP task!",
        "We are learning about Natural Language Processing, which is very exciting.",
        "Preprocessing text is an important and fundamental step."
    ]
    
    print("\n=== 1단계: 텍스트 전처리 ===")
    
    # 문제 1.1의 전처리 적용
    preprocessed_data = preprocess_text(raw_texts)
    
    # TfidfVectorizer는 토큰화된 리스트가 아닌, 공백으로 구분된 문자열을 입력으로 받습니다.
    corpus = [' '.join(tokens) for tokens in preprocessed_data]
    
    for i, (original, processed) in enumerate(zip(raw_texts, corpus), 1):
        print(f"원본 {i}: {original}")
        print(f"전처리 {i}: {processed}")
        print()
    
    print("\n=== 2단계: TF-IDF 벡터화 ===")
    
    # TfidfVectorizer 객체 생성 및 학습
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # 피처 이름 (단어 사전) 확인
    feature_names = vectorizer.get_feature_names_out()
    print("피처 이름 (어휘 사전):")
    print(feature_names)
    print(f"총 {len(feature_names)}개의 고유 단어")
    
    print("\n" + "="*50)
    
    # TF-IDF 행렬을 DataFrame으로 변환하여 가독성 높이기
    df_tfidf = pd.DataFrame(
        tfidf_matrix.toarray(), 
        columns=feature_names, 
        index=[f'문서 {i+1}' for i in range(len(corpus))]
    )
    
    print("TF-IDF 행렬:")
    print(df_tfidf)
    
    # 0이 아닌 값들만 표시
    print("\n=== 0이 아닌 TF-IDF 값들 ===")
    for i, row_name in enumerate(df_tfidf.index):
        print(f"\n{row_name}:")
        non_zero = df_tfidf.iloc[i][df_tfidf.iloc[i] > 0]
        for word, score in non_zero.items():
            print(f"  {word}: {score:.4f}")
    
    # 각 문서에서 가장 중요한 단어들
    print("\n=== 각 문서의 가장 중요한 단어들 (TF-IDF 기준) ===")
    for i, row_name in enumerate(df_tfidf.index):
        top_words = df_tfidf.iloc[i].nlargest(3)
        print(f"{row_name}: {list(top_words.index)} (점수: {top_words.values})")
    
    # TF-IDF 이해를 위한 추가 분석
    print("\n=== TF-IDF 분석 ===")
    
    # 수동 계산과 비교
    manual_vocab, manual_tfidf = analyze_tfidf_manually(corpus)
    
    # 벡터 간 코사인 유사도 계산
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    print("\n=== 문서 간 코사인 유사도 ===")
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=[f'문서 {i+1}' for i in range(len(corpus))],
        columns=[f'문서 {i+1}' for i in range(len(corpus))]
    )
    print(similarity_df)
    
    # TF-IDF 벡터의 특성
    print("\n=== TF-IDF 벡터 특성 ===")
    for i, row_name in enumerate(df_tfidf.index):
        vector = df_tfidf.iloc[i].values
        print(f"{row_name}:")
        print(f"  벡터 차원: {len(vector)}")
        print(f"  0이 아닌 원소 수: {np.count_nonzero(vector)}")
        print(f"  희소성(sparsity): {(1 - np.count_nonzero(vector)/len(vector))*100:.1f}%")
        print(f"  L2 노름: {np.linalg.norm(vector):.4f}")
    
    print("\n=== TF-IDF 설명 ===")
    print("1. TF (Term Frequency): 문서 내에서 단어의 빈도")
    print("2. IDF (Inverse Document Frequency): 단어의 희귀성을 나타냄")
    print("3. TF-IDF = TF × IDF: 문서 내 빈도는 높지만 전체 문서에서는 드문 단어가 높은 점수")
    print("4. 각 문서는 고차원 희소 벡터로 표현됨")
    print("5. 코사인 유사도로 문서 간 유사성을 측정할 수 있음")

if __name__ == "__main__":
    main()