"""
문제 1.1: 텍스트 전처리 파이프라인 구축

지시사항:
구두점, 대소문자, 불용어(stopwords)가 포함된 원시 텍스트 문장 리스트가 주어졌을 때, 
NLTK 라이브러리를 사용하여 토큰화, 소문자 변환, 구두점 제거, 불용어 제거를 수행하는 
Python 함수를 작성하세요.
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def download_nltk_data():
    """NLTK 데이터 다운로드 (최초 1회 실행 필요)"""
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')

def preprocess_text(text_list):
    """
    텍스트 리스트를 받아 전처리 파이프라인을 적용하는 함수.
    :param text_list: 원시 텍스트 문자열의 리스트
    :return: 전처리된 토큰들의 리스트 (문장별로 나뉨)
    """
    # 영어 불용어 세트 로드
    stop_words = set(stopwords.words('english'))
    
    processed_texts = []
    for text in text_list:
        # 1. 소문자 변환
        text = text.lower()
        
        # 2. 토큰화
        tokens = word_tokenize(text)
        
        # 3. 구두점 제거 및 불용어 제거
        processed_tokens = [
            word for word in tokens 
            if word.isalpha() and word not in stop_words
        ]
        
        processed_texts.append(processed_tokens)
        
    return processed_texts

def main():
    print("=== 문제 1.1: 텍스트 전처리 파이프라인 구축 ===")
    
    # NLTK 데이터 다운로드
    download_nltk_data()
    
    # 제공 데이터
    raw_texts = [
        "Hello everyone, this is the first document for our NLP task!",
        "We are learning about Natural Language Processing, which is very exciting.",
        "Preprocessing text is an important and fundamental step."
    ]
    
    print("\n=== 원본 텍스트 ===")
    for i, text in enumerate(raw_texts, 1):
        print(f"{i}. {text}")
    
    # 전처리 실행
    preprocessed_data = preprocess_text(raw_texts)
    
    print("\n=== 전처리 결과 ===")
    for i, tokens in enumerate(preprocessed_data, 1):
        print(f"{i}. {tokens}")
    
    # 상세 분석
    print("\n=== 전처리 과정 상세 분석 ===")
    stop_words = set(stopwords.words('english'))
    
    for i, (original, processed) in enumerate(zip(raw_texts, preprocessed_data), 1):
        print(f"\n문장 {i}: '{original}'")
        
        # 단계별 처리 과정 보기
        text_lower = original.lower()
        tokens = word_tokenize(text_lower)
        
        print(f"  1단계 (소문자): {text_lower}")
        print(f"  2단계 (토큰화): {tokens}")
        
        # 제거된 단어들 분석
        removed_punctuation = [token for token in tokens if not token.isalpha()]
        removed_stopwords = [token for token in tokens if token.isalpha() and token in stop_words]
        
        print(f"  제거된 구두점: {removed_punctuation}")
        print(f"  제거된 불용어: {removed_stopwords}")
        print(f"  최종 결과: {processed}")
    
    # 통계 정보
    print("\n=== 통계 정보 ===")
    total_original_words = sum(len(word_tokenize(text)) for text in raw_texts)
    total_processed_words = sum(len(tokens) for tokens in preprocessed_data)
    
    print(f"원본 총 단어 수: {total_original_words}")
    print(f"전처리 후 단어 수: {total_processed_words}")
    print(f"제거된 단어 수: {total_original_words - total_processed_words}")
    print(f"압축률: {(1 - total_processed_words/total_original_words)*100:.1f}%")
    
    # 전체 어휘 사전
    all_vocab = set()
    for tokens in preprocessed_data:
        all_vocab.update(tokens)
    
    print(f"고유 단어 수: {len(all_vocab)}")
    print(f"어휘 사전: {sorted(list(all_vocab))}")
    
    print("\n=== 해설 ===")
    print("1. lower(): 대소문자 통일로 'JUMPS'와 'jumps'를 같은 단어로 인식")
    print("2. word_tokenize(): 문장을 단어 단위로 분리")
    print("3. isalpha(): 구두점과 숫자 제거")
    print("4. stopwords 제거: 'the', 'a', 'is' 등 빈번하지만 의미가 적은 단어 제거")
    print("5. 이 과정으로 분석에 불필요한 노이즈를 제거하고 핵심 단어만 추출")

if __name__ == "__main__":
    main()