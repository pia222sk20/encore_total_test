"""
문제 1.1: 텍스트 전처리 파이프라인

요구사항:
1. 텍스트 소문자 변환 및 토큰화
2. 구두점 제거
3. 불용어 제거
4. 통계 정보 계산
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NLTK 데이터 다운로드 (처음 실행 시만 필요)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("="*60)
print("문제 1.1: 텍스트 전처리 파이프라인 구축")
print("="*60)

# 샘플 텍스트
texts = [
    "Hello everyone, this is the first document for our NLP task!",
    "We are learning about Natural Language Processing, which is very exciting.",
    "Preprocessing text is an important and fundamental step."
]

print("\n[1] 원본 텍스트")
print("-" * 60)
for i, text in enumerate(texts, 1):
    print(f"{i}. {text}")

# 불용어 설정
stop_words = set(stopwords.words('english'))

print("\n[2] 전처리 결과")
print("-" * 60)

all_tokens_before = []
all_tokens_after = []

for i, text in enumerate(texts, 1):
    # Step 1: 소문자 변환
    lower_text = text.lower()
    
    # Step 2: 토큰화
    tokens = word_tokenize(lower_text)
    all_tokens_before.extend(tokens)
    
    # Step 3: 구두점 제거
    alphabetic_tokens = [word for word in tokens if word.isalpha()]
    
    # Step 4: 불용어 제거
    filtered_tokens = [word for word in alphabetic_tokens if word not in stop_words]
    all_tokens_after.extend(filtered_tokens)
    
    print(f"\n문장 {i}:")
    print(f"  원본: {text}")
    print(f"  소문자 및 토큰화: {tokens}")
    print(f"  구두점 제거: {alphabetic_tokens}")
    print(f"  최종 결과: {filtered_tokens}")

print("\n[3] 통계 정보")
print("-" * 60)
print(f"원본 총 토큰 수: {len(all_tokens_before)}")
print(f"전처리 후 토큰 수: {len(all_tokens_after)}")
compression_ratio = (1 - len(all_tokens_after) / len(all_tokens_before)) * 100
print(f"압축률: {compression_ratio:.1f}%")

print(f"\n고유 단어 개수: {len(set(all_tokens_after))}")
print(f"고유 단어 목록: {sorted(set(all_tokens_after))}")

print("\n" + "="*60)
print("전처리 완료")
print("="*60)
