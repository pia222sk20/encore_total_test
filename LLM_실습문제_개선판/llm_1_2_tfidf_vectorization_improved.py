"""
문제 1.2: TF-IDF 벡터화

요구사항:
1. 문서 컬렉션 준비
2. TF-IDF 행렬 생성
3. 문서별 중요 단어 추출
4. 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

print("="*60)
print("문제 1.2: TF-IDF 벡터화")
print("="*60)

# 샘플 문서 (5개)
documents = [
    "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
    "Deep learning uses neural networks with many layers to process complex patterns.",
    "Natural language processing helps computers understand and process human language.",
    "Computer vision enables machines to interpret visual information from images and videos.",
    "Artificial intelligence is transforming many industries with applications in healthcare, finance, and education."
]

print("\n[1] 문서 컬렉션")
print("-" * 60)
for i, doc in enumerate(documents, 1):
    print(f"문서 {i}: {doc[:60]}...")

# TF-IDF 벡터화
print("\n[2] TF-IDF 벡터화")
print("-" * 60)

vectorizer = TfidfVectorizer(max_features=30, lowercase=True, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

print(f"문서 수: {tfidf_matrix.shape[0]}")
print(f"특성(단어) 수: {tfidf_matrix.shape[1]}")
print(f"행렬 타입: {type(tfidf_matrix)} (희소 행렬)")

# 특성 이름(단어) 얻기
feature_names = np.array(vectorizer.get_feature_names_out())

print(f"\n추출된 상위 10개 단어:")
print(feature_names[:10])

# 문서별 상위 단어 추출
print("\n[3] 문서별 상위 5개 중요 단어")
print("-" * 60)

for doc_idx in range(tfidf_matrix.shape[0]):
    # 해당 문서의 TF-IDF 값
    tfidf_scores = tfidf_matrix.getrow(doc_idx).toarray().flatten()
    
    # 상위 5개 단어의 인덱스
    top_indices = np.argsort(tfidf_scores)[-5:][::-1]
    
    print(f"\n문서 {doc_idx + 1}:")
    for rank, idx in enumerate(top_indices, 1):
        word = feature_names[idx]
        score = tfidf_scores[idx]
        if score > 0:
            print(f"  {rank}. {word}: {score:.4f}")

# 시각화
print("\n[4] TF-IDF 점수 시각화")
print("-" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for doc_idx in range(min(5, tfidf_matrix.shape[0])):
    tfidf_scores = tfidf_matrix.getrow(doc_idx).toarray().flatten()
    top_indices = np.argsort(tfidf_scores)[-10:][::-1]
    
    top_words = feature_names[top_indices]
    top_scores = tfidf_scores[top_indices]
    
    ax = axes[doc_idx]
    ax.barh(range(len(top_words)), top_scores, color='steelblue')
    ax.set_yticks(range(len(top_words)))
    ax.set_yticklabels(top_words)
    ax.set_xlabel('TF-IDF 점수')
    ax.set_title(f'문서 {doc_idx + 1} 상위 단어')
    ax.grid(True, alpha=0.3, axis='x')

# 마지막 서브플롯은 숨기기
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('llm_1_2_tfidf_vectorization.png', dpi=100)
print("✓ 시각화 저장: llm_1_2_tfidf_vectorization.png")
plt.close()

# 문서 간 유사도 계산
print("\n[5] 문서 간 유사도 (코사인 유사도)")
print("-" * 60)

from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tfidf_matrix)

print("유사도 행렬:")
print(similarity_matrix.round(3))

print("\n가장 유사한 문서 쌍:")
np.fill_diagonal(similarity_matrix, -1)  # 자신과의 유사도 제외
for i in range(len(documents)):
    most_similar_idx = np.argmax(similarity_matrix[i])
    max_similarity = similarity_matrix[i, most_similar_idx]
    print(f"  문서 {i+1} ↔ 문서 {most_similar_idx+1}: {max_similarity:.4f}")

print("\n" + "="*60)
print("TF-IDF 벡터화 완료")
print("="*60)
