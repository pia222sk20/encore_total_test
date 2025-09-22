"""
문제 1.3: 시맨틱 능력의 발현 - 단어 임베딩 비교

지시사항:
sentence-transformers 라이브러리를 사용하여 사전 훈련된 언어 모델을 로드하세요. 
("king", "queen") 쌍과 ("king", "man") 쌍에 대해 각각 코사인 유사도를 계산하고 비교하세요. 
마지막으로, "king" - "man" + "woman" 벡터 연산을 수행한 결과가 "queen"의 벡터와 
얼마나 유사한지 확인하여, 단어 임베딩이 어떻게 의미적 관계를 학습하는지 확인하세요.
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("torch가 설치되지 않았습니다.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("numpy가 설치되지 않았습니다.")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("sentence-transformers가 설치되지 않았습니다.")
    print("설치 명령어: pip install sentence-transformers")

def calculate_cosine_similarity_manual(vec1, vec2):
    """수동으로 코사인 유사도 계산"""
    if not NUMPY_AVAILABLE:
        return 0.5  # 모의값 반환
        
    if TORCH_AVAILABLE and isinstance(vec1, torch.Tensor):
        vec1 = vec1.numpy()
    if TORCH_AVAILABLE and isinstance(vec2, torch.Tensor):
        vec2 = vec2.numpy()
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot_product / (norm1 * norm2)

def demonstrate_word_embeddings():
    """단어 임베딩 시연 (sentence-transformers 없이도 실행 가능)"""
    print("=== 단어 임베딩 개념 시연 ===")
    
    # 간단한 단어 벡터 예시 (실제로는 더 고차원)
    # 이는 개념 설명을 위한 가상의 저차원 벡터입니다
    word_vectors = {
        'king': np.array([0.8, 0.2, 0.9, 0.1, 0.7]),
        'queen': np.array([0.7, 0.8, 0.9, 0.2, 0.6]),
        'man': np.array([0.9, 0.1, 0.3, 0.1, 0.8]),
        'woman': np.array([0.6, 0.9, 0.3, 0.2, 0.7])
    }
    
    print("가상의 5차원 단어 벡터:")
    for word, vector in word_vectors.items():
        print(f"{word}: {vector}")
    
    # 코사인 유사도 계산
    print("\n=== 코사인 유사도 비교 ===")
    
    similarity_kq = calculate_cosine_similarity_manual(
        word_vectors['king'], word_vectors['queen']
    )
    similarity_km = calculate_cosine_similarity_manual(
        word_vectors['king'], word_vectors['man']
    )
    
    print(f"코사인 유사도 ('king', 'queen'): {similarity_kq:.4f}")
    print(f"코사인 유사도 ('king', 'man'): {similarity_km:.4f}")
    
    # 벡터 연산
    print("\n=== 벡터 연산 유추 테스트 ===")
    result_vector = (word_vectors['king'] - 
                    word_vectors['man'] + 
                    word_vectors['woman'])
    
    analogy_similarity = calculate_cosine_similarity_manual(
        result_vector, word_vectors['queen']
    )
    
    print(f"king - man + woman = {result_vector}")
    print(f"queen = {word_vectors['queen']}")
    print(f"연산 결과와 'queen'의 유사도: {analogy_similarity:.4f}")
    
    return similarity_kq, similarity_km, analogy_similarity

def real_embeddings_demo():
    """실제 sentence-transformers를 사용한 데모"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\nsentence-transformers가 설치되지 않아 실제 임베딩 데모를 건너뜁니다.")
        return None
    
    print("\n=== 실제 언어 모델 임베딩 ===")
    
    try:
        # 1. 사전 훈련된 모델 로드
        print("SentenceTransformer 모델 로딩 중...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("모델 로딩 완료.")
        
        # 2. 단어 임베딩 생성
        words = ['king', 'queen', 'man', 'woman']
        word_embeddings = model.encode(words, convert_to_tensor=True)
        
        # 단어별 임베딩 딕셔너리 생성
        embeddings_dict = {word: emb for word, emb in zip(words, word_embeddings)}
        
        print(f"임베딩 차원: {word_embeddings.shape[1]}")
        
        # 3. 코사인 유사도 비교
        print("\n" + "="*50)
        print("유사도 비교")
        print("="*50)
        
        # (king, queen) 유사도
        cosine_kq = util.cos_sim(embeddings_dict['king'], embeddings_dict['queen'])
        print(f"코사인 유사도 ('king', 'queen'): {cosine_kq.item():.4f}")
        
        # (king, man) 유사도
        cosine_km = util.cos_sim(embeddings_dict['king'], embeddings_dict['man'])
        print(f"코사인 유사도 ('king', 'man'): {cosine_km.item():.4f}")
        
        # 추가 유사도 비교
        cosine_qw = util.cos_sim(embeddings_dict['queen'], embeddings_dict['woman'])
        print(f"코사인 유사도 ('queen', 'woman'): {cosine_qw.item():.4f}")
        
        cosine_mw = util.cos_sim(embeddings_dict['man'], embeddings_dict['woman'])
        print(f"코사인 유사도 ('man', 'woman'): {cosine_mw.item():.4f}")
        
        # 4. 벡터 연산을 통한 유추(Analogy) 테스트
        print("\n" + "="*50)
        print("단어 유추 테스트: king - man + woman ≈ queen?")
        print("="*50)
        
        # 벡터 연산 수행
        result_vector = (embeddings_dict['king'] - 
                        embeddings_dict['man'] + 
                        embeddings_dict['woman'])
        
        # 결과 벡터와 'queen' 벡터 간의 유사도 계산
        analogy_similarity = util.cos_sim(result_vector, embeddings_dict['queen'])
        
        print(f"연산 결과 벡터와 'queen' 벡터의 유사도: {analogy_similarity.item():.4f}")
        
        # 다른 단어들과의 유사도도 확인
        print("\n연산 결과 벡터와 다른 단어들의 유사도:")
        for word in words:
            if word != 'queen':
                sim = util.cos_sim(result_vector, embeddings_dict[word])
                print(f"  vs '{word}': {sim.item():.4f}")
        
        # 5. 추가 유추 테스트
        print("\n=== 추가 유추 테스트 ===")
        
        # 더 많은 단어로 테스트
        extended_words = ['king', 'queen', 'man', 'woman', 'prince', 'princess', 
                         'boy', 'girl', 'father', 'mother']
        
        try:
            extended_embeddings = model.encode(extended_words, convert_to_tensor=True)
            extended_dict = {word: emb for word, emb in zip(extended_words, extended_embeddings)}
            
            # prince - man + woman ≈ princess?
            result2 = (extended_dict['prince'] - 
                      extended_dict['man'] + 
                      extended_dict['woman'])
            
            princess_sim = util.cos_sim(result2, extended_dict['princess'])
            print(f"prince - man + woman ≈ princess: {princess_sim.item():.4f}")
            
            # father - man + woman ≈ mother?
            result3 = (extended_dict['father'] - 
                      extended_dict['man'] + 
                      extended_dict['woman'])
            
            mother_sim = util.cos_sim(result3, extended_dict['mother'])
            print(f"father - man + woman ≈ mother: {mother_sim.item():.4f}")
            
        except Exception as e:
            print(f"확장 테스트 중 오류: {e}")
        
        return {
            'king_queen': cosine_kq.item(),
            'king_man': cosine_km.item(),
            'analogy': analogy_similarity.item()
        }
        
    except Exception as e:
        print(f"실제 임베딩 실행 중 오류: {e}")
        return None

def main():
    print("=== 문제 1.3: 시맨틱 능력의 발현 - 단어 임베딩 비교 ===")
    
    # 개념 설명을 위한 가상 임베딩 데모
    demo_results = demonstrate_word_embeddings()
    
    # 실제 모델을 사용한 데모
    real_results = real_embeddings_demo()
    
    # 결과 비교 및 해석
    print("\n" + "="*60)
    print("=== 결과 해석 및 의미 ===")
    print("="*60)
    
    print("\n1. 코사인 유사도가 1에 가까울수록 의미적으로 유사함")
    print("2. king-queen은 성별 관계, king-man은 지위 관계를 반영")
    print("3. 벡터 연산을 통해 의미적 관계를 수학적으로 표현 가능")
    
    if real_results:
        print(f"\n실제 모델 결과:")
        print(f"- king-queen 유사도: {real_results['king_queen']:.4f}")
        print(f"- king-man 유사도: {real_results['king_man']:.4f}")
        print(f"- 벡터 연산 결과: {real_results['analogy']:.4f}")
    
    print("\n=== 임베딩의 놀라운 특성 ===")
    print("1. 의미적 유사성: 비슷한 의미의 단어들이 벡터 공간에서 가까이 위치")
    print("2. 선형 관계: king - man + woman ≈ queen 같은 선형 연산이 의미적 관계를 보존")
    print("3. 고차원 표현: 384차원(MiniLM) 또는 768차원(BERT) 공간에서 복잡한 언어적 관계 학습")
    print("4. 전이 학습: 대규모 텍스트로 사전 훈련된 지식을 다양한 작업에 활용 가능")
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n=== 설치 안내 ===")
        print("실제 임베딩 모델을 사용하려면 다음 라이브러리를 설치하세요:")
        print("pip install sentence-transformers torch")

if __name__ == "__main__":
    main()