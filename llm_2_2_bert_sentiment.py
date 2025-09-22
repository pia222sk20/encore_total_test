"""
문제 2.2: BERT를 활용한 전이학습의 위력

지시사항:
문제 2.1과 동일한 감성 분류 작업을, 직접 모델을 구축하는 대신 Hugging Face transformers 
라이브러리의 사전 훈련된 BERT 모델과 pipeline API를 사용하여 수행하세요. 
코드의 간결성과 (개념적인) 성능을 LSTM 모델과 비교해보세요.
"""

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    print("Transformers 라이브러리가 사용 가능합니다.")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers 라이브러리가 설치되지 않았습니다.")

class MockSentimentPipeline:
    """Transformers가 없을 때 사용할 모의 감성 분석 파이프라인"""
    
    def __init__(self):
        self.model_name = "Mock BERT Model"
        # 간단한 키워드 기반 감성 분석
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'awesome', 'brilliant', 'outstanding', 'superb', 'marvelous',
            'enjoyed', 'love', 'loved', 'like', 'beautiful', 'perfect',
            'thrilled', 'delicious', 'pleasant', 'incredible', 'stunning'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
            'worst', 'boring', 'disappointing', 'waste', 'stupid', 'ridiculous',
            'annoying', 'frustrating', 'pathetic', 'useless', 'down'
        }
    
    def __call__(self, texts):
        """감성 분석 실행"""
        results = []
        
        for text in texts:
            text_lower = text.lower()
            words = text_lower.split()
            
            positive_score = sum(1 for word in words if word in self.positive_words)
            negative_score = sum(1 for word in words if word in self.negative_words)
            
            if positive_score > negative_score:
                label = 'POSITIVE'
                score = 0.6 + (positive_score - negative_score) * 0.1
                score = min(score, 0.99)
            elif negative_score > positive_score:
                label = 'NEGATIVE'
                score = 0.6 + (negative_score - positive_score) * 0.1
                score = min(score, 0.99)
            else:
                # 중립적이거나 판단 어려운 경우
                label = 'POSITIVE' if len(words) > 5 else 'NEGATIVE'
                score = 0.51
            
            results.append({
                'label': label,
                'score': score
            })
        
        return results

def create_sentiment_pipeline():
    """감성 분석 파이프라인 생성"""
    
    if TRANSFORMERS_AVAILABLE:
        try:
            print("Hugging Face BERT 모델 로딩 중...")
            # 감성 분석 파이프라인 로드
            # 기본 모델: distilbert-base-uncased-finetuned-sst-2-english
            sentiment_pipeline = pipeline("sentiment-analysis")
            print("모델 로딩 완료!")
            return sentiment_pipeline, "Real BERT"
        except Exception as e:
            print(f"실제 모델 로딩 실패: {e}")
            print("모의 파이프라인을 사용합니다.")
            return MockSentimentPipeline(), "Mock BERT"
    else:
        print("Transformers가 설치되지 않아 모의 파이프라인을 사용합니다.")
        return MockSentimentPipeline(), "Mock BERT"

def analyze_with_bert():
    """BERT를 사용한 감성 분석"""
    
    # 테스트 문장들
    test_texts = [
        "I am absolutely thrilled with the results!",  # 매우 긍정적
        "This is the worst experience I have ever had.",  # 매우 부정적
        "The movie was okay, but I probably wouldn't watch it again.",  # 중립/미묘함
        "Despite the long wait, the food was incredibly delicious.",  # 복합적 감정
        "The weather today is sunny and pleasant.",  # 긍정적
        "I'm feeling a bit down today.",  # 부정적
        "This document contains the quarterly financial report.",  # 중립/사실적
        # 추가: 문제 2.1의 데이터와 비교용
        "This movie was fantastic and amazing",
        "The acting was terrible and the story was boring",
        "I really enjoyed the plot and the characters",
        "A complete waste of time and money"
    ]
    
    print("=== BERT 감성 분석 결과 ===")
    
    # 파이프라인 생성
    sentiment_pipeline, model_type = create_sentiment_pipeline()
    
    print(f"사용 모델: {model_type}")
    print("-" * 60)
    
    # 감성 분석 수행
    results = sentiment_pipeline(test_texts)
    
    # 결과 출력 및 분석
    positive_count = 0
    negative_count = 0
    high_confidence_count = 0
    
    for text, result in zip(test_texts, results):
        label = result['label']
        score = result['score']
        
        # 통계 집계
        if label == 'POSITIVE':
            positive_count += 1
        else:
            negative_count += 1
        
        if score > 0.8:
            high_confidence_count += 1
        
        print(f"Text: {text}")
        print(f"  -> Prediction: {label}")
        print(f"  -> Confidence: {score:.4f}")
        print(f"  -> Interpretation: {'High' if score > 0.8 else 'Medium' if score > 0.6 else 'Low'} confidence")
        print()
    
    # 통계 요약
    print("=== 분석 통계 ===")
    print(f"총 분석 문장 수: {len(test_texts)}")
    print(f"긍정 예측: {positive_count}개")
    print(f"부정 예측: {negative_count}개")
    print(f"고신뢰도 예측 (>0.8): {high_confidence_count}개")
    print(f"평균 신뢰도: {sum(r['score'] for r in results)/len(results):.4f}")
    
    return results

def compare_approaches():
    """BERT와 LSTM 접근 방식 비교"""
    
    print("\n=== BERT vs LSTM 접근 방식 비교 ===")
    
    comparison_table = {
        "측면": ["코드 복잡성", "훈련 시간", "성능", "메모리 사용량", "전이학습", "설명가능성"],
        "LSTM (직접 구현)": [
            "높음 (모델 구조, 훈련 루프 등)",
            "중간 (소규모 데이터)",
            "중간 (데이터에 따라)",
            "낮음",
            "제한적 (임베딩만)",
            "상대적으로 높음"
        ],
        "BERT (Pipeline)": [
            "매우 낮음 (몇 줄)",
            "없음 (사전 훈련됨)",
            "높음 (대규모 사전 훈련)",
            "높음",
            "완전함 (전체 모델)",
            "낮음 (블랙박스)"
        ]
    }
    
    print(f"{'측면':<15} {'LSTM (직접 구현)':<25} {'BERT (Pipeline)':<25}")
    print("-" * 65)
    
    for i, aspect in enumerate(comparison_table["측면"]):
        lstm_val = comparison_table["LSTM (직접 구현)"][i]
        bert_val = comparison_table["BERT (Pipeline)"][i]
        print(f"{aspect:<15} {lstm_val:<25} {bert_val:<25}")
    
    print("\n=== 상세 비교 ===")
    
    print("\n1. 코드 복잡성:")
    print("   LSTM: 데이터 전처리, 모델 정의, 훈련 루프, 최적화 등 모든 과정을 직접 구현")
    print("   BERT: pipeline('sentiment-analysis') 한 줄로 완성")
    
    print("\n2. 성능:")
    print("   LSTM: 작은 데이터셋에서는 제한적, 도메인 특화 학습 가능")
    print("   BERT: 대규모 텍스트로 사전 훈련되어 일반적으로 우수한 성능")
    
    print("\n3. 실용성:")
    print("   LSTM: 연구용, 교육용, 특수한 요구사항이 있는 경우")
    print("   BERT: 실제 서비스, 빠른 프로토타이핑, 일반적인 NLP 작업")
    
    print("\n4. 전이학습:")
    print("   LSTM: 단어 임베딩 정도만 전이 가능")
    print("   BERT: 언어의 전반적 이해가 전이됨")

def demonstrate_bert_variants():
    """다양한 BERT 모델 변형 소개"""
    
    print("\n=== BERT 모델 변형들 ===")
    
    models_info = {
        "BERT-base": "12층, 768차원, 110M 파라미터 - 원조 BERT",
        "DistilBERT": "6층, 768차원, 66M 파라미터 - BERT 지식 증류, 60% 빠름", 
        "RoBERTa": "BERT 개선버전 - 더 많은 데이터, 더 긴 훈련",
        "ALBERT": "파라미터 공유로 경량화, 더 깊은 네트워크",
        "DeBERTa": "분리된 어텐션으로 성능 개선"
    }
    
    for model, description in models_info.items():
        print(f"{model}: {description}")
    
    print("\n=== 감성 분석 특화 모델들 ===")
    specialized_models = {
        "cardiffnlp/twitter-roberta-base-sentiment": "트위터 데이터로 파인튜닝",
        "nlptown/bert-base-multilingual-uncased-sentiment": "다국어 지원",
        "ProsusAI/finbert": "금융 도메인 특화"
    }
    
    for model, description in specialized_models.items():
        print(f"{model}: {description}")

def practical_tips():
    """실용적인 팁들"""
    
    print("\n=== 실용적인 사용 팁 ===")
    
    tips = [
        "1. 빠른 프로토타이핑: pipeline API 사용",
        "2. 높은 성능 필요: 특화된 모델 선택 또는 파인튜닝",
        "3. 메모리 제약: DistilBERT 같은 경량 모델 사용",
        "4. 특정 도메인: 도메인 특화 모델이나 자체 파인튜닝",
        "5. 다국어: multilingual 모델 사용",
        "6. 실시간 서비스: 모델 최적화 및 캐싱 전략 필요"
    ]
    
    for tip in tips:
        print(tip)
    
    print("\n=== 실제 서비스 고려사항 ===")
    considerations = [
        "• 모델 크기와 추론 속도의 트레이드오프",
        "• GPU 메모리 요구사항",
        "• 배치 처리로 효율성 향상",
        "• 모델 버전 관리 및 업데이트 전략",
        "• A/B 테스트를 통한 성능 검증"
    ]
    
    for consideration in considerations:
        print(consideration)

def main():
    print("=== 문제 2.2: BERT를 활용한 전이학습의 위력 ===")
    
    # 1. BERT를 사용한 감성 분석
    results = analyze_with_bert()
    
    # 2. 접근 방식 비교
    compare_approaches()
    
    # 3. BERT 변형들 소개
    demonstrate_bert_variants()
    
    # 4. 실용적인 팁
    practical_tips()
    
    # 설치 안내
    if not TRANSFORMERS_AVAILABLE:
        print("\n=== 설치 안내 ===")
        print("실제 BERT 모델을 사용하려면:")
        print("pip install transformers torch")
        print("또는 TensorFlow 버전:")
        print("pip install transformers tensorflow")

if __name__ == "__main__":
    main()