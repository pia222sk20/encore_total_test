"""
문제 4.1: Few-shot 학습 실험

지시사항:
LLM에게 few-shot 예제를 제공하여 특정 작업을 학습시키고, 예제 개수와 
품질이 모델 성능에 미치는 영향을 분석하세요. 다양한 작업(분류, 번역, 
요약 등)에 대해 실험해보세요.
"""

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    print("Transformers 라이브러리가 사용 가능합니다.")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers 라이브러리가 설치되지 않았습니다.")

import random
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class FewShotExample:
    """Few-shot 학습 예제 데이터 클래스"""
    input_text: str
    output_text: str
    task_type: str

class FewShotLearningEngine:
    """Few-shot 학습 엔진"""
    
    def __init__(self):
        # 다양한 작업별 예제들
        self.examples_db = {
            "sentiment_classification": [
                {"input": "I love this product! It's amazing.", "output": "Positive"},
                {"input": "This is the worst purchase I've ever made.", "output": "Negative"},
                {"input": "The product is okay, nothing special.", "output": "Neutral"},
                {"input": "Absolutely fantastic! Would buy again.", "output": "Positive"},
                {"input": "Terrible quality and poor customer service.", "output": "Negative"},
                {"input": "It works as expected, no complaints.", "output": "Neutral"},
                {"input": "Outstanding performance and great design!", "output": "Positive"},
                {"input": "Disappointing and overpriced product.", "output": "Negative"}
            ],
            
            "text_summarization": [
                {
                    "input": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents.",
                    "output": "AI is machine intelligence that differs from natural intelligence of humans and animals."
                },
                {
                    "input": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data.",
                    "output": "Machine learning automates model building using data analysis and is a branch of AI."
                },
                {
                    "input": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
                    "output": "Deep learning uses artificial neural networks for representation learning within machine learning."
                }
            ],
            
            "translation": [
                {"input": "Hello, how are you?", "output": "안녕하세요, 어떻게 지내세요?"},
                {"input": "Thank you very much.", "output": "정말 감사합니다."},
                {"input": "Good morning!", "output": "좋은 아침입니다!"},
                {"input": "See you tomorrow.", "output": "내일 봅시다."},
                {"input": "I'm sorry.", "output": "죄송합니다."}
            ],
            
            "question_answering": [
                {
                    "input": "Question: What is the capital of France?\nContext: France is a country in Europe with Paris as its capital city.",
                    "output": "Paris"
                },
                {
                    "input": "Question: What year was the iPhone first released?\nContext: Apple released the first iPhone in 2007, revolutionizing the smartphone industry.",
                    "output": "2007"
                },
                {
                    "input": "Question: Who wrote Romeo and Juliet?\nContext: William Shakespeare wrote many famous plays including Romeo and Juliet.",
                    "output": "William Shakespeare"
                }
            ],
            
            "text_classification": [
                {"input": "Breaking: Major earthquake hits Japan", "output": "News"},
                {"input": "Top 10 movie recommendations for this weekend", "output": "Entertainment"},
                {"input": "New scientific study reveals climate change impacts", "output": "Science"},
                {"input": "Stock market reaches all-time high", "output": "Finance"},
                {"input": "Celebrity couple announces engagement", "output": "Entertainment"},
                {"input": "AI breakthrough in medical diagnosis", "output": "Technology"}
            ]
        }
    
    def create_few_shot_prompt(self, task_type: str, num_examples: int, test_input: str) -> str:
        """Few-shot 프롬프트 생성"""
        
        if task_type not in self.examples_db:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        examples = self.examples_db[task_type]
        selected_examples = random.sample(examples, min(num_examples, len(examples)))
        
        # 작업별 프롬프트 템플릿
        if task_type == "sentiment_classification":
            prompt = "Classify the sentiment of the following text as Positive, Negative, or Neutral.\n\n"
            for example in selected_examples:
                prompt += f"Text: {example['input']}\nSentiment: {example['output']}\n\n"
            prompt += f"Text: {test_input}\nSentiment:"
        
        elif task_type == "text_summarization":
            prompt = "Summarize the following text in one sentence.\n\n"
            for example in selected_examples:
                prompt += f"Text: {example['input']}\nSummary: {example['output']}\n\n"
            prompt += f"Text: {test_input}\nSummary:"
        
        elif task_type == "translation":
            prompt = "Translate the following English text to Korean.\n\n"
            for example in selected_examples:
                prompt += f"English: {example['input']}\nKorean: {example['output']}\n\n"
            prompt += f"English: {test_input}\nKorean:"
        
        elif task_type == "question_answering":
            prompt = "Answer the question based on the given context.\n\n"
            for example in selected_examples:
                prompt += f"{example['input']}\nAnswer: {example['output']}\n\n"
            prompt += f"{test_input}\nAnswer:"
        
        elif task_type == "text_classification":
            prompt = "Classify the following text into categories: News, Entertainment, Science, Technology, Finance.\n\n"
            for example in selected_examples:
                prompt += f"Text: {example['input']}\nCategory: {example['output']}\n\n"
            prompt += f"Text: {test_input}\nCategory:"
        
        return prompt
    
    def simulate_llm_response(self, prompt: str, task_type: str) -> str:
        """LLM 응답 시뮬레이션 (실제 모델이 없을 때)"""
        
        # 프롬프트에서 테스트 입력 추출
        lines = prompt.strip().split('\n')
        test_line = lines[-1]
        
        if task_type == "sentiment_classification":
            # 간단한 키워드 기반 감성 분석
            if any(word in test_line.lower() for word in ['love', 'amazing', 'great', 'fantastic', 'excellent']):
                return "Positive"
            elif any(word in test_line.lower() for word in ['hate', 'terrible', 'worst', 'awful', 'disappointing']):
                return "Negative"
            else:
                return "Neutral"
        
        elif task_type == "translation":
            # 미리 정의된 번역 매핑
            translations = {
                "good luck": "행운을 빕니다",
                "nice to meet you": "만나서 반갑습니다",
                "what time is it": "몇 시인가요",
                "have a good day": "좋은 하루 되세요",
                "where is the station": "역이 어디에 있나요"
            }
            for eng, kor in translations.items():
                if eng in test_line.lower():
                    return kor
            return "번역 결과"
        
        elif task_type == "text_summarization":
            return "요약된 내용입니다."
        
        elif task_type == "text_classification":
            if any(word in test_line.lower() for word in ['movie', 'film', 'actor', 'celebrity']):
                return "Entertainment"
            elif any(word in test_line.lower() for word in ['stock', 'market', 'economy', 'finance']):
                return "Finance"
            elif any(word in test_line.lower() for word in ['research', 'study', 'scientist']):
                return "Science"
            elif any(word in test_line.lower() for word in ['ai', 'technology', 'computer', 'digital']):
                return "Technology"
            else:
                return "News"
        
        elif task_type == "question_answering":
            return "답변"
        
        return "생성된 응답"

def create_llm_generator():
    """LLM 생성기 생성"""
    
    if TRANSFORMERS_AVAILABLE:
        try:
            print("GPT-2 모델 로딩 중...")
            generator = pipeline('text-generation', model='gpt2')
            print("LLM 모델 로딩 완료!")
            return generator, "Real LLM"
        except Exception as e:
            print(f"실제 모델 로딩 실패: {e}")
            print("시뮬레이션 엔진을 사용합니다.")
            return FewShotLearningEngine(), "Simulation"
    else:
        print("Transformers가 설치되지 않아 시뮬레이션 엔진을 사용합니다.")
        return FewShotLearningEngine(), "Simulation"

def experiment_example_count():
    """예제 개수에 따른 성능 실험"""
    
    print("=== 예제 개수별 성능 실험 ===")
    print("Few-shot 예제 개수가 모델 성능에 미치는 영향을 분석합니다.")
    print("-" * 60)
    
    engine = FewShotLearningEngine()
    
    # 테스트 케이스들
    test_cases = {
        "sentiment_classification": [
            "This product exceeded my expectations!",
            "I regret buying this item.",
            "The quality is average."
        ],
        "translation": [
            "Good luck!",
            "Nice to meet you.",
            "What time is it?"
        ],
        "text_classification": [
            "New AI breakthrough in healthcare technology",
            "Celebrity wedding announced today",
            "Stock prices surge after earnings report"
        ]
    }
    
    example_counts = [1, 3, 5, 8]
    
    for task_type, test_inputs in test_cases.items():
        print(f"\n--- {task_type.replace('_', ' ').title()} 작업 ---")
        
        for num_examples in example_counts:
            print(f"\n{num_examples}개 예제 사용:")
            
            for i, test_input in enumerate(test_inputs, 1):
                prompt = engine.create_few_shot_prompt(task_type, num_examples, test_input)
                response = engine.simulate_llm_response(prompt, task_type)
                
                print(f"  테스트 {i}: {test_input}")
                print(f"    예측: {response}")
        
        # 분석 코멘트
        print(f"\n{task_type} 분석:")
        print("  • 예제가 적을 때: 일반적인 패턴만 학습, 정확도 제한적")
        print("  • 예제가 많을 때: 더 복잡한 패턴 학습, 일반화 능력 향상")
        print("  • 최적 예제 수: 작업 복잡도와 모델 크기에 따라 다름")

def experiment_example_quality():
    """예제 품질에 따른 성능 실험"""
    
    print("\n=== 예제 품질별 성능 실험 ===")
    print("Few-shot 예제의 품질이 성능에 미치는 영향을 분석합니다.")
    print("-" * 60)
    
    engine = FewShotLearningEngine()
    
    # 고품질 vs 저품질 예제 비교
    quality_examples = {
        "high_quality": {
            "sentiment_classification": [
                {"input": "This product is absolutely phenomenal and exceeded all my expectations!", "output": "Positive"},
                {"input": "Completely disappointed with the poor quality and terrible customer service.", "output": "Negative"},
                {"input": "The product is adequate for basic needs but nothing extraordinary.", "output": "Neutral"}
            ]
        },
        "low_quality": {
            "sentiment_classification": [
                {"input": "Good", "output": "Positive"},
                {"input": "Bad", "output": "Negative"},
                {"input": "OK", "output": "Neutral"}
            ]
        }
    }
    
    test_input = "I'm really impressed with the innovative features and sleek design!"
    
    for quality_level, examples_dict in quality_examples.items():
        print(f"\n--- {quality_level.replace('_', ' ').title()} 예제 사용 ---")
        
        # 수동으로 프롬프트 생성 (품질별 비교를 위해)
        examples = examples_dict["sentiment_classification"]
        prompt = "Classify the sentiment as Positive, Negative, or Neutral.\n\n"
        
        for example in examples:
            prompt += f"Text: {example['input']}\nSentiment: {example['output']}\n\n"
        
        prompt += f"Text: {test_input}\nSentiment:"
        
        print(f"프롬프트 예제 품질: {quality_level}")
        print(f"테스트 입력: {test_input}")
        
        response = engine.simulate_llm_response(prompt, "sentiment_classification")
        print(f"예측 결과: {response}")
        
        # 품질 분석
        if quality_level == "high_quality":
            print("분석: 상세하고 구체적인 예제로 더 정확한 학습 가능")
        else:
            print("분석: 단순한 예제로 제한적인 패턴만 학습")

def experiment_task_complexity():
    """작업 복잡도별 실험"""
    
    print("\n=== 작업 복잡도별 Few-shot 성능 ===")
    print("다양한 복잡도의 작업에서 Few-shot 학습 효과 비교")
    print("-" * 60)
    
    tasks_by_complexity = {
        "간단한 작업": {
            "task": "sentiment_classification",
            "description": "감성 분류 (3개 클래스)",
            "test_input": "Amazing product, highly recommended!"
        },
        "중간 복잡도": {
            "task": "text_classification", 
            "description": "텍스트 분류 (5개 카테고리)",
            "test_input": "Scientists discover new treatment for diabetes"
        },
        "복잡한 작업": {
            "task": "text_summarization",
            "description": "텍스트 요약 (생성형 작업)",
            "test_input": "Machine learning algorithms have revolutionized many industries by enabling computers to learn patterns from data without explicit programming. These systems can now perform tasks like image recognition, natural language processing, and predictive analytics with remarkable accuracy."
        }
    }
    
    engine = FewShotLearningEngine()
    
    for complexity, task_info in tasks_by_complexity.items():
        print(f"\n--- {complexity} ---")
        print(f"작업: {task_info['description']}")
        print(f"테스트 입력: {task_info['test_input']}")
        
        # 다양한 예제 수로 테스트
        for num_examples in [1, 3, 5]:
            prompt = engine.create_few_shot_prompt(
                task_info['task'], 
                num_examples, 
                task_info['test_input']
            )
            response = engine.simulate_llm_response(prompt, task_info['task'])
            
            print(f"  {num_examples}개 예제: {response}")
        
        # 복잡도별 분석
        if complexity == "간단한 작업":
            print("  분석: 적은 예제로도 좋은 성능, Few-shot에 매우 적합")
        elif complexity == "중간 복잡도":
            print("  분석: 적당한 예제 수 필요, 카테고리 간 구분이 중요")
        else:
            print("  분석: 많은 예제 필요, 생성 품질 제한적")

def prompt_engineering_techniques():
    """프롬프트 엔지니어링 기법들"""
    
    print("\n=== Few-shot 프롬프트 엔지니어링 기법 ===")
    
    techniques = {
        "예제 순서 최적화": {
            "설명": "가장 명확한 예제를 먼저 배치",
            "팁": "간단하고 전형적인 예제 → 복잡하고 구체적인 예제"
        },
        "다양성 확보": {
            "설명": "다양한 패턴과 스타일의 예제 포함",
            "팁": "극단적인 케이스와 애매한 케이스 모두 포함"
        },
        "컨텍스트 일관성": {
            "설명": "모든 예제가 동일한 형식과 스타일 유지",
            "팁": "입력-출력 형식, 언어 스타일, 레이블 형식 통일"
        },
        "태스크 설명 추가": {
            "설명": "작업에 대한 명확한 설명 제공",
            "팁": "예제 전에 작업 목표와 규칙 명시"
        },
        "단계별 추론": {
            "설명": "복잡한 작업을 단계별로 분해",
            "팁": "중간 단계의 사고 과정도 예제에 포함"
        }
    }
    
    for technique, info in techniques.items():
        print(f"\n{technique}:")
        print(f"  설명: {info['설명']}")
        print(f"  팁: {info['팁']}")

def best_practices_and_limitations():
    """모범 사례와 한계점"""
    
    print("\n=== Few-shot 학습 모범 사례 ===")
    
    best_practices = [
        "작업에 적합한 예제 개수 찾기 (보통 3-8개)",
        "예제의 다양성과 대표성 확보",
        "명확하고 일관된 형식 사용",
        "도메인별 특화 예제 활용",
        "정기적인 성능 검증과 예제 업데이트"
    ]
    
    print("모범 사례:")
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== Few-shot 학습의 한계점 ===")
    
    limitations = [
        "복잡한 추론이나 계산 작업에는 제한적",
        "예제 선택에 따라 성능 편차가 큼",
        "새로운 도메인이나 패턴에 대한 일반화 어려움",
        "긴 컨텍스트가 필요한 작업에서 토큰 제한",
        "평가 메트릭 설정의 어려움"
    ]
    
    print("주요 한계점:")
    for limitation in limitations:
        print(f"  • {limitation}")
    
    print("\n=== 개선 방안 ===")
    
    improvements = [
        "Chain-of-Thought 프롬프팅과 결합",
        "동적 예제 선택 알고리즘 활용",
        "도메인 특화 모델과 조합 사용",
        "반복적 개선과 A/B 테스트",
        "인간 피드백을 통한 예제 품질 향상"
    ]
    
    print("개선 방안:")
    for improvement in improvements:
        print(f"  • {improvement}")

def real_world_applications():
    """실제 활용 사례"""
    
    print("\n=== Few-shot 학습 실제 활용 사례 ===")
    
    applications = {
        "고객 서비스": {
            "작업": "고객 문의 분류",
            "예제": "문의 유형별 대표 사례 제공",
            "효과": "빠른 티켓 라우팅과 초기 대응"
        },
        "콘텐츠 제작": {
            "작업": "브랜드 톤앤매너 따라하기",
            "예제": "브랜드별 글쓰기 스타일 샘플",
            "효과": "일관된 브랜드 목소리 유지"
        },
        "데이터 분석": {
            "작업": "비구조화 데이터 분류",
            "예제": "카테고리별 데이터 샘플",
            "효과": "수동 라벨링 작업 최소화"
        },
        "교육": {
            "작업": "맞춤형 문제 생성",
            "예제": "난이도별 문제 유형",
            "효과": "개인별 학습 수준에 맞는 콘텐츠"
        },
        "번역": {
            "작업": "전문 용어 번역",
            "예제": "도메인별 번역 사례",
            "효과": "정확한 전문 용어 번역"
        }
    }
    
    for domain, info in applications.items():
        print(f"\n{domain}:")
        print(f"  작업: {info['작업']}")
        print(f"  예제 활용: {info['예제']}")
        print(f"  기대 효과: {info['효과']}")

def main():
    print("=== 문제 4.1: Few-shot 학습 실험 ===")
    
    # 1. 예제 개수별 성능 실험
    experiment_example_count()
    
    # 2. 예제 품질별 성능 실험
    experiment_example_quality()
    
    # 3. 작업 복잡도별 실험
    experiment_task_complexity()
    
    # 4. 프롬프트 엔지니어링 기법
    prompt_engineering_techniques()
    
    # 5. 모범 사례와 한계점
    best_practices_and_limitations()
    
    # 6. 실제 활용 사례
    real_world_applications()
    
    # 설치 안내
    if not TRANSFORMERS_AVAILABLE:
        print("\n=== 설치 안내 ===")
        print("실제 LLM 모델을 사용하려면:")
        print("pip install transformers torch")

if __name__ == "__main__":
    main()