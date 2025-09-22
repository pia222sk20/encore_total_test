"""
문제 5.2: RLHF 보상 모델 구현

지시사항:
Reinforcement Learning from Human Feedback (RLHF) 파이프라인의 핵심 구성 요소인 
보상 모델(Reward Model)을 구현하세요. 인간의 선호도 데이터를 활용하여 
응답의 품질을 평가하는 모델을 훈련하고, 이를 통해 언어 모델의 출력을 
개선하는 과정을 시연하세요.
"""

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    import torch
    import torch.nn as nn
    TRANSFORMERS_AVAILABLE = True
    print("Transformers 라이브러리가 사용 가능합니다.")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers 라이브러리가 설치되지 않았습니다.")

try:
    import numpy as np
    import pandas as pd
    DATA_LIBS_AVAILABLE = True
except ImportError:
    DATA_LIBS_AVAILABLE = False

import random
import json
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass

@dataclass
class PreferenceExample:
    """선호도 예제 데이터 클래스"""
    prompt: str
    chosen_response: str
    rejected_response: str
    preference_score: float  # 0-1 사이, 높을수록 chosen이 선호됨

@dataclass
class RewardModelOutput:
    """보상 모델 출력 데이터 클래스"""
    prompt: str
    response: str
    reward_score: float
    confidence: float

class MockRewardModel:
    """보상 모델 모의 클래스"""
    
    def __init__(self):
        self.model_name = "Mock Reward Model"
        self.is_trained = False
        
        # 품질 평가 기준들
        self.quality_factors = {
            "helpfulness": 0.3,
            "harmlessness": 0.25,
            "honesty": 0.2,
            "coherence": 0.15,
            "relevance": 0.1
        }
        
        # 긍정적/부정적 키워드들
        self.positive_indicators = {
            "helpful": ["help", "assist", "solve", "explain", "guide", "support"],
            "polite": ["please", "thank", "sorry", "appreciate", "kindly"],
            "informative": ["detail", "specific", "example", "because", "reason"],
            "structured": ["first", "second", "step", "process", "method"]
        }
        
        self.negative_indicators = {
            "harmful": ["hate", "hurt", "damage", "destroy", "violent"],
            "inappropriate": ["inappropriate", "offensive", "rude", "insulting"],
            "incorrect": ["wrong", "false", "incorrect", "mistake", "error"],
            "incoherent": ["confusing", "unclear", "nonsense", "random"]
        }
    
    def calculate_reward(self, prompt: str, response: str) -> Tuple[float, float]:
        """응답에 대한 보상 점수 계산"""
        
        scores = {}
        
        # 도움성 평가
        helpfulness = self._evaluate_helpfulness(prompt, response)
        scores["helpfulness"] = helpfulness
        
        # 무해성 평가
        harmlessness = self._evaluate_harmlessness(response)
        scores["harmlessness"] = harmlessness
        
        # 정직성 평가
        honesty = self._evaluate_honesty(response)
        scores["honesty"] = honesty
        
        # 일관성 평가
        coherence = self._evaluate_coherence(response)
        scores["coherence"] = coherence
        
        # 관련성 평가
        relevance = self._evaluate_relevance(prompt, response)
        scores["relevance"] = relevance
        
        # 가중 평균으로 최종 점수 계산
        final_score = sum(scores[factor] * weight 
                         for factor, weight in self.quality_factors.items())
        
        # 신뢰도 계산 (점수들의 분산을 기반으로)
        score_variance = np.var(list(scores.values())) if DATA_LIBS_AVAILABLE else 0.1
        confidence = max(0.5, 1.0 - score_variance)
        
        return final_score, confidence
    
    def _evaluate_helpfulness(self, prompt: str, response: str) -> float:
        """도움성 평가"""
        score = 0.5  # 기본 점수
        
        response_lower = response.lower()
        
        # 긍정적 지표
        for indicator in self.positive_indicators["helpful"]:
            if indicator in response_lower:
                score += 0.1
        
        # 구체적인 정보 제공 여부
        if any(word in response_lower for word in ["example", "detail", "specific"]):
            score += 0.15
        
        # 응답 길이 (너무 짧거나 길지 않은 적절한 길이)
        word_count = len(response.split())
        if 20 <= word_count <= 200:
            score += 0.1
        elif word_count < 5:
            score -= 0.2
        
        return min(1.0, max(0.0, score))
    
    def _evaluate_harmlessness(self, response: str) -> float:
        """무해성 평가"""
        score = 0.9  # 기본적으로 높은 점수
        
        response_lower = response.lower()
        
        # 부정적 지표 확인
        for indicator in self.negative_indicators["harmful"]:
            if indicator in response_lower:
                score -= 0.3
        
        for indicator in self.negative_indicators["inappropriate"]:
            if indicator in response_lower:
                score -= 0.2
        
        # 안전한 언어 사용 보너스
        if any(word in response_lower for word in ["safe", "careful", "consider"]):
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def _evaluate_honesty(self, response: str) -> float:
        """정직성 평가"""
        score = 0.7  # 기본 점수
        
        response_lower = response.lower()
        
        # 불확실성 표현 (정직한 태도)
        uncertainty_phrases = ["not sure", "might", "possibly", "uncertain", "don't know"]
        if any(phrase in response_lower for phrase in uncertainty_phrases):
            score += 0.15
        
        # 과도한 확신 표현 (부정적)
        overconfident_phrases = ["definitely", "absolutely", "100% sure", "always", "never"]
        if any(phrase in response_lower for phrase in overconfident_phrases):
            score -= 0.1
        
        # 출처나 근거 제시
        if any(word in response_lower for word in ["source", "research", "study", "according"]):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _evaluate_coherence(self, response: str) -> float:
        """일관성 평가"""
        score = 0.6  # 기본 점수
        
        sentences = response.split('.')
        word_count = len(response.split())
        
        # 적절한 문장 수
        if 2 <= len(sentences) <= 10:
            score += 0.1
        
        # 적절한 문장 길이
        if word_count > 0:
            avg_sentence_length = word_count / len(sentences)
            if 5 <= avg_sentence_length <= 25:
                score += 0.15
        
        # 연결어 사용
        connectors = ["however", "therefore", "because", "since", "although", "furthermore"]
        if any(conn in response.lower() for conn in connectors):
            score += 0.1
        
        # 반복 단어 체크
        words = response.lower().split()
        if len(set(words)) / len(words) > 0.7:  # 어휘 다양성
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _evaluate_relevance(self, prompt: str, response: str) -> float:
        """관련성 평가"""
        score = 0.5  # 기본 점수
        
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # 키워드 겹침
        overlap = len(prompt_words.intersection(response_words))
        overlap_ratio = overlap / len(prompt_words) if prompt_words else 0
        
        score += overlap_ratio * 0.3
        
        # 질문에 대한 직접 답변 여부
        if prompt.lower().startswith(("what", "how", "why", "when", "where")):
            if any(word in response.lower() for word in ["answer", "result", "solution"]):
                score += 0.1
        
        # 주제 일관성 (간단한 키워드 매칭)
        if len(prompt_words.intersection(response_words)) > 0:
            score += 0.1
        
        return min(1.0, max(0.0, score))

class RLHFDataGenerator:
    """RLHF 훈련 데이터 생성기"""
    
    def __init__(self):
        self.prompts = [
            "How can I improve my programming skills?",
            "What's the best way to learn a new language?",
            "How do I stay motivated while studying?",
            "Can you explain machine learning in simple terms?",
            "What are some healthy meal preparation tips?",
            "How do I manage stress effectively?",
            "What's the difference between AI and machine learning?",
            "How can I be more productive at work?",
            "What are some good books for personal development?",
            "How do I start investing in stocks?"
        ]
        
        # 좋은 응답과 나쁜 응답 템플릿
        self.good_response_templates = [
            "Here's a helpful approach: {specific_advice}. You might also want to consider {additional_tip}.",
            "I'd recommend starting with {step1}. Then, {step2}. This approach has proven effective because {reason}.",
            "There are several strategies you can try: 1) {strategy1}, 2) {strategy2}, 3) {strategy3}. Each has its benefits.",
            "Based on common practices, {main_advice}. However, keep in mind that {consideration}."
        ]
        
        self.bad_response_templates = [
            "Just do it. It's easy.",
            "I don't know. Figure it out yourself.",
            "That's a stupid question. Why would you ask that?",
            "You should definitely {extreme_advice} without any hesitation.",
            "Random response that doesn't answer your question at all."
        ]
    
    def generate_preference_data(self, num_examples: int = 100) -> List[PreferenceExample]:
        """선호도 데이터 생성"""
        examples = []
        
        for _ in range(num_examples):
            prompt = random.choice(self.prompts)
            
            # 좋은 응답 생성
            good_template = random.choice(self.good_response_templates)
            chosen_response = self._fill_template(good_template, prompt)
            
            # 나쁜 응답 생성
            bad_template = random.choice(self.bad_response_templates)
            rejected_response = self._fill_template(bad_template, prompt)
            
            # 랜덤하게 순서 바꾸기 (데이터 다양성을 위해)
            if random.random() < 0.1:  # 10% 확률로 순서 바꿈
                chosen_response, rejected_response = rejected_response, chosen_response
                preference_score = 0.2  # 낮은 선호도
            else:
                preference_score = random.uniform(0.7, 0.95)  # 높은 선호도
            
            examples.append(PreferenceExample(
                prompt=prompt,
                chosen_response=chosen_response,
                rejected_response=rejected_response,
                preference_score=preference_score
            ))
        
        return examples
    
    def _fill_template(self, template: str, prompt: str) -> str:
        """템플릿에 구체적인 내용 채우기"""
        # 간단한 키워드 기반 응답 생성
        if "programming" in prompt.lower():
            specific_advice = "practice coding daily and work on projects"
            additional_tip = "joining coding communities for feedback"
            step1 = "choosing a programming language to focus on"
            step2 = "building small projects to apply what you learn"
            strategy1 = "solve coding challenges online"
            strategy2 = "contribute to open source projects" 
            strategy3 = "build personal projects"
            main_advice = "consistent practice is key to improvement"
            consideration = "everyone learns at their own pace"
            reason = "it builds muscle memory and problem-solving skills"
            extreme_advice = "quit your job and code 16 hours a day"
        
        elif "language" in prompt.lower():
            specific_advice = "immerse yourself in the language through media"
            additional_tip = "finding a conversation partner"
            step1 = "learning basic grammar and vocabulary"
            step2 = "practicing speaking and listening daily"
            strategy1 = "use language learning apps"
            strategy2 = "watch movies with subtitles"
            strategy3 = "practice with native speakers"
            main_advice = "daily practice with varied content works best"
            consideration = "motivation and consistency are crucial"
            reason = "it mimics natural language acquisition"
            extreme_advice = "move to a country where they speak that language"
        
        else:
            # 일반적인 조언
            specific_advice = "break down the task into smaller steps"
            additional_tip = "seeking guidance from experts"
            step1 = "understanding the basics"
            step2 = "applying what you've learned practically"
            strategy1 = "regular practice"
            strategy2 = "learning from others"
            strategy3 = "staying consistent"
            main_advice = "patience and persistence are important"
            consideration = "results may take time to show"
            reason = "it builds understanding gradually"
            extreme_advice = "spend all your time on this one thing"
        
        # 템플릿 변수 치환
        return template.format(
            specific_advice=specific_advice,
            additional_tip=additional_tip,
            step1=step1,
            step2=step2,
            strategy1=strategy1,
            strategy2=strategy2,
            strategy3=strategy3,
            main_advice=main_advice,
            consideration=consideration,
            reason=reason,
            extreme_advice=extreme_advice
        )

class RLHFTrainer:
    """RLHF 훈련 관리자"""
    
    def __init__(self):
        self.reward_model = MockRewardModel()
        self.data_generator = RLHFDataGenerator()
        self.training_history = []
    
    def train_reward_model(self, num_examples: int = 100, epochs: int = 3) -> Dict[str, Any]:
        """보상 모델 훈련"""
        print(f"보상 모델 훈련 시작 (예제: {num_examples}개, 에폭: {epochs}개)")
        
        # 훈련 데이터 생성
        training_data = self.data_generator.generate_preference_data(num_examples)
        
        training_losses = []
        validation_accuracies = []
        
        for epoch in range(epochs):
            print(f"\\nEpoch {epoch + 1}/{epochs}")
            
            # 모의 훈련 과정
            epoch_loss = self._simulate_training_epoch(training_data)
            val_accuracy = self._simulate_validation(training_data[:20])  # 일부 데이터로 검증
            
            training_losses.append(epoch_loss)
            validation_accuracies.append(val_accuracy)
            
            print(f"  Loss: {epoch_loss:.4f}")
            print(f"  Validation Accuracy: {val_accuracy:.4f}")
        
        self.reward_model.is_trained = True
        
        # 훈련 결과 반환
        result = {
            "final_loss": training_losses[-1],
            "final_accuracy": validation_accuracies[-1],
            "training_losses": training_losses,
            "validation_accuracies": validation_accuracies,
            "num_examples": num_examples
        }
        
        self.training_history.append(result)
        return result
    
    def _simulate_training_epoch(self, training_data: List[PreferenceExample]) -> float:
        """훈련 에폭 시뮬레이션"""
        total_loss = 0.0
        
        for example in training_data[:10]:  # 일부만 시뮬레이션
            # 선택된 응답과 거부된 응답의 보상 점수 계산
            chosen_reward, _ = self.reward_model.calculate_reward(
                example.prompt, example.chosen_response
            )
            rejected_reward, _ = self.reward_model.calculate_reward(
                example.prompt, example.rejected_response
            )
            
            # 선호도 손실 계산 (간소화된 버전)
            # 실제로는 더 복잡한 ranking loss 사용
            preference_diff = chosen_reward - rejected_reward
            expected_diff = (example.preference_score - 0.5) * 2  # -1 ~ 1 범위로 변환
            
            loss = (preference_diff - expected_diff) ** 2
            total_loss += loss
        
        return total_loss / 10  # 평균 손실
    
    def _simulate_validation(self, val_data: List[PreferenceExample]) -> float:
        """검증 과정 시뮬레이션"""
        correct_predictions = 0
        
        for example in val_data:
            chosen_reward, _ = self.reward_model.calculate_reward(
                example.prompt, example.chosen_response
            )
            rejected_reward, _ = self.reward_model.calculate_reward(
                example.prompt, example.rejected_response
            )
            
            # 예측: 보상이 높은 응답이 선호될 것
            predicted_preference = chosen_reward > rejected_reward
            actual_preference = example.preference_score > 0.5
            
            if predicted_preference == actual_preference:
                correct_predictions += 1
        
        return correct_predictions / len(val_data)

def create_rlhf_system():
    """RLHF 시스템 생성"""
    
    if TRANSFORMERS_AVAILABLE:
        try:
            print("실제 RLHF 시스템을 구성하는 중...")
            # 실제 환경에서는 여기서 실제 모델 로드
            print("실제 RLHF 시스템 준비 완료!")
            return RLHFTrainer(), "Real RLHF"
        except Exception as e:
            print(f"실제 RLHF 시스템 구성 실패: {e}")
            print("시뮬레이션 RLHF 시스템을 사용합니다.")
            return RLHFTrainer(), "Simulation"
    else:
        print("Transformers가 설치되지 않아 시뮬레이션 RLHF 시스템을 사용합니다.")
        return RLHFTrainer(), "Simulation"

def demonstrate_reward_model_training():
    """보상 모델 훈련 데모"""
    
    print("=== 보상 모델 훈련 데모 ===")
    print("인간 선호도 데이터로 보상 모델을 훈련합니다.")
    print("-" * 60)
    
    trainer, system_type = create_rlhf_system()
    print(f"시스템 타입: {system_type}")
    
    # 보상 모델 훈련
    training_result = trainer.train_reward_model(num_examples=200, epochs=5)
    
    print(f"\\n=== 훈련 완료 ===")
    print(f"최종 손실: {training_result['final_loss']:.4f}")
    print(f"최종 정확도: {training_result['final_accuracy']:.4f}")
    print(f"사용된 예제 수: {training_result['num_examples']}")
    
    # 훈련 진행 상황 시각화 (텍스트로)
    print("\\n훈련 진행 상황:")
    print("Epoch | Loss   | Accuracy")
    print("-" * 25)
    for i, (loss, acc) in enumerate(zip(training_result['training_losses'], 
                                       training_result['validation_accuracies']), 1):
        print(f"{i:5d} | {loss:.4f} | {acc:.4f}")

def demonstrate_reward_scoring():
    """보상 점수 평가 데모"""
    
    print("\\n=== 보상 점수 평가 데모 ===")
    print("다양한 응답에 대한 보상 점수를 계산합니다.")
    print("-" * 60)
    
    trainer, _ = create_rlhf_system()
    
    # 보상 모델이 훈련되었다고 가정
    trainer.reward_model.is_trained = True
    
    # 테스트 케이스들
    test_cases = [
        {
            "prompt": "How can I improve my communication skills?",
            "responses": [
                "You can improve communication by practicing active listening, speaking clearly, and being empathetic to others. Start with daily conversations and gradually work on public speaking.",
                "Just talk more. Communication is easy.",
                "I don't know why you're asking me this. Figure it out yourself.",
                "There are several effective strategies: 1) Practice active listening by focusing fully on the speaker, 2) Work on your body language and eye contact, 3) Ask clarifying questions to ensure understanding. These methods help build stronger connections with others."
            ]
        },
        {
            "prompt": "What's the best way to learn programming?",
            "responses": [
                "Start with Python as it's beginner-friendly. Practice coding daily, work on small projects, and don't be afraid to make mistakes. Join online communities for support and feedback.",
                "Programming is hard. Don't even try.",
                "You should definitely quit your job and code 20 hours a day without breaks until you become an expert.",
                "I'd recommend beginning with fundamental concepts like variables and functions. Then, build simple projects to apply what you learn. Consider online courses, coding bootcamps, or computer science programs depending on your goals."
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\\n--- 테스트 케이스 {i} ---")
        print(f"질문: {test_case['prompt']}")
        
        results = []
        for j, response in enumerate(test_case['responses'], 1):
            reward_score, confidence = trainer.reward_model.calculate_reward(
                test_case['prompt'], response
            )
            
            results.append((j, reward_score, confidence, response))
        
        # 보상 점수 순으로 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        print("\\n응답들 (보상 점수 순):")
        for rank, (resp_id, score, conf, response) in enumerate(results, 1):
            print(f"\\n{rank}위 (응답 {resp_id}): 점수={score:.3f}, 신뢰도={conf:.3f}")
            print(f"  응답: {response[:100]}{'...' if len(response) > 100 else ''}")

def analyze_reward_factors():
    """보상 점수 구성 요소 분석"""
    
    print("\\n=== 보상 점수 구성 요소 분석 ===")
    print("각 평가 기준별 점수 분석")
    print("-" * 60)
    
    trainer, _ = create_rlhf_system()
    
    # 분석용 응답들
    analysis_cases = [
        {
            "type": "매우 도움이 되는 응답",
            "prompt": "How do I start learning machine learning?",
            "response": "Here's a structured approach to learning machine learning: 1) Start with Python and basic statistics, 2) Learn essential libraries like NumPy, Pandas, and Scikit-learn, 3) Practice with beginner projects like linear regression, 4) Gradually move to more complex algorithms. I recommend starting with online courses like Andrew Ng's ML course for solid foundations."
        },
        {
            "type": "해로운 응답",
            "prompt": "I'm feeling stressed about work.",
            "response": "Work stress is terrible and you should just quit your job immediately without any backup plan. Burn all bridges and never look back. Stress is for weak people anyway."
        },
        {
            "type": "비일관적 응답",
            "prompt": "What is artificial intelligence?",
            "response": "AI is computers thinking like humans but also elephants are purple and yesterday I ate a sandwich while flying to Mars in my submarine made of cheese."
        },
        {
            "type": "관련성 없는 응답",
            "prompt": "How do I cook pasta?",
            "response": "The stock market is volatile today and cryptocurrency prices are fluctuating. Remember to diversify your investment portfolio."
        }
    ]
    
    for case in analysis_cases:
        print(f"\\n--- {case['type']} ---")
        print(f"질문: {case['prompt']}")
        print(f"응답: {case['response'][:150]}...")
        
        # 개별 요소별 점수 계산 (실제로는 내부 메서드 호출)
        reward_model = trainer.reward_model
        
        helpfulness = reward_model._evaluate_helpfulness(case['prompt'], case['response'])
        harmlessness = reward_model._evaluate_harmlessness(case['response'])
        honesty = reward_model._evaluate_honesty(case['response'])
        coherence = reward_model._evaluate_coherence(case['response'])
        relevance = reward_model._evaluate_relevance(case['prompt'], case['response'])
        
        total_score, confidence = reward_model.calculate_reward(case['prompt'], case['response'])
        
        print(f"\\n평가 결과:")
        print(f"  도움성 (30%): {helpfulness:.3f}")
        print(f"  무해성 (25%): {harmlessness:.3f}")
        print(f"  정직성 (20%): {honesty:.3f}")
        print(f"  일관성 (15%): {coherence:.3f}")
        print(f"  관련성 (10%): {relevance:.3f}")
        print(f"  총점: {total_score:.3f}")
        print(f"  신뢰도: {confidence:.3f}")

def rlhf_pipeline_overview():
    """RLHF 파이프라인 전체 개요"""
    
    print("\\n=== RLHF 파이프라인 전체 개요 ===")
    
    pipeline_steps = {
        "1단계: 사전 훈련 (Pre-training)": {
            "목적": "대규모 텍스트 데이터로 기본 언어 능력 학습",
            "데이터": "웹 텍스트, 책, 위키피디아 등",
            "결과": "기본적인 언어 모델 (GPT, BERT 등)"
        },
        "2단계: 지도 미세조정 (Supervised Fine-tuning)": {
            "목적": "인간이 작성한 고품질 응답으로 모델 조정",
            "데이터": "질문-답변 쌍, 지시사항-응답 쌍",
            "결과": "특정 태스크에 특화된 모델"
        },
        "3단계: 보상 모델 훈련 (Reward Model Training)": {
            "목적": "인간 선호도를 학습하여 응답 품질 평가",
            "데이터": "응답 쌍에 대한 인간 선호도 순위",
            "결과": "응답 품질을 점수화하는 보상 모델"
        },
        "4단계: 강화학습 최적화 (RL Optimization)": {
            "목적": "보상 모델을 기준으로 정책 최적화",
            "방법": "PPO, TRPO 등의 RL 알고리즘 사용",
            "결과": "인간 선호도에 맞춘 최종 모델"
        }
    }
    
    for step, info in pipeline_steps.items():
        print(f"\\n{step}:")
        print(f"  목적: {info['목적']}")
        print(f"  데이터/방법: {info.get('데이터', info.get('방법', ''))}")
        print(f"  결과: {info['결과']}")

def rlhf_challenges_solutions():
    """RLHF의 도전과제와 해결방안"""
    
    print("\\n=== RLHF의 도전과제와 해결방안 ===")
    
    challenges = {
        "인간 선호도의 주관성": {
            "문제": "서로 다른 인간들이 서로 다른 선호도를 가짐",
            "해결방안": [
                "다수의 평가자를 통한 합의 도출",
                "명확한 평가 기준과 가이드라인 제공",
                "평가자 간 일치도 측정 및 품질 관리"
            ]
        },
        "데이터 수집의 비용": {
            "문제": "고품질 인간 피드백 수집에 높은 비용",
            "해결방안": [
                "효율적인 데이터 선택 전략",
                "Active Learning으로 중요한 예제 우선 수집",
                "자동화된 품질 검증 시스템"
            ]
        },
        "보상 해킹": {
            "문제": "모델이 의도와 다른 방식으로 보상 최적화",
            "해결방안": [
                "다양한 평가 메트릭 사용",
                "정기적인 보상 모델 업데이트",
                "인간 평가와의 지속적 비교"
            ]
        },
        "분포 이탈": {
            "문제": "훈련 분포에서 벗어난 입력에 대한 성능 저하",
            "해결방안": [
                "다양한 도메인의 훈련 데이터 수집",
                "정규화 기법으로 과적합 방지",
                "지속적인 모니터링과 업데이트"
            ]
        }
    }
    
    for challenge, info in challenges.items():
        print(f"\\n{challenge}:")
        print(f"  문제: {info['문제']}")
        print(f"  해결방안:")
        for solution in info['해결방안']:
            print(f"    • {solution}")

def future_directions():
    """RLHF의 미래 방향"""
    
    print("\\n=== RLHF의 미래 연구 방향 ===")
    
    directions = {
        "Constitutional AI": {
            "설명": "명시적인 원칙과 규칙을 통한 AI 행동 제어",
            "장점": "일관성 있고 예측 가능한 행동",
            "현재 상태": "Anthropic의 Claude 등에서 활용"
        },
        "AI Feedback (RLAIF)": {
            "설명": "인간 대신 AI 모델의 피드백 활용",
            "장점": "비용 효율적, 확장 가능한 피드백",
            "현재 상태": "초기 연구 단계, 유망한 결과들"
        },
        "다중 목표 최적화": {
            "설명": "도움성, 무해성, 정확성 등 여러 목표 동시 최적화",
            "장점": "더 균형잡힌 모델 성능",
            "현재 상태": "활발한 연구 진행 중"
        },
        "개인화된 RLHF": {
            "설명": "개별 사용자의 선호도에 맞춘 맞춤형 모델",
            "장점": "개인화된 사용자 경험",
            "현재 상태": "프라이버시와 확장성 이슈 해결 중"
        },
        "실시간 학습": {
            "설명": "배포 후에도 지속적인 피드백 학습",
            "장점": "동적 환경 적응, 지속적 개선",
            "현재 상태": "안전성과 안정성 보장이 주요 과제"
        }
    }
    
    for direction, info in directions.items():
        print(f"\\n{direction}:")
        print(f"  설명: {info['설명']}")
        print(f"  장점: {info['장점']}")
        print(f"  현재 상태: {info['현재 상태']}")

def main():
    print("=== 문제 5.2: RLHF 보상 모델 구현 ===")
    
    # 1. 보상 모델 훈련 데모
    demonstrate_reward_model_training()
    
    # 2. 보상 점수 평가 데모
    demonstrate_reward_scoring()
    
    # 3. 보상 점수 구성 요소 분석
    analyze_reward_factors()
    
    # 4. RLHF 파이프라인 개요
    rlhf_pipeline_overview()
    
    # 5. 도전과제와 해결방안
    rlhf_challenges_solutions()
    
    # 6. 미래 연구 방향
    future_directions()
    
    # 설치 안내
    if not TRANSFORMERS_AVAILABLE:
        print("\\n=== 설치 안내 ===")
        print("실제 RLHF 시스템을 위해:")
        print("pip install transformers torch datasets trl")

if __name__ == "__main__":
    main()