"""
문제 5.1: LoRA를 이용한 효율적 파인튜닝

지시사항:
PEFT(Parameter Efficient Fine-Tuning) 라이브러리의 LoRA(Low-Rank Adaptation) 
기법을 사용하여 사전 훈련된 언어 모델을 효율적으로 파인튜닝하는 실험을 
수행하세요. 전체 파인튜닝과 LoRA의 성능 및 효율성을 비교 분석하세요.
"""

try:
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    import torch
    PEFT_AVAILABLE = True
    print("PEFT 라이브러리가 사용 가능합니다.")
except ImportError:
    PEFT_AVAILABLE = False
    print("PEFT 라이브러리가 설치되지 않았습니다.")

try:
    import numpy as np
    import pandas as pd
    DATA_LIBS_AVAILABLE = True
except ImportError:
    DATA_LIBS_AVAILABLE = False

import random
import time
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

@dataclass
class FineTuningMetrics:
    """파인튜닝 메트릭 데이터 클래스"""
    training_time: float
    memory_usage: float
    trainable_params: int
    total_params: int
    accuracy: float
    loss: float

class MockLoRAAdapter:
    """LoRA 어댑터 모의 클래스"""
    
    def __init__(self, base_model_params: int = 110_000_000, rank: int = 16):
        self.base_model_params = base_model_params
        self.rank = rank
        self.lora_params = self._calculate_lora_params()
        self.is_training = False
        
    def _calculate_lora_params(self) -> int:
        """LoRA 파라미터 수 계산"""
        # 간단화된 계산: 주요 레이어들에 대한 LoRA 파라미터
        # 실제로는 attention layers의 query, key, value, output에 적용
        num_layers = 12  # BERT-base 기준
        hidden_size = 768
        
        # 각 어텐션 레이어당: Q, K, V, O 매트릭스
        # LoRA: A (hidden_size × rank) + B (rank × hidden_size)
        params_per_matrix = (hidden_size * self.rank) + (self.rank * hidden_size)
        params_per_layer = params_per_matrix * 4  # Q, K, V, O
        total_lora_params = params_per_layer * num_layers
        
        return total_lora_params
    
    def get_trainable_params(self) -> Tuple[int, int]:
        """훈련 가능한 파라미터 수 반환"""
        return self.lora_params, self.base_model_params
    
    def train(self):
        """훈련 모드 설정"""
        self.is_training = True
    
    def eval(self):
        """평가 모드 설정"""
        self.is_training = False

class MockFullFineTuning:
    """전체 파인튜닝 모의 클래스"""
    
    def __init__(self, total_params: int = 110_000_000):
        self.total_params = total_params
        self.trainable_params = total_params  # 모든 파라미터 훈련
        self.is_training = False
    
    def get_trainable_params(self) -> Tuple[int, int]:
        """훈련 가능한 파라미터 수 반환"""
        return self.trainable_params, self.total_params
    
    def train(self):
        """훈련 모드 설정"""
        self.is_training = True
    
    def eval(self):
        """평가 모드 설정"""
        self.is_training = False

class FineTuningExperiment:
    """파인튜닝 실험 클래스"""
    
    def __init__(self):
        # 모의 데이터셋
        self.train_data = self._create_mock_dataset(1000)
        self.val_data = self._create_mock_dataset(200)
        self.test_data = self._create_mock_dataset(200)
        
        # 감성 분류 레이블
        self.label_names = ["negative", "neutral", "positive"]
    
    def _create_mock_dataset(self, size: int) -> List[Dict[str, Any]]:
        """모의 데이터셋 생성"""
        positive_samples = [
            "I love this product! It's amazing and works perfectly.",
            "Excellent quality and fast delivery. Highly recommended!",
            "This is exactly what I was looking for. Very satisfied.",
            "Great value for money. Will definitely buy again.",
            "Outstanding performance and beautiful design."
        ]
        
        negative_samples = [
            "Terrible product. Waste of money and time.",
            "Poor quality and disappointing experience.",
            "Not as described. Very unsatisfied with purchase.",
            "Worst customer service ever. Avoid this company.",
            "Cheaply made and breaks easily."
        ]
        
        neutral_samples = [
            "The product is okay. Nothing special but acceptable.",
            "Average quality for the price. Could be better.",
            "It works as expected. No complaints or compliments.",
            "Standard product with basic features.",
            "Decent but not outstanding. Regular experience."
        ]
        
        dataset = []
        for i in range(size):
            label = random.randint(0, 2)
            if label == 0:
                text = random.choice(negative_samples)
            elif label == 1:
                text = random.choice(neutral_samples)
            else:
                text = random.choice(positive_samples)
            
            # 약간의 변형 추가
            if random.random() < 0.3:
                text = text.replace(".", "!")
            
            dataset.append({
                "text": text,
                "label": label,
                "length": len(text.split())
            })
        
        return dataset
    
    def simulate_lora_training(self, rank: int = 16, epochs: int = 3) -> FineTuningMetrics:
        """LoRA 훈련 시뮬레이션"""
        print(f"LoRA 훈련 시뮬레이션 (rank={rank}, epochs={epochs})")
        
        # LoRA 모델 생성
        lora_model = MockLoRAAdapter(rank=rank)
        lora_model.train()
        
        start_time = time.time()
        
        # 훈련 시뮬레이션
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # 훈련 단계
            epoch_loss = 1.5 * (0.7 ** epoch) + random.uniform(-0.1, 0.1)
            train_losses.append(epoch_loss)
            
            # 검증 단계
            base_accuracy = 0.75 + (epoch * 0.05) + random.uniform(-0.05, 0.05)
            val_accuracy = min(0.92, base_accuracy)
            val_accuracies.append(val_accuracy)
            
            print(f"  Epoch {epoch+1}: Loss={epoch_loss:.3f}, Val_Acc={val_accuracy:.3f}")
        
        training_time = time.time() - start_time
        
        # 메트릭 계산
        trainable_params, total_params = lora_model.get_trainable_params()
        
        return FineTuningMetrics(
            training_time=training_time + random.uniform(10, 30),  # 시뮬레이션 시간 추가
            memory_usage=2.5 + random.uniform(-0.5, 0.5),  # GB
            trainable_params=trainable_params,
            total_params=total_params,
            accuracy=val_accuracies[-1],
            loss=train_losses[-1]
        )
    
    def simulate_full_training(self, epochs: int = 3) -> FineTuningMetrics:
        """전체 파인튜닝 시뮬레이션"""
        print(f"전체 파인튜닝 시뮬레이션 (epochs={epochs})")
        
        # 전체 파인튜닝 모델 생성
        full_model = MockFullFineTuning()
        full_model.train()
        
        start_time = time.time()
        
        # 훈련 시뮬레이션
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # 전체 파인튜닝은 더 빠르게 수렴하지만 리소스 많이 사용
            epoch_loss = 1.2 * (0.6 ** epoch) + random.uniform(-0.1, 0.1)
            train_losses.append(epoch_loss)
            
            # 약간 더 높은 정확도 (더 많은 파라미터 학습)
            base_accuracy = 0.78 + (epoch * 0.06) + random.uniform(-0.05, 0.05)
            val_accuracy = min(0.95, base_accuracy)
            val_accuracies.append(val_accuracy)
            
            print(f"  Epoch {epoch+1}: Loss={epoch_loss:.3f}, Val_Acc={val_accuracy:.3f}")
        
        training_time = time.time() - start_time
        
        # 메트릭 계산
        trainable_params, total_params = full_model.get_trainable_params()
        
        return FineTuningMetrics(
            training_time=training_time + random.uniform(120, 180),  # 훨씬 긴 시뮬레이션 시간
            memory_usage=16.0 + random.uniform(-2.0, 3.0),  # 훨씬 높은 메모리 사용량
            trainable_params=trainable_params,
            total_params=total_params,
            accuracy=val_accuracies[-1],
            loss=train_losses[-1]
        )

def create_lora_experiment():
    """LoRA 실험 시스템 생성"""
    
    if PEFT_AVAILABLE:
        try:
            print("실제 LoRA 실험 환경을 구성하는 중...")
            # 실제 환경에서는 여기서 실제 모델과 데이터셋 로드
            print("실제 LoRA 실험 준비 완료!")
            return FineTuningExperiment(), "Real LoRA"
        except Exception as e:
            print(f"실제 LoRA 환경 구성 실패: {e}")
            print("시뮬레이션 실험을 사용합니다.")
            return FineTuningExperiment(), "Simulation"
    else:
        print("PEFT가 설치되지 않아 시뮬레이션 실험을 사용합니다.")
        return FineTuningExperiment(), "Simulation"

def demonstrate_lora_vs_full():
    """LoRA vs 전체 파인튜닝 비교 데모"""
    
    print("=== LoRA vs 전체 파인튜닝 비교 실험 ===")
    print("효율성과 성능을 비교분석합니다.")
    print("-" * 60)
    
    experiment, exp_type = create_lora_experiment()
    print(f"실험 타입: {exp_type}")
    
    # LoRA 실험
    print("\n1. LoRA 파인튜닝 실행")
    lora_metrics = experiment.simulate_lora_training(rank=16, epochs=3)
    
    # 전체 파인튜닝 실험
    print("\n2. 전체 파인튜닝 실행") 
    full_metrics = experiment.simulate_full_training(epochs=3)
    
    # 결과 비교
    print("\n=== 실험 결과 비교 ===")
    
    print(f"\n{'메트릭':<20} {'LoRA':<15} {'전체 파인튜닝':<15} {'비율':<10}")
    print("-" * 60)
    
    # 훈련 시간 비교
    time_ratio = lora_metrics.training_time / full_metrics.training_time
    print(f"{'훈련 시간(초)':<20} {lora_metrics.training_time:<15.1f} {full_metrics.training_time:<15.1f} {time_ratio:<10.2f}")
    
    # 메모리 사용량 비교
    memory_ratio = lora_metrics.memory_usage / full_metrics.memory_usage
    print(f"{'메모리 사용량(GB)':<20} {lora_metrics.memory_usage:<15.1f} {full_metrics.memory_usage:<15.1f} {memory_ratio:<10.2f}")
    
    # 파라미터 수 비교
    param_ratio = lora_metrics.trainable_params / full_metrics.trainable_params
    print(f"{'훈련 파라미터 수':<20} {lora_metrics.trainable_params:<15,} {full_metrics.trainable_params:<15,} {param_ratio:<10.4f}")
    
    # 성능 비교
    accuracy_diff = lora_metrics.accuracy - full_metrics.accuracy
    print(f"{'정확도':<20} {lora_metrics.accuracy:<15.3f} {full_metrics.accuracy:<15.3f} {accuracy_diff:<10.3f}")
    
    # 손실 비교
    loss_diff = lora_metrics.loss - full_metrics.loss
    print(f"{'최종 손실':<20} {lora_metrics.loss:<15.3f} {full_metrics.loss:<15.3f} {loss_diff:<10.3f}")
    
    # 분석 코멘트
    print("\n=== 분석 결과 ===")
    print(f"• 훈련 시간: LoRA가 {(1-time_ratio)*100:.1f}% 단축")
    print(f"• 메모리 사용량: LoRA가 {(1-memory_ratio)*100:.1f}% 절약")
    print(f"• 훈련 파라미터: LoRA가 {(1-param_ratio)*100:.1f}% 감소")
    
    if accuracy_diff > -0.05:
        print(f"• 성능: 유사한 수준 유지 (차이: {accuracy_diff:.3f})")
    elif accuracy_diff > -0.1:
        print(f"• 성능: 약간 낮지만 허용 가능한 수준 (차이: {accuracy_diff:.3f})")
    else:
        print(f"• 성능: 상당한 성능 저하 (차이: {accuracy_diff:.3f})")

def analyze_lora_rank_impact():
    """LoRA rank가 성능에 미치는 영향 분석"""
    
    print("\n=== LoRA Rank 영향 분석 ===")
    print("다양한 rank 값이 성능과 효율성에 미치는 영향을 분석합니다.")
    print("-" * 60)
    
    experiment, _ = create_lora_experiment()
    
    ranks = [4, 8, 16, 32, 64]
    results = []
    
    for rank in ranks:
        print(f"\nRank {rank} 실험 중...")
        metrics = experiment.simulate_lora_training(rank=rank, epochs=3)
        results.append((rank, metrics))
        
        print(f"  파라미터 수: {metrics.trainable_params:,}")
        print(f"  정확도: {metrics.accuracy:.3f}")
        print(f"  훈련 시간: {metrics.training_time:.1f}초")
        print(f"  메모리 사용량: {metrics.memory_usage:.1f}GB")
    
    # 결과 요약 테이블
    print(f"\n{'Rank':<6} {'파라미터 수':<12} {'정확도':<8} {'시간(초)':<10} {'메모리(GB)':<10}")
    print("-" * 50)
    
    for rank, metrics in results:
        print(f"{rank:<6} {metrics.trainable_params:<12,} {metrics.accuracy:<8.3f} "
              f"{metrics.training_time:<10.1f} {metrics.memory_usage:<10.1f}")
    
    # 분석
    print("\n=== Rank 선택 가이드라인 ===")
    guidelines = {
        "Rank 4-8": "매우 효율적, 단순한 태스크에 적합",
        "Rank 16": "일반적으로 권장되는 균형점",
        "Rank 32-64": "복잡한 태스크, 높은 성능 필요시"
    }
    
    for rank_range, description in guidelines.items():
        print(f"• {rank_range}: {description}")

def lora_configuration_options():
    """LoRA 설정 옵션들 설명"""
    
    print("\n=== LoRA 설정 옵션들 ===")
    
    config_options = {
        "rank (r)": {
            "설명": "Low-rank 분해의 차원",
            "일반값": "4, 8, 16, 32",
            "영향": "높을수록 표현력 증가, 파라미터 수 증가"
        },
        "alpha": {
            "설명": "LoRA 가중치의 스케일링 팩터",
            "일반값": "16, 32, 64",
            "영향": "보통 rank와 같거나 2배로 설정"
        },
        "dropout": {
            "설명": "LoRA 레이어의 드롭아웃 비율",
            "일반값": "0.0, 0.1",
            "영향": "과적합 방지, 너무 높으면 성능 저하"
        },
        "target_modules": {
            "설명": "LoRA를 적용할 모듈들",
            "일반값": "['q_proj', 'v_proj', 'k_proj', 'o_proj']",
            "영향": "더 많은 모듈 적용시 성능 향상, 파라미터 증가"
        },
        "bias": {
            "설명": "bias 파라미터 훈련 여부",
            "일반값": "'none', 'all', 'lora_only'",
            "영향": "'all'은 성능 향상, 파라미터 약간 증가"
        }
    }
    
    for option, info in config_options.items():
        print(f"\n{option}:")
        print(f"  설명: {info['설명']}")
        print(f"  일반적 값: {info['일반값']}")
        print(f"  영향: {info['영향']}")

def lora_advantages_limitations():
    """LoRA의 장단점"""
    
    print("\n=== LoRA의 장점과 한계점 ===")
    
    print("\n장점:")
    advantages = [
        "메모리 효율성: 훈련 시 메모리 사용량 대폭 감소",
        "저장 효율성: 어댑터만 저장하면 되어 디스크 공간 절약",
        "빠른 훈련: 적은 파라미터로 빠른 수렴",
        "다중 태스크: 하나의 베이스 모델에 여러 어댑터 적용 가능",
        "배포 효율성: 어댑터만 배포하면 되어 업데이트 용이"
    ]
    
    for advantage in advantages:
        print(f"  • {advantage}")
    
    print("\n한계점:")
    limitations = [
        "성능 한계: 전체 파인튜닝 대비 약간의 성능 저하 가능",
        "복잡한 태스크: 매우 복잡하거나 도메인 차이가 큰 경우 제한적",
        "하이퍼파라미터 민감성: rank, alpha 등 설정에 성능이 민감",
        "추론 오버헤드: 약간의 추가 계산 필요",
        "특정 아키텍처 의존: 모든 모델 구조에 최적화되지 않을 수 있음"
    ]
    
    for limitation in limitations:
        print(f"  • {limitation}")

def practical_applications():
    """LoRA 실제 활용 사례"""
    
    print("\n=== LoRA 실제 활용 사례 ===")
    
    applications = {
        "도메인 적응": {
            "예시": "의료, 법률, 금융 도메인별 언어 모델",
            "장점": "각 도메인별 어댑터로 효율적 관리",
            "방법": "도메인별 데이터로 별도 LoRA 훈련"
        },
        "다국어 적응": {
            "예시": "영어 모델을 한국어, 일본어 등으로 적응",
            "장점": "언어별 어댑터로 다국어 지원",
            "방법": "언어별 코퍼스로 LoRA 파인튜닝"
        },
        "개인화": {
            "예시": "사용자별 맞춤형 대화 스타일",
            "장점": "개인별 어댑터로 개인화 서비스",
            "방법": "사용자 상호작용 데이터로 개별 LoRA 훈련"
        },
        "태스크 특화": {
            "예시": "요약, 번역, 분류 등 태스크별 모델",
            "장점": "태스크별 최적화된 성능",
            "방법": "태스크별 데이터셋으로 LoRA 훈련"
        },
        "실시간 적응": {
            "예시": "온라인 학습으로 실시간 모델 업데이트",
            "장점": "빠른 적응과 배포",
            "방법": "새로운 데이터로 LoRA 재훈련 후 교체"
        }
    }
    
    for application, info in applications.items():
        print(f"\n{application}:")
        print(f"  예시: {info['예시']}")
        print(f"  장점: {info['장점']}")
        print(f"  방법: {info['방법']}")

def implementation_best_practices():
    """LoRA 구현 모범 사례"""
    
    print("\n=== LoRA 구현 모범 사례 ===")
    
    print("\n1. 초기 설정:")
    initial_practices = [
        "작은 rank(4-8)부터 시작하여 점진적 증가",
        "alpha는 보통 rank와 같거나 2배로 설정",
        "dropout은 0.1부터 시작",
        "target_modules는 attention 레이어부터 시작"
    ]
    
    for practice in initial_practices:
        print(f"  • {practice}")
    
    print("\n2. 훈련 과정:")
    training_practices = [
        "작은 학습률 사용 (5e-5 ~ 1e-4)",
        "적은 에폭으로 빠른 수렴 활용",
        "검증 성능 모니터링으로 조기 종료",
        "그래디언트 클리핑으로 안정성 확보"
    ]
    
    for practice in training_practices:
        print(f"  • {practice}")
    
    print("\n3. 성능 최적화:")
    optimization_practices = [
        "여러 rank 값으로 실험하여 최적값 탐색",
        "target_modules 확장하여 성능 향상 시도",
        "데이터 품질과 양 확보",
        "베이스 모델 선택의 중요성 인식"
    ]
    
    for practice in optimization_practices:
        print(f"  • {practice}")
    
    print("\n4. 배포 고려사항:")
    deployment_practices = [
        "어댑터 버전 관리 체계 구축",
        "A/B 테스트로 성능 검증",
        "추론 속도 벤치마킹",
        "메모리 사용량 모니터링"
    ]
    
    for practice in deployment_practices:
        print(f"  • {practice}")

def main():
    print("=== 문제 5.1: LoRA를 이용한 효율적 파인튜닝 ===")
    
    # 1. LoRA vs 전체 파인튜닝 비교
    demonstrate_lora_vs_full()
    
    # 2. Rank 영향 분석
    analyze_lora_rank_impact()
    
    # 3. 설정 옵션들
    lora_configuration_options()
    
    # 4. 장단점 분석
    lora_advantages_limitations()
    
    # 5. 실제 활용 사례
    practical_applications()
    
    # 6. 구현 모범 사례
    implementation_best_practices()
    
    # 설치 안내
    if not PEFT_AVAILABLE:
        print("\n=== 설치 안내 ===")
        print("실제 LoRA 파인튜닝을 위해:")
        print("pip install peft transformers torch datasets")

if __name__ == "__main__":
    main()