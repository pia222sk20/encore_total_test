"""
문제 4.3: Chain-of-Thought 프롬프팅 (개선된 버전)

개선 사항:
1. 모델 로딩 최적화: main 함수에서 모델을 한 번만 로딩하여 각 데모 함수에 전달합니다.
2. 기능 완성: 논리, 상식 추론 데모에도 실제 LLM을 호출하는 로직을 추가했습니다.
3. 성능 경고 추가: gpt2 같은 소형 모델의 한계를 명확히 설명하여 사용자 기대를 관리합니다.
4. 코드 명확성 개선: 파라미터 정리 및 구조 개선으로 가독성을 높였습니다.
"""

try:
    from transformers import pipeline, set_seed
    TRANSFORMERS_AVAILABLE = True
    print("Transformers 라이브러리가 사용 가능합니다.")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers 라이브러리가 설치되지 않았습니다. 시뮬레이션 모드로만 실행됩니다.")

import re
import random
from typing import List, Dict, Any
from dataclasses import dataclass

# --- 데이터 클래스 및 프롬프터

@dataclass
class ReasoningStep:
    """추론 단계 데이터 클래스"""
    step_number: int
    description: str
    calculation: str = ""
    result: str = ""

class ChainOfThoughtPrompter:
    """Chain-of-Thought 프롬프팅 시스템"""
    
    def __init__(self):
        # CoT 예제들
        self.math_examples = [
            {
                "question": "사라는 사과 15개를 가지고 있었습니다. 친구에게 4개를 주고, 상점에서 7개를 더 샀습니다. 이제 사라는 사과를 몇 개 가지고 있나요?",
                "reasoning": """단계별로 계산해보겠습니다.
1단계: 사라가 처음에 가진 사과 수를 확인합니다.
 - 처음 사과: 15개
2단계: 친구에게 준 사과를 빼줍니다.
 - 계산: 15 - 4 = 11개
 - 친구에게 준 후: 11개
3단계: 상점에서 산 사과를 더해줍니다.
 - 계산: 11 + 7 = 18개
 - 최종 사과 수: 18개""",
                "answer": "18개"
            },
            {
                "question": "한 반에 학생이 24명 있습니다. 이 중 2/3가 남학생이라면, 여학생은 몇 명인가요?",
                "reasoning": """단계별로 해결해보겠습니다.
1단계: 전체 학생 수를 확인합니다.
 - 전체 학생: 24명
2단계: 남학생 수를 계산합니다.
 - 남학생 비율: 2/3
 - 계산: 24 × (2/3) = 16명
 - 남학생: 16명
3단계: 여학생 수를 계산합니다.
 - 계산: 전체 - 남학생 = 24 - 16 = 8명
 - 여학생: 8명""",
                "answer": "8명"
            }
        ]
        
        self.logic_examples = [
            {
                "question": "모든 A는 B다. C는 A다. C는 B인가?",
                "reasoning": "C는 A이고 모든 A는 B이므로, C는 B이다.",
                "answer": "예"
            }
        ]
        
        self.common_sense_examples = [
            {
                "question": "비가 오는 날 밖에 나갈 때 가져가야 할 가장 중요한 물건은 무엇인가요?",
                "reasoning": """상식적으로 단계별 생각해보겠습니다.
1단계: 비가 오는 상황을 분석합니다. 비를 맞으면 몸이 젖고 불편합니다.
2단계: 비로부터 몸을 보호할 방법을 생각합니다. 비를 막을 수 있는 도구가 필요합니다.
3단계: 가장 효과적인 도구는 우산입니다. 휴대하기 편하고 비를 잘 막아줍니다.""",
                "answer": "우산"
            }
        ]
    
    def create_cot_prompt(self, question: str, problem_type: str = "math") -> str:
        """Chain-of-Thought 프롬프트 생성"""
        examples = []
        if problem_type == "math":
            examples = self.math_examples
        elif problem_type == "logic":
            examples = self.logic_examples
        elif problem_type == "common_sense":
            examples = self.common_sense_examples
        
        selected_example = random.choice(examples)
        
        prompt = f"다음은 '{problem_type}' 문제를 단계별로 해결하는 예시입니다.\n\n"
        prompt += f"--- 예시 시작 ---\n"
        prompt += f"문제: {selected_example['question']}\n"
        prompt += f"해결과정:\n{selected_example['reasoning']}\n"
        prompt += f"답: {selected_example['answer']}\n"
        prompt += f"--- 예시 끝 ---\n\n"
        prompt += f"이제 다음 문제를 같은 방식으로 단계별로 해결해주세요:\n"
        prompt += f"문제: {question}\n"
        prompt += f"해결과정:"
        
        return prompt

# --- 모델 로더 ---

def load_model_or_simulator():
    """실제 LLM 모델 또는 시뮬레이터를 로드합니다."""
    if TRANSFORMERS_AVAILABLE:
        try:
            print("\n⏳ GPT-2 모델을 로딩하고 있습니다. 잠시 기다려주세요...")
            # 재현성을 위해 시드 설정
            set_seed(42)
            # 좀 더 성능이 좋은 gpt2-medium 모델 사용
            generator = pipeline('text-generation', model='gpt2-medium')
            print("CoT LLM 모델 로딩 완료!")
            print("경고: gpt2-medium은 CoT 추론을 수행하기에 충분히 크지 않을 수 있습니다.")
            print("   부정확하거나 반복적인 결과가 나올 수 있으며, 이는 모델의 한계 때문입니다.\n")
            return generator, "Real LLM"
        except Exception as e:
            print(f"실제 모델 로딩 실패: {e}")
            print("🔄 CoT 시뮬레이션 엔진을 사용합니다.")
            return ChainOfThoughtPrompter(), "Simulation"
    else:
        print("\n🔄 CoT 시뮬레이션 엔진을 사용합니다.")
        return ChainOfThoughtPrompter(), "Simulation"

def run_demonstration(title, problems, problem_type, prompter, generator, system_type):
    """통합된 데모 실행 함수"""
    print(f"\n{'='*20} {title} {'='*20}")
    
    for i, problem in enumerate(problems, 1):
        print(f"\n--- 문제 {i}: {problem} ---")
        
        if system_type == "Real LLM":
            prompt = prompter.create_cot_prompt(problem, problem_type)
            try:
                result = generator(
                    prompt,
                    max_new_tokens=256,  # 새로 생성할 최대 토큰 수
                    temperature=0.4,
                    do_sample=True,
                    pad_token_id=generator.tokenizer.eos_token_id
                )
                response = result[0]['generated_text'][len(prompt):].strip()
                print(f"LLM 추론:\n{response}")
            except Exception as e:
                print(f"LLM 추론 오류: {e}")
        else: # Simulation
            # 시뮬레이션은 매우 단순화되어 있으므로, 개념 이해용으로만 사용합니다.
            print(" симуляция 단계별 해결과정 (개념 예시):")
            print(f" 1단계: 문제를 분석합니다.")
            print(f" 2단계: 해결에 필요한 단계를 구성합니다.")
            print(f" 3단계: 단계에 따라 최종 답을 도출합니다.")
            print(f"\n 최종 답: (시뮬레이션된 답변)")

def analyze_cot_effectiveness():
    """CoT 기법의 효과성 분석"""
    print(f"\n{'='*20} Chain-of-Thought 기법 효과성 분석 {'='*20}")
    print("\nCoT는 복잡한 문제를 작은 단계로 나누어 LLM의 추론 과정을 명확하게 만듭니다.")
    print("이를 통해 최종 답변의 정확도를 높이고, 과정의 투명성을 확보할 수 있습니다.")

def best_practices():
    """CoT 프롬프팅 모범 사례"""
    print(f"\n{'='*20} Chain-of-Thought 프롬프팅 모범 사례 {'='*20}")
    print("\n1. 명확한 예제: 문제와 유사한 유형의 고품질 예제를 1~3개 제공하는 것이 효과적입니다.")
    print("2. 단계적 표현: '단계별로 생각해보자'와 같은 명시적인 지시어를 사용하면 모델이 CoT를 따를 확률이 높아집니다.")
    print("3. 일관된 형식: 예제와 실제 문제의 형식을 일관되게 유지하는 것이 중요합니다.")

# --- 메인 실행 함수 ---

def main():
    """메인 프로그램 실행"""
    print("문제 4.3: Chain-of-Thought 프롬프팅")
    
    prompter = ChainOfThoughtPrompter()
    generator, system_type = load_model_or_simulator()

    # 수학 문제 데모
    math_problems = [
        "사과 5개와 오렌지 7개가 있습니다. 과일은 총 몇 개인가요?",
        "연필 12자루를 4명에게 똑같이 나눠주면 한 명당 몇 자루를 받나요?",
    ]
    run_demonstration("수학 문제 CoT 데모", math_problems, "math", prompter, generator, system_type)

    # 논리 추론 데모
    logic_problems = [
        "A는 B다. B는 C다. A는 C인가?",
        "비가 오면 춥다. 비가 온다. 지금 추운가?",
    ]
    run_demonstration("논리 추론 CoT 데모", logic_problems, "logic", prompter, generator, system_type)

    # 상식 추론 데모
    common_sense_problems = [
        "불이 났을 때 가장 먼저 해야 할 일은 무엇인가요?",
        "배가 고플 때, 가장 먼저 해야 할 일은?",
    ]
    run_demonstration("상식 추론 CoT 데모", common_sense_problems, "common_sense", prompter, generator, system_type)


    # 분석 및 요약 정보 출력
    analyze_cot_effectiveness()
    best_practices()

    if not TRANSFORMERS_AVAILABLE:
        print("\n=== 설치 안내 ===")
        print("실제 LLM 모델을 사용하려면 다음 명령어를 실행하세요:")
        print("pip install transformers torch tensorflow")

if __name__ == "__main__":
    main()