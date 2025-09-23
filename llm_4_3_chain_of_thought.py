"""
문제 4.3: Chain-of-Thought 프롬프팅

지시사항:
Chain-of-Thought (CoT) 프롬프팅 기법을 구현하여 복잡한 추론 문제를 
단계별로 해결하는 시스템을 만들어보세요. 수학 문제, 논리 문제, 
상식 추론 등 다양한 유형의 문제에 적용해보세요.
"""

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    print("Transformers 라이브러리가 사용 가능합니다.")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers 라이브러리가 설치되지 않았습니다.")

import re
import random
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

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
   - 계산: 24 × (2/3) = 24 × 2 ÷ 3 = 48 ÷ 3 = 16명
   - 남학생: 16명

3단계: 여학생 수를 계산합니다.
   - 계산: 전체 - 남학생 = 24 - 16 = 8명
   - 여학생: 8명""",
                "answer": "8명"
            }
        ]
        
        self.logic_examples = [
            {
                "question": "모든 고양이는 포유류입니다. 모든 포유류는 동물입니다. 페르시안은 고양이입니다. 따라서 페르시안은 무엇인가요?",
                "reasoning": """논리적으로 단계별 추론해보겠습니다.
1단계: 주어진 정보를 정리합니다.
   - 전제 1: 모든 고양이는 포유류이다
   - 전제 2: 모든 포유류는 동물이다
   - 전제 3: 페르시안은 고양이다

2단계: 페르시안이 포유류인지 확인합니다.
   - 페르시안은 고양이 (전제 3)
   - 모든 고양이는 포유류 (전제 1)
   - 따라서: 페르시안은 포유류

3단계: 페르시안이 동물인지 확인합니다.
   - 페르시안은 포유류 (2단계 결과)
   - 모든 포유류는 동물 (전제 2)
   - 따라서: 페르시안은 동물""",
                "answer": "페르시안은 동물입니다"
            }
        ]
        
        self.common_sense_examples = [
            {
                "question": "비가 오는 날 밖에 나갈 때 가져가야 할 가장 중요한 물건은 무엇인가요?",
                "reasoning": """상식적으로 단계별 생각해보겠습니다.
1단계: 비가 오는 상황을 분석합니다.
   - 상황: 하늘에서 물(비)이 떨어지고 있음
   - 문제: 젖을 수 있음

2단계: 비로부터 보호받는 방법을 생각합니다.
   - 방법 1: 실내에 머물기 (하지만 밖에 나가야 함)
   - 방법 2: 비를 막을 수 있는 도구 사용

3단계: 비를 막을 수 있는 도구들을 나열합니다.
   - 우산: 휴대 가능, 비를 효과적으로 막음
   - 우비: 몸 전체를 보호하지만 번거로움
   - 모자: 머리만 보호, 제한적

4단계: 가장 실용적인 선택을 결정합니다.
   - 우산이 가장 효과적이고 편리함""",
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
        
        # 예제 1-2개 선택
        selected_examples = random.sample(examples, min(1, len(examples)))
        
        prompt = f"다음은 {problem_type} 문제를 단계별로 해결하는 예시입니다.\n\n"
        
        for i, example in enumerate(selected_examples, 1):
            prompt += f"예시 {i}:\n"
            prompt += f"문제: {example['question']}\n"
            prompt += f"해결과정:\n{example['reasoning']}\n"
            prompt += f"답: {example['answer']}\n\n"
        
        prompt += f"이제 다음 문제를 같은 방식으로 단계별로 해결해주세요:\n"
        prompt += f"문제: {question}\n"
        prompt += f"해결과정:"
        
        return prompt
    
    def simulate_cot_reasoning(self, question: str, problem_type: str = "math") -> Dict[str, Any]:
        """CoT 추론 시뮬레이션"""
        
        steps = []
        
        if problem_type == "math":
            steps = self._simulate_math_reasoning(question)
        elif problem_type == "logic":
            steps = self._simulate_logic_reasoning(question)
        elif problem_type == "common_sense":
            steps = self._simulate_common_sense_reasoning(question)
        
        # 최종 답안 생성
        if steps:
            final_answer = steps[-1].result
        else:
            final_answer = "답을 찾을 수 없습니다."
        
        return {
            "question": question,
            "reasoning_steps": steps,
            "final_answer": final_answer
        }
    
    def _simulate_math_reasoning(self, question: str) -> List[ReasoningStep]:
        """수학 문제 추론 시뮬레이션"""
        steps = []
        
        # 숫자 추출
        numbers = re.findall(r'\d+', question)
        
        if len(numbers) >= 2:
            steps.append(ReasoningStep(
                step_number=1,
                description="문제에서 주어진 숫자들을 확인합니다.",
                result=f"주어진 숫자: {', '.join(numbers)}"
            ))
            
            # 간단한 연산 시뮬레이션
            if "더" in question or "추가" in question or "샀" in question:
                calculation = f"{numbers[0]} + {numbers[1]}"
                result = str(int(numbers[0]) + int(numbers[1]))
                steps.append(ReasoningStep(
                    step_number=2,
                    description="덧셈을 수행합니다.",
                    calculation=calculation,
                    result=f"계산 결과: {result}"
                ))
            
            elif "빼" in question or "주고" in question or "사용" in question:
                calculation = f"{numbers[0]} - {numbers[1]}"
                result = str(int(numbers[0]) - int(numbers[1]))
                steps.append(ReasoningStep(
                    step_number=2,
                    description="뺄셈을 수행합니다.",
                    calculation=calculation,
                    result=f"계산 결과: {result}"
                ))
            
            elif "곱" in question or "배" in question:
                calculation = f"{numbers[0]} × {numbers[1]}"
                result = str(int(numbers[0]) * int(numbers[1]))
                steps.append(ReasoningStep(
                    step_number=2,
                    description="곱셈을 수행합니다.",
                    calculation=calculation,
                    result=f"계산 결과: {result}"
                ))
        
        return steps
    
    def _simulate_logic_reasoning(self, question: str) -> List[ReasoningStep]:
        """논리 문제 추론 시뮬레이션"""
        steps = []
        
        steps.append(ReasoningStep(
            step_number=1,
            description="주어진 전제들을 정리합니다.",
            result="전제들을 체계적으로 나열하였습니다."
        ))
        
        steps.append(ReasoningStep(
            step_number=2,
            description="논리적 연결 관계를 파악합니다.",
            result="전제들 간의 논리적 관계를 확인하였습니다."
        ))
        
        steps.append(ReasoningStep(
            step_number=3,
            description="결론을 도출합니다.",
            result="논리적 추론을 통해 결론에 도달하였습니다."
        ))
        
        return steps
    
    def _simulate_common_sense_reasoning(self, question: str) -> List[ReasoningStep]:
        """상식 추론 시뮬레이션"""
        steps = []
        
        steps.append(ReasoningStep(
            step_number=1,
            description="상황을 분석합니다.",
            result="주어진 상황의 특징을 파악하였습니다."
        ))
        
        steps.append(ReasoningStep(
            step_number=2,
            description="가능한 선택지들을 고려합니다.",
            result="여러 대안들을 검토하였습니다."
        ))
        
        steps.append(ReasoningStep(
            step_number=3,
            description="최적의 해결책을 선택합니다.",
            result="가장 적절한 답을 결정하였습니다."
        ))
        
        return steps

def create_cot_generator():
    """CoT 생성기 생성"""
    
    if TRANSFORMERS_AVAILABLE:
        try:
            print("GPT-2 모델 로딩 중...")
            generator = pipeline('text-generation', model='gpt2')
            print("CoT LLM 모델 로딩 완료!")
            return generator, "Real LLM"
        except Exception as e:
            print(f"실제 모델 로딩 실패: {e}")
            print("CoT 시뮬레이션 엔진을 사용합니다.")
            return ChainOfThoughtPrompter(), "Simulation"
    else:
        print("Transformers가 설치되지 않아 CoT 시뮬레이션 엔진을 사용합니다.")
        return ChainOfThoughtPrompter(), "Simulation"

def demonstrate_math_cot():
    """수학 문제 CoT 데모"""
    
    print("=== 수학 문제 Chain-of-Thought 데모 ===")
    print("복잡한 수학 문제를 단계별로 해결합니다.")
    print("-" * 60)
    
    generator, system_type = create_cot_generator()
    
    math_problems = [
        "철수는 구슬 25개를 가지고 있었습니다. 영희에게 8개를 주고, 민수에게 5개를 주었습니다. 그 후 상점에서 12개를 더 샀습니다. 이제 철수는 구슬을 몇 개 가지고 있나요?",
        "한 상자에 사탕이 36개 들어있습니다. 3명의 아이들이 똑같이 나누어 가지려고 합니다. 각자 몇 개씩 가져갈 수 있나요?",
        "직사각형의 가로가 8cm, 세로가 5cm입니다. 이 직사각형의 넓이와 둘레는 각각 얼마인가요?"
    ]
    
    for i, problem in enumerate(math_problems, 1):
        print(f"\n--- 수학 문제 {i} ---")
        print(f"문제: {problem}")
        
        if system_type == "Real LLM":
            # 실제 LLM 사용
            prompt = generator.create_cot_prompt(problem, "math")
            try:
                result = generator(
                    prompt,
                    max_length=200,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=generator.tokenizer.eos_token_id
                )
                
                response = result[0]['generated_text'][len(prompt):].strip()
                print(f"LLM 추론:\n{response}")
                
            except Exception as e:
                print(f"LLM 추론 오류: {e}")
        
        else:
            # 시뮬레이션 사용
            result = generator.simulate_cot_reasoning(problem, "math")
            
            print("단계별 해결과정:")
            for step in result["reasoning_steps"]:
                print(f"{step.step_number}단계: {step.description}")
                if step.calculation:
                    print(f"  계산: {step.calculation}")
                print(f"  결과: {step.result}")
            
            print(f"\n최종 답: {result['final_answer']}")

def demonstrate_logic_cot():
    """논리 문제 CoT 데모"""
    
    print("\n=== 논리 추론 Chain-of-Thought 데모 ===")
    print("복잡한 논리 문제를 단계별로 해결합니다.")
    print("-" * 60)
    
    generator, system_type = create_cot_generator()
    
    logic_problems = [
        "모든 새는 날 수 있습니다. 펭귄은 새입니다. 하지만 펭귄은 날 수 없습니다. 이 명제들에 모순이 있나요?",
        "A가 B보다 크고, B가 C보다 큽니다. 그리고 C가 D보다 큽니다. A, B, C, D 중 가장 작은 것은 무엇인가요?",
        "만약 비가 오면 땅이 젖습니다. 땅이 젖어 있습니다. 따라서 비가 왔다고 결론지을 수 있나요?"
    ]
    
    for i, problem in enumerate(logic_problems, 1):
        print(f"\n--- 논리 문제 {i} ---")
        print(f"문제: {problem}")
        
        if system_type == "Simulation":
            result = generator.simulate_cot_reasoning(problem, "logic")
            
            print("논리적 추론 과정:")
            for step in result["reasoning_steps"]:
                print(f"{step.step_number}단계: {step.description}")
                print(f"  분석: {step.result}")
            
            # 특별한 논리 분석
            if i == 1:
                print("\n특별 분석: 펭귄 역설")
                print("  이는 '예외가 있는 일반화'의 예시입니다.")
                print("  '일반적으로 새는 날 수 있다'가 더 정확한 표현입니다.")
            elif i == 2:
                print("\n특별 분석: 추이적 관계")
                print("  A > B > C > D 순서로, D가 가장 작습니다.")
            elif i == 3:
                print("\n특별 분석: 논리적 오류")
                print("  이는 '후건 긍정의 오류'입니다.")
                print("  결과로부터 원인을 확신할 수 없습니다.")

def demonstrate_common_sense_cot():
    """상식 추론 CoT 데모"""
    
    print("\n=== 상식 추론 Chain-of-Thought 데모 ===")
    print("일상적인 상식 문제를 단계별로 해결합니다.")
    print("-" * 60)
    
    generator, system_type = create_cot_generator()
    
    common_sense_problems = [
        "겨울에 외출할 때 가져가야 할 중요한 물건들은 무엇인가요?",
        "친구의 생일 파티에 가는데 적절한 선물은 무엇일까요?",
        "새로운 도시로 이사했을 때 가장 먼저 해야 할 일은 무엇인가요?"
    ]
    
    for i, problem in enumerate(common_sense_problems, 1):
        print(f"\n--- 상식 문제 {i} ---")
        print(f"문제: {problem}")
        
        if system_type == "Simulation":
            result = generator.simulate_cot_reasoning(problem, "common_sense")
            
            print("상식적 추론 과정:")
            for step in result["reasoning_steps"]:
                print(f"{step.step_number}단계: {step.description}")
                print(f"  고려사항: {step.result}")
            
            # 구체적인 상식 답변
            if i == 1:
                print("\n구체적 제안:")
                print("  • 따뜻한 옷 (코트, 장갑, 목도리)")
                print("  • 미끄럼 방지 신발")
                print("  • 핫팩이나 보온용품")
            elif i == 2:
                print("\n구체적 제안:")
                print("  • 개인의 취미나 관심사 고려")
                print("  • 실용적이면서 의미있는 물건")
                print("  • 예산에 맞는 적절한 가격")
            elif i == 3:
                print("\n구체적 제안:")
                print("  • 주요 생활 인프라 파악 (병원, 마트, 은행)")
                print("  • 대중교통 노선 확인")
                print("  • 이웃과 인사하기")

def analyze_cot_effectiveness():
    """CoT 기법의 효과성 분석"""
    
    print("\n=== Chain-of-Thought 기법 효과성 분석 ===")
    
    print("\n1. CoT 기법의 장점:")
    advantages = [
        "복잡한 문제를 작은 단위로 분해",
        "추론 과정의 투명성 확보",
        "중간 단계에서의 오류 발견 용이",
        "학습과 이해에 도움",
        "신뢰성 있는 답변 생성"
    ]
    
    for advantage in advantages:
        print(f"  • {advantage}")
    
    print("\n2. 적용 분야별 효과:")
    effectiveness = {
        "수학 문제": {
            "효과": "매우 높음",
            "이유": "명확한 계산 단계, 검증 가능한 중간 결과"
        },
        "논리 추론": {
            "효과": "높음", 
            "이유": "전제와 결론의 명확한 연결 고리"
        },
        "상식 추론": {
            "효과": "중간",
            "이유": "주관적 판단이 포함되어 일관성 확보 어려움"
        },
        "창작 활동": {
            "효과": "낮음",
            "이유": "창의성과 단계적 접근의 상충"
        }
    }
    
    for field, info in effectiveness.items():
        print(f"\n{field}:")
        print(f"  효과: {info['효과']}")
        print(f"  이유: {info['이유']}")

def cot_variations():
    """CoT 기법의 변형들"""
    
    print("\n=== Chain-of-Thought 기법 변형들 ===")
    
    variations = {
        "Zero-shot CoT": {
            "설명": "예제 없이 '단계별로 생각해보세요' 프롬프트만 추가",
            "장점": "예제 준비 불필요, 다양한 문제에 적용 가능",
            "단점": "성능이 few-shot보다 낮을 수 있음"
        },
        "Few-shot CoT": {
            "설명": "1-3개의 단계별 해결 예제 제공",
            "장점": "높은 성능, 구체적인 추론 패턴 학습",
            "단점": "예제 준비 필요, 도메인별 맞춤화 요구"
        },
        "Self-Consistency": {
            "설명": "여러 번 추론하여 가장 일관된 답 선택",
            "장점": "정확도 향상, 오류 감소",
            "단점": "계산 비용 증가"
        },
        "Tree of Thoughts": {
            "설명": "여러 추론 경로를 트리 구조로 탐색",
            "장점": "복잡한 문제 해결 능력 향상",
            "단점": "매우 높은 계산 비용"
        },
        "Program-aided CoT": {
            "설명": "추론 과정에 코드 실행 단계 포함",
            "장점": "정확한 계산, 복잡한 수학 문제 해결",
            "단점": "코드 생성 능력 필요"
        }
    }
    
    for variation, info in variations.items():
        print(f"\n{variation}:")
        print(f"  설명: {info['설명']}")
        print(f"  장점: {info['장점']}")
        print(f"  단점: {info['단점']}")

def best_practices():
    """CoT 프롬프팅 모범 사례"""
    
    print("\n=== Chain-of-Thought 프롬프팅 모범 사례 ===")
    
    print("\n1. 프롬프트 설계:")
    prompt_tips = [
        "명확하고 구체적인 단계별 예제 제공",
        "'단계별로', '차근차근' 등의 지시어 사용",
        "중간 계산 결과 명시적으로 표시",
        "최종 답안을 명확히 구분",
        "일관된 형식과 스타일 유지"
    ]
    
    for tip in prompt_tips:
        print(f"  • {tip}")
    
    print("\n2. 예제 선택:")
    example_tips = [
        "문제 유형과 복잡도가 유사한 예제",
        "추론 단계가 명확하고 논리적인 예제",
        "중간 실수나 수정 과정도 포함 가능",
        "다양한 접근 방법을 보여주는 예제",
        "너무 많지 않은 적절한 수의 예제 (1-3개)"
    ]
    
    for tip in example_tips:
        print(f"  • {tip}")
    
    print("\n3. 품질 검증:")
    validation_tips = [
        "추론 과정의 논리적 일관성 확인",
        "계산 결과의 정확성 검증",
        "다른 방법으로도 같은 답이 나오는지 확인",
        "상식적으로 타당한 결과인지 점검",
        "단계별 설명이 이해하기 쉬운지 평가"
    ]
    
    for tip in validation_tips:
        print(f"  • {tip}")

def limitations_and_considerations():
    """한계점과 고려사항"""
    
    print("\n=== Chain-of-Thought 한계점과 고려사항 ===")
    
    print("\n1. 주요 한계점:")
    limitations = [
        "토큰 사용량 증가로 인한 비용 상승",
        "추론 시간 증가",
        "단계별 오류 누적 가능성",
        "주관적이거나 창의적 문제에는 부적합",
        "모델의 기본 능력에 여전히 의존"
    ]
    
    for limitation in limitations:
        print(f"  • {limitation}")
    
    print("\n2. 고려사항:")
    considerations = [
        "문제 유형에 따른 적용 여부 신중 결정",
        "비용 대비 성능 향상 효과 평가",
        "사용자의 이해도와 신뢰도 고려",
        "도메인별 예제와 패턴 개발 필요",
        "정기적인 성능 평가와 개선"
    ]
    
    for consideration in considerations:
        print(f"  • {consideration}")
    
    print("\n3. 개선 방향:")
    improvements = [
        "자동화된 예제 생성 시스템",
        "오류 감지 및 수정 메커니즘",
        "도메인별 특화 CoT 패턴",
        "효율적인 추론 경로 탐색",
        "인간 피드백을 통한 품질 향상"
    ]
    
    for improvement in improvements:
        print(f"  • {improvement}")

def main():
    print("=== 문제 4.3: Chain-of-Thought 프롬프팅 ===")
    
    # 1. 수학 문제 CoT 데모
    demonstrate_math_cot()
    
    # 2. 논리 추론 CoT 데모
    demonstrate_logic_cot()
    
    # 3. 상식 추론 CoT 데모
    demonstrate_common_sense_cot()
    
    # 4. CoT 효과성 분석
    analyze_cot_effectiveness()
    
    # 5. CoT 기법 변형들
    cot_variations()
    
    # 6. 모범 사례
    best_practices()
    
    # 7. 한계점과 고려사항
    limitations_and_considerations()
    
    # 설치 안내
    if not TRANSFORMERS_AVAILABLE:
        print("\n=== 설치 안내 ===")
        print("실제 LLM 모델을 사용하려면:")
        print("pip install transformers torch")

if __name__ == "__main__":
    main()