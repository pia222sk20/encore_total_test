"""
문제 3.1: GPT를 이용한 텍스트 완성

지시사항:
GPT 모델(Hugging Face transformers의 GPT-2 또는 유사한 모델)을 사용하여 
간단한 텍스트 완성 작업을 수행하세요. 주어진 프롬프트에 대해 여러 개의 
완성 결과를 생성하고 비교해보세요.
"""

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("Transformers 라이브러리가 사용 가능합니다.")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers 라이브러리가 설치되지 않았습니다.")

import random
import re

class MockGPTModel:
    """GPT가 없을 때 사용할 모의 텍스트 생성 모델"""
    
    def __init__(self):
        self.model_name = "Mock GPT-2 Model"
        # 간단한 템플릿 기반 생성
        self.story_templates = [
            " and discovered a hidden treasure beneath the old oak tree.",
            " but suddenly realized they were not alone in the forest.",
            " when a mysterious stranger appeared out of nowhere.",
            " and decided to embark on an adventure of a lifetime.",
            " which led to an unexpected friendship with a talking animal."
        ]
        
        self.business_templates = [
            " should focus on customer satisfaction and innovation.",
            " requires careful market research and strategic planning.",
            " can benefit from digital transformation and automation.",
            " needs to adapt to changing consumer preferences.",
            " must prioritize sustainability and social responsibility."
        ]
        
        self.tech_templates = [
            " will revolutionize how we interact with technology.",
            " has the potential to solve many complex problems.",
            " requires careful consideration of ethical implications.",
            " is becoming increasingly accessible to developers.",
            " will likely impact various industries in the coming years."
        ]
        
        self.general_templates = [
            " and this opens up many new possibilities.",
            " which represents a significant step forward.",
            " that could change everything we know about this topic.",
            " and the implications are far-reaching.",
            " but there are still challenges to overcome."
        ]
    
    def generate_completion(self, prompt, max_length=50, num_completions=3, temperature=0.7):
        """텍스트 완성 생성"""
        completions = []
        
        prompt_lower = prompt.lower()
        
        # 프롬프트 내용에 따라 적절한 템플릿 선택
        if any(word in prompt_lower for word in ['story', 'once', 'character', 'adventure']):
            templates = self.story_templates
        elif any(word in prompt_lower for word in ['business', 'company', 'market', 'strategy']):
            templates = self.business_templates
        elif any(word in prompt_lower for word in ['technology', 'ai', 'machine', 'computer']):
            templates = self.tech_templates
        else:
            templates = self.general_templates
        
        for i in range(num_completions):
            # temperature에 따라 랜덤성 조절
            if temperature > 0.8:
                template = random.choice(templates)
            elif temperature > 0.5:
                template = templates[i % len(templates)]
            else:
                template = templates[0]  # 가장 일반적인 것
            
            # 단어 수 조절
            words = template.split()
            if len(prompt.split()) + len(words) > max_length:
                words = words[:max_length - len(prompt.split())]
                template = " " + " ".join(words)
            
            completion = prompt + template
            completions.append(completion)
        
        return completions

def create_gpt_pipeline():
    """GPT 파이프라인 생성"""
    
    if TRANSFORMERS_AVAILABLE:
        try:
            print("GPT-2 모델 로딩 중...")
            # 텍스트 생성 파이프라인 생성
            generator = pipeline('text-generation', model='gpt2')
            print("GPT-2 모델 로딩 완료!")
            return generator, "Real GPT-2"
        except Exception as e:
            print(f"실제 모델 로딩 실패: {e}")
            print("모의 GPT 모델을 사용합니다.")
            return MockGPTModel(), "Mock GPT"
    else:
        print("Transformers가 설치되지 않아 모의 GPT 모델을 사용합니다.")
        return MockGPTModel(), "Mock GPT"

def demonstrate_text_completion():
    """텍스트 완성 데모"""
    
    print("=== GPT 텍스트 완성 데모 ===")
    
    # 테스트 프롬프트들
    test_prompts = [
        "Once upon a time, in a distant land",
        "The future of artificial intelligence",
        "Starting a successful business requires",
        "Climate change is a global challenge that",
        "The most important lesson I learned was"
    ]
    
    # GPT 모델 로드
    generator, model_type = create_gpt_pipeline()
    print(f"사용 모델: {model_type}")
    print("-" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n=== 프롬프트 {i}: {prompt} ===")
        
        if model_type == "Real GPT-2":
            # 실제 GPT-2 사용
            try:
                # 다양한 설정으로 생성
                results = generator(
                    prompt,
                    max_length=50,
                    num_return_sequences=3,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=generator.tokenizer.eos_token_id
                )
                
                for j, result in enumerate(results, 1):
                    completion = result['generated_text']
                    print(f"\n완성 {j}:")
                    print(f"  {completion}")
                    
            except Exception as e:
                print(f"생성 중 오류: {e}")
        
        else:
            # 모의 GPT 사용
            completions = generator.generate_completion(prompt, max_length=50, num_completions=3)
            
            for j, completion in enumerate(completions, 1):
                print(f"\n완성 {j}:")
                print(f"  {completion}")

def explore_generation_parameters():
    """생성 파라미터들 탐색"""
    
    print("\n=== 생성 파라미터 탐색 ===")
    
    prompt = "Artificial intelligence will"
    
    generator, model_type = create_gpt_pipeline()
    
    if model_type == "Real GPT-2":
        print("실제 GPT-2 모델을 사용한 파라미터 실험:")
        
        # 다양한 temperature 값 테스트
        temperatures = [0.1, 0.7, 1.2]
        
        for temp in temperatures:
            print(f"\n--- Temperature = {temp} ---")
            try:
                results = generator(
                    prompt,
                    max_length=40,
                    num_return_sequences=2,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=generator.tokenizer.eos_token_id
                )
                
                for i, result in enumerate(results, 1):
                    completion = result['generated_text']
                    # 원래 프롬프트 제거하고 완성 부분만 표시
                    new_text = completion[len(prompt):].strip()
                    print(f"결과 {i}: {prompt}{new_text}")
                    
            except Exception as e:
                print(f"생성 오류: {e}")
    
    else:
        print("모의 GPT 모델을 사용한 파라미터 시뮬레이션:")
        
        temperatures = [0.3, 0.7, 1.0]
        
        for temp in temperatures:
            print(f"\n--- Temperature = {temp} ---")
            completions = generator.generate_completion(
                prompt, 
                max_length=40, 
                num_completions=2, 
                temperature=temp
            )
            
            for i, completion in enumerate(completions, 1):
                print(f"결과 {i}: {completion}")
    
    # 파라미터 설명
    print("\n=== 생성 파라미터 설명 ===")
    
    param_explanations = {
        "temperature": {
            "설명": "생성의 랜덤성 조절",
            "낮은 값 (0.1-0.3)": "더 예측 가능하고 보수적인 텍스트",
            "중간 값 (0.7-0.9)": "창의적이면서도 일관성 있는 텍스트",
            "높은 값 (1.0+)": "매우 창의적이지만 때로는 일관성 없는 텍스트"
        },
        "max_length": {
            "설명": "생성할 최대 토큰 수",
            "짧은 길이": "간결한 완성",
            "긴 길이": "상세한 완성"
        },
        "top_k": {
            "설명": "각 단계에서 고려할 상위 k개 토큰",
            "낮은 값": "더 예측 가능한 선택",
            "높은 값": "더 다양한 선택"
        },
        "top_p": {
            "설명": "누적 확률이 p에 도달할 때까지의 토큰들 고려",
            "낮은 값": "높은 확률 토큰들만 사용",
            "높은 값": "더 많은 토큰 후보 고려"
        }
    }
    
    for param, info in param_explanations.items():
        print(f"\n{param.upper()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

def creative_writing_examples():
    """창작 활용 예시"""
    
    print("\n=== 창작 활용 예시 ===")
    
    creative_prompts = [
        {
            "genre": "공상과학",
            "prompt": "In the year 2150, humans discovered",
            "context": "미래 사회에서의 새로운 발견"
        },
        {
            "genre": "미스터리",
            "prompt": "The detective noticed something strange about the room",
            "context": "수사관의 관찰과 추리"
        },
        {
            "genre": "로맨스",
            "prompt": "Their eyes met across the crowded coffee shop and",
            "context": "우연한 만남의 순간"
        }
    ]
    
    generator, model_type = create_gpt_pipeline()
    
    for example in creative_prompts:
        print(f"\n--- {example['genre']} 장르 ---")
        print(f"상황: {example['context']}")
        print(f"프롬프트: {example['prompt']}")
        
        if model_type == "Real GPT-2":
            try:
                results = generator(
                    example['prompt'],
                    max_length=60,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=generator.tokenizer.eos_token_id
                )
                
                completion = results[0]['generated_text']
                print(f"생성 결과:\n  {completion}")
                
            except Exception as e:
                print(f"생성 오류: {e}")
        
        else:
            completions = generator.generate_completion(
                example['prompt'], 
                max_length=60, 
                num_completions=1,
                temperature=0.8
            )
            print(f"생성 결과:\n  {completions[0]}")

def practical_applications():
    """실용적 활용 방안"""
    
    print("\n=== 실용적 활용 방안 ===")
    
    applications = {
        "콘텐츠 제작": [
            "블로그 포스트 초안 작성",
            "소셜 미디어 캡션 생성",
            "제품 설명 자동 생성",
            "이메일 템플릿 작성"
        ],
        "교육": [
            "예시 문장 생성",
            "문제 출제 도움",
            "학습 자료 보완",
            "언어 학습 연습"
        ],
        "비즈니스": [
            "보고서 요약 초안",
            "제안서 아이디어 생성",
            "브레인스토밍 지원",
            "고객 응답 템플릿"
        ],
        "개발": [
            "코드 주석 생성",
            "문서 작성 지원",
            "API 설명 생성",
            "테스트 케이스 아이디어"
        ]
    }
    
    for category, uses in applications.items():
        print(f"\n{category}:")
        for use in uses:
            print(f"  • {use}")

def limitations_and_considerations():
    """한계점과 고려사항"""
    
    print("\n=== 한계점과 고려사항 ===")
    
    limitations = [
        "사실성 검증 필요 - 생성된 내용이 항상 정확하지 않음",
        "편향성 - 훈련 데이터의 편향이 결과에 반영될 수 있음",
        "일관성 - 긴 텍스트에서 논리적 일관성 유지 어려움",
        "저작권 - 훈련 데이터와 유사한 내용 생성 가능성",
        "윤리적 사용 - 악의적 목적으로 오용될 수 있음"
    ]
    
    print("주요 한계점:")
    for limitation in limitations:
        print(f"  • {limitation}")
    
    best_practices = [
        "생성된 내용은 항상 검토하고 수정하기",
        "중요한 정보는 별도로 사실 확인하기",
        "다양한 프롬프트로 여러 결과 비교하기",
        "최종 사용자에게 AI 생성임을 명시하기",
        "윤리적 가이드라인 준수하기"
    ]
    
    print("\n모범 사례:")
    for practice in best_practices:
        print(f"  • {practice}")

def main():
    print("=== 문제 3.1: GPT를 이용한 텍스트 완성 ===")
    
    # 1. 기본 텍스트 완성 데모
    demonstrate_text_completion()
    
    # 2. 생성 파라미터 탐색
    explore_generation_parameters()
    
    # 3. 창작 활용 예시
    creative_writing_examples()
    
    # 4. 실용적 활용 방안
    practical_applications()
    
    # 5. 한계점과 고려사항
    limitations_and_considerations()
    
    # 설치 안내
    if not TRANSFORMERS_AVAILABLE:
        print("\n=== 설치 안내 ===")
        print("실제 GPT-2 모델을 사용하려면:")
        print("pip install transformers torch")

if __name__ == "__main__":
    main()