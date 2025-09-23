"""
문제 3.2: 생성 파라미터 실험

지시사항:
다양한 생성 파라미터(temperature, top_k, top_p 등)가 텍스트 생성 결과에 
미치는 영향을 실험해보세요. 동일한 프롬프트에 대해 다른 파라미터 설정으로 
생성된 결과들을 비교 분석하세요.
"""

try:
    from transformers import pipeline, set_seed
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("Transformers 라이브러리가 사용 가능합니다.")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers 라이브러리가 설치되지 않았습니다.")

import random
import json
from typing import Dict, List, Any

class ParameterExperiment:
    """생성 파라미터 실험 클래스"""
    
    def __init__(self):
        self.results = {}
        self.base_prompt = "The future of technology"
        
        # 모의 생성을 위한 어휘
        self.tech_vocab = [
            "artificial intelligence", "machine learning", "quantum computing",
            "blockchain", "virtual reality", "augmented reality", "robotics",
            "biotechnology", "nanotechnology", "renewable energy", "automation",
            "digital transformation", "cloud computing", "edge computing",
            "internet of things", "5G networks", "cybersecurity", "big data"
        ]
        
        self.continuation_patterns = {
            "conservative": [
                "will continue to evolve",
                "is expected to grow",
                "shows promising developments",
                "has significant potential",
                "requires careful consideration"
            ],
            "moderate": [
                "might revolutionize how we",
                "could transform various industries",
                "presents both opportunities and challenges",
                "is rapidly advancing across multiple domains",
                "will likely reshape our daily lives"
            ],
            "creative": [
                "dances on the edge of impossibility",
                "whispers secrets of tomorrow's dreams",
                "unleashes a symphony of digital wonders",
                "paints reality with strokes of pure innovation",
                "transcends the boundaries of human imagination"
            ]
        }
    
    def mock_generate(self, prompt: str, temperature: float, top_k: int = None, 
                     top_p: float = None, max_length: int = 50) -> str:
        """모의 텍스트 생성"""
        
        # temperature에 따른 스타일 결정
        if temperature < 0.3:
            style = "conservative"
            randomness = 0.1
        elif temperature < 0.8:
            style = "moderate"
            randomness = 0.5
        else:
            style = "creative"
            randomness = 0.9
        
        # 시드 설정 (재현 가능한 결과를 위해)
        random.seed(int(temperature * 100))
        
        # 기본 연결 선택
        continuation = random.choice(self.continuation_patterns[style])
        
        # top_k/top_p 시뮬레이션: 어휘 선택 제한
        available_vocab = self.tech_vocab.copy()
        
        if top_k:
            # top_k 시뮬레이션: 상위 k개만 선택
            available_vocab = available_vocab[:min(top_k, len(available_vocab))]
        
        if top_p:
            # top_p 시뮬레이션: 확률에 따른 어휘 제한
            cutoff = int(len(available_vocab) * top_p)
            available_vocab = available_vocab[:max(1, cutoff)]
        
        # 추가 내용 생성
        num_concepts = min(3, max(1, int(randomness * 4)))
        selected_concepts = random.sample(available_vocab, 
                                        min(num_concepts, len(available_vocab)))
        
        # 텍스트 조합
        result = f"{prompt} {continuation}"
        
        if selected_concepts:
            if style == "creative":
                concept_text = f" through {' and '.join(selected_concepts)}"
            else:
                concept_text = f" in areas like {', '.join(selected_concepts)}"
            
            result += concept_text
        
        # 길이 제한
        words = result.split()
        if len(words) > max_length:
            words = words[:max_length]
            result = " ".join(words) + "..."
        
        return result

def create_text_generator():
    """텍스트 생성기 생성"""
    
    if TRANSFORMERS_AVAILABLE:
        try:
            print("GPT-2 모델 로딩 중...")
            generator = pipeline('text-generation', model='gpt2')
            print("GPT-2 모델 로딩 완료!")
            return generator, "Real GPT-2"
        except Exception as e:
            print(f"실제 모델 로딩 실패: {e}")
            print("모의 생성기를 사용합니다.")
            return ParameterExperiment(), "Mock Generator"
    else:
        print("Transformers가 설치되지 않아 모의 생성기를 사용합니다.")
        return ParameterExperiment(), "Mock Generator"

def experiment_temperature():
    """Temperature 파라미터 실험"""
    
    print("=== Temperature 실험 ===")
    print("Temperature는 생성의 랜덤성을 조절합니다.")
    print("낮은 값: 예측 가능, 높은 값: 창의적")
    print("-" * 50)
    
    generator, model_type = create_text_generator()
    prompt = "The future of artificial intelligence"
    temperatures = [0.1, 0.5, 0.8, 1.2]
    
    results = {}
    
    for temp in temperatures:
        print(f"\n--- Temperature = {temp} ---")
        
        if model_type == "Real GPT-2":
            try:
                # 재현 가능한 결과를 위해 시드 설정
                set_seed(42)
                outputs = generator(
                    prompt,
                    max_length=40,
                    num_return_sequences=2,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=generator.tokenizer.eos_token_id
                )
                
                generated_texts = [output['generated_text'] for output in outputs]
                
            except Exception as e:
                print(f"생성 오류: {e}")
                generated_texts = [f"오류로 인한 모의 결과: {prompt} [temp={temp}]"]
        
        else:
            # 모의 생성
            generated_texts = [
                generator.mock_generate(prompt, temp, max_length=40),
                generator.mock_generate(prompt, temp + 0.01, max_length=40)  # 약간 다른 결과
            ]
        
        results[temp] = generated_texts
        
        for i, text in enumerate(generated_texts, 1):
            print(f"결과 {i}: {text}")
        
        # 특성 분석
        print(f"특성: ", end="")
        if temp < 0.3:
            print("매우 보수적, 예측 가능한 결과")
        elif temp < 0.7:
            print("균형잡힌 창의성과 일관성")
        elif temp < 1.0:
            print("높은 창의성, 다양한 표현")
        else:
            print("매우 창의적, 때로는 예상치 못한 결과")
    
    return results

def experiment_top_k():
    """Top-k 파라미터 실험"""
    
    print("\n=== Top-k 실험 ===")
    print("Top-k는 각 단계에서 고려할 상위 k개 토큰을 제한합니다.")
    print("낮은 값: 보수적 선택, 높은 값: 다양한 선택")
    print("-" * 50)
    
    generator, model_type = create_text_generator()
    prompt = "Technology companies should"
    top_k_values = [10, 50, 100, None]  # None은 제한 없음
    
    results = {}
    
    for top_k in top_k_values:
        k_label = top_k if top_k else "무제한"
        print(f"\n--- Top-k = {k_label} ---")
        
        if model_type == "Real GPT-2":
            try:
                set_seed(42)
                
                # top_k 파라미터 설정
                generate_kwargs = {
                    'max_length': 35,
                    'num_return_sequences': 2,
                    'temperature': 0.7,
                    'do_sample': True,
                    'pad_token_id': generator.tokenizer.eos_token_id
                }
                
                if top_k:
                    generate_kwargs['top_k'] = top_k
                
                outputs = generator(prompt, **generate_kwargs)
                generated_texts = [output['generated_text'] for output in outputs]
                
            except Exception as e:
                print(f"생성 오류: {e}")
                generated_texts = [f"오류로 인한 모의 결과: {prompt} [top_k={top_k}]"]
        
        else:
            # 모의 생성
            generated_texts = [
                generator.mock_generate(prompt, 0.7, top_k=top_k or 18, max_length=35),
                generator.mock_generate(prompt, 0.72, top_k=top_k or 18, max_length=35)
            ]
        
        results[k_label] = generated_texts
        
        for i, text in enumerate(generated_texts, 1):
            print(f"결과 {i}: {text}")
        
        # 특성 분석
        print(f"특성: ", end="")
        if not top_k or top_k > 80:
            print("높은 다양성, 창의적 선택")
        elif top_k > 30:
            print("적당한 다양성, 균형잡힌 선택")
        else:
            print("제한된 다양성, 안전한 선택")
    
    return results

def experiment_top_p():
    """Top-p (nucleus sampling) 파라미터 실험"""
    
    print("\n=== Top-p (Nucleus Sampling) 실험 ===")
    print("Top-p는 누적 확률이 p에 도달할 때까지의 토큰들을 고려합니다.")
    print("낮은 값: 고확률 토큰만, 높은 값: 더 많은 후보 고려")
    print("-" * 50)
    
    generator, model_type = create_text_generator()
    prompt = "In the next decade, we will see"
    top_p_values = [0.1, 0.5, 0.9, 0.95]
    
    results = {}
    
    for top_p in top_p_values:
        print(f"\n--- Top-p = {top_p} ---")
        
        if model_type == "Real GPT-2":
            try:
                set_seed(42)
                outputs = generator(
                    prompt,
                    max_length=40,
                    num_return_sequences=2,
                    temperature=0.8,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=generator.tokenizer.eos_token_id
                )
                
                generated_texts = [output['generated_text'] for output in outputs]
                
            except Exception as e:
                print(f"생성 오류: {e}")
                generated_texts = [f"오류로 인한 모의 결과: {prompt} [top_p={top_p}]"]
        
        else:
            # 모의 생성
            generated_texts = [
                generator.mock_generate(prompt, 0.8, top_p=top_p, max_length=40),
                generator.mock_generate(prompt, 0.82, top_p=top_p, max_length=40)
            ]
        
        results[top_p] = generated_texts
        
        for i, text in enumerate(generated_texts, 1):
            print(f"결과 {i}: {text}")
        
        # 특성 분석
        print(f"특성: ", end="")
        if top_p < 0.3:
            print("매우 보수적, 예측 가능한 선택")
        elif top_p < 0.7:
            print("적당한 다양성, 품질과 창의성의 균형")
        elif top_p < 0.9:
            print("높은 다양성, 창의적 표현")
        else:
            print("최대 다양성, 예상치 못한 창의성")
    
    return results

def combined_parameter_experiment():
    """여러 파라미터 조합 실험"""
    
    print("\n=== 파라미터 조합 실험 ===")
    print("다양한 파라미터 조합이 결과에 미치는 영향")
    print("-" * 50)
    
    generator, model_type = create_text_generator()
    prompt = "The biggest challenge facing humanity"
    
    # 다양한 조합 설정
    combinations = [
        {"name": "보수적 설정", "temp": 0.3, "top_k": 20, "top_p": 0.5},
        {"name": "균형잡힌 설정", "temp": 0.7, "top_k": 50, "top_p": 0.8},
        {"name": "창의적 설정", "temp": 1.0, "top_k": 100, "top_p": 0.95},
        {"name": "극도로 창의적", "temp": 1.5, "top_k": None, "top_p": 0.99}
    ]
    
    for combo in combinations:
        print(f"\n--- {combo['name']} ---")
        print(f"Temperature: {combo['temp']}, Top-k: {combo['top_k']}, Top-p: {combo['top_p']}")
        
        if model_type == "Real GPT-2":
            try:
                set_seed(42)
                
                generate_kwargs = {
                    'max_length': 45,
                    'num_return_sequences': 1,
                    'temperature': combo['temp'],
                    'top_p': combo['top_p'],
                    'do_sample': True,
                    'pad_token_id': generator.tokenizer.eos_token_id
                }
                
                if combo['top_k']:
                    generate_kwargs['top_k'] = combo['top_k']
                
                outputs = generator(prompt, **generate_kwargs)
                generated_text = outputs[0]['generated_text']
                
            except Exception as e:
                print(f"생성 오류: {e}")
                generated_text = f"오류로 인한 모의 결과: {prompt} [{combo['name']}]"
        
        else:
            # 모의 생성
            generated_text = generator.mock_generate(
                prompt, 
                combo['temp'], 
                top_k=combo['top_k'] or 18, 
                top_p=combo['top_p'], 
                max_length=45
            )
        
        print(f"결과: {generated_text}")

def analyze_parameter_effects():
    """파라미터 효과 분석"""
    
    print("\n=== 파라미터 효과 종합 분석 ===")
    
    analysis = {
        "Temperature": {
            "역할": "생성의 랜덤성과 창의성 조절",
            "낮은 값 (0.1-0.3)": {
                "특징": "예측 가능, 보수적, 일관성 높음",
                "사용 사례": "사실적 정보, 공식 문서, 정확성 중요"
            },
            "중간 값 (0.5-0.8)": {
                "특징": "창의성과 일관성의 균형",
                "사용 사례": "일반적인 창작, 대화, 콘텐츠 생성"
            },
            "높은 값 (1.0+)": {
                "특징": "매우 창의적, 예상치 못한 결과",
                "사용 사례": "예술적 창작, 브레인스토밍, 실험적 텍스트"
            }
        },
        
        "Top-k": {
            "역할": "각 단계에서 고려할 토큰 수 제한",
            "낮은 값 (10-30)": {
                "특징": "안전하고 일관된 선택",
                "사용 사례": "정확성이 중요한 작업"
            },
            "중간 값 (40-80)": {
                "특징": "적당한 다양성과 품질",
                "사용 사례": "일반적인 텍스트 생성"
            },
            "높은 값 (100+)": {
                "특징": "높은 다양성과 창의성",
                "사용 사례": "창의적 글쓰기, 다양한 아이디어 생성"
            }
        },
        
        "Top-p": {
            "역할": "확률 기반 토큰 선택 제한",
            "낮은 값 (0.1-0.5)": {
                "특징": "고확률 토큰만 선택, 안정적",
                "사용 사례": "정확한 정보 전달, 요약"
            },
            "중간 값 (0.6-0.9)": {
                "특징": "품질과 다양성의 균형",
                "사용 사례": "대부분의 일반적 용도"
            },
            "높은 값 (0.9+)": {
                "특징": "최대 다양성, 예상치 못한 선택",
                "사용 사례": "실험적 창작, 혁신적 아이디어"
            }
        }
    }
    
    for param, info in analysis.items():
        print(f"\n{param}:")
        print(f"  역할: {info['역할']}")
        
        for setting, details in info.items():
            if setting != "역할":
                print(f"\n  {setting}:")
                print(f"    특징: {details['특징']}")
                print(f"    사용 사례: {details['사용 사례']}")

def practical_recommendations():
    """실용적 권장사항"""
    
    print("\n=== 실용적 파라미터 설정 권장사항 ===")
    
    recommendations = {
        "콘텐츠 작성": {
            "블로그 포스트": "temp=0.7, top_k=50, top_p=0.8",
            "소셜 미디어": "temp=0.8, top_k=40, top_p=0.9",
            "제품 설명": "temp=0.5, top_k=30, top_p=0.7"
        },
        "창작 활동": {
            "소설/시": "temp=1.0, top_k=80, top_p=0.95",
            "아이디어 브레인스토밍": "temp=1.2, top_k=100, top_p=0.98",
            "캐릭터 대화": "temp=0.8, top_k=60, top_p=0.85"
        },
        "비즈니스": {
            "이메일 초안": "temp=0.4, top_k=25, top_p=0.6",
            "보고서 요약": "temp=0.3, top_k=20, top_p=0.5",
            "제안서 아이디어": "temp=0.6, top_k=40, top_p=0.7"
        },
        "교육": {
            "예제 생성": "temp=0.5, top_k=35, top_p=0.7",
            "설명 텍스트": "temp=0.4, top_k=30, top_p=0.6",
            "문제 출제": "temp=0.6, top_k=45, top_p=0.8"
        }
    }
    
    for category, tasks in recommendations.items():
        print(f"\n{category}:")
        for task, params in tasks.items():
            print(f"  {task}: {params}")
    
    print("\n=== 파라미터 조정 팁 ===")
    tips = [
        "• 시작은 중간 값(temp=0.7, top_k=50, top_p=0.8)으로",
        "• 결과가 너무 반복적이면 temperature 증가",
        "• 결과가 너무 이상하면 temperature 감소",
        "• 다양성이 필요하면 top_k와 top_p 증가",
        "• 일관성이 필요하면 top_k와 top_p 감소",
        "• 여러 설정을 시도해보고 최적값 찾기",
        "• 도메인과 목적에 따라 설정 조정"
    ]
    
    for tip in tips:
        print(tip)

def main():
    print("=== 문제 3.2: 생성 파라미터 실험 ===")
    
    # 1. Temperature 실험
    experiment_temperature()
    
    # 2. Top-k 실험
    experiment_top_k()
    
    # 3. Top-p 실험
    experiment_top_p()
    
    # 4. 파라미터 조합 실험
    combined_parameter_experiment()
    
    # 5. 파라미터 효과 분석
    analyze_parameter_effects()
    
    # 6. 실용적 권장사항
    practical_recommendations()
    
    # 설치 안내
    if not TRANSFORMERS_AVAILABLE:
        print("\n=== 설치 안내 ===")
        print("실제 GPT-2 모델을 사용하려면:")
        print("pip install transformers torch")

if __name__ == "__main__":
    main()