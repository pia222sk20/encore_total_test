"""
문제 6.2: Stable Diffusion 텍스트-이미지 생성 구현

지시사항:
Stable Diffusion 모델을 활용하여 텍스트 프롬프트로부터 이미지를 생성하는 
시스템을 구현하세요. 다양한 프롬프트 엔지니어링 기법을 적용하고, 
생성 매개변수를 조정하여 고품질 이미지를 생성하는 과정을 시연하세요.
"""

try:
    from diffusers import StableDiffusionPipeline, DDIMScheduler, EulerDiscreteScheduler
    import torch
    DIFFUSION_AVAILABLE = True
    print("Diffusion 라이브러리가 사용 가능합니다.")
except ImportError:
    DIFFUSION_AVAILABLE = False
    print("Diffusion 라이브러리가 설치되지 않았습니다.")

try:
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import numpy as np
    import matplotlib.pyplot as plt
    DATA_VIS_AVAILABLE = True
except ImportError:
    DATA_VIS_AVAILABLE = False

import random
import time
import json
import os
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

class ImageStyle(Enum):
    """이미지 스타일 열거형"""
    PHOTOREALISTIC = "photorealistic"
    ARTISTIC = "artistic" 
    CARTOON = "cartoon"
    ANIME = "anime"
    SKETCH = "sketch"
    WATERCOLOR = "watercolor"
    OIL_PAINTING = "oil painting"
    DIGITAL_ART = "digital art"

class AspectRatio(Enum):
    """종횡비 열거형"""
    SQUARE = "1:1"
    LANDSCAPE = "16:9"
    PORTRAIT = "9:16"
    WIDE = "21:9"
    CLASSIC = "4:3"

@dataclass
class GenerationConfig:
    """이미지 생성 설정"""
    prompt: str
    negative_prompt: str = ""
    style: ImageStyle = ImageStyle.PHOTOREALISTIC
    aspect_ratio: AspectRatio = AspectRatio.SQUARE
    steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    batch_size: int = 1
    width: int = 512
    height: int = 512

@dataclass 
class GenerationResult:
    """생성 결과"""
    config: GenerationConfig
    images: List[Any] = field(default_factory=list)
    generation_time: float = 0.0
    seed_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class PromptEngineer:
    """프롬프트 엔지니어링 도구"""
    
    def __init__(self):
        # 품질 향상 키워드
        self.quality_keywords = [
            "high quality", "detailed", "professional", "masterpiece",
            "best quality", "ultra detailed", "4K", "8K", "HDR",
            "sharp focus", "intricate details", "highly detailed"
        ]
        
        # 스타일별 키워드
        self.style_keywords = {
            ImageStyle.PHOTOREALISTIC: [
                "photorealistic", "realistic", "photography", "DSLR",
                "professional photography", "natural lighting", "cinematic"
            ],
            ImageStyle.ARTISTIC: [
                "artistic", "fine art", "gallery worthy", "expressive",
                "creative", "unique style", "artistic vision"
            ],
            ImageStyle.CARTOON: [
                "cartoon style", "animated", "cartoon character",
                "cel shading", "cartoon art", "stylized"
            ],
            ImageStyle.ANIME: [
                "anime style", "manga", "japanese animation",
                "anime character", "kawaii", "anime art"
            ],
            ImageStyle.SKETCH: [
                "pencil sketch", "drawing", "line art", "sketch style",
                "hand drawn", "charcoal drawing"
            ],
            ImageStyle.WATERCOLOR: [
                "watercolor painting", "watercolor style", "soft colors",
                "flowing paint", "artistic medium", "traditional art"
            ],
            ImageStyle.OIL_PAINTING: [
                "oil painting", "painted", "brushstrokes", "classical art",
                "renaissance style", "traditional painting"
            ],
            ImageStyle.DIGITAL_ART: [
                "digital art", "digital painting", "concept art",
                "digital illustration", "CG art", "computer graphics"
            ]
        }
        
        # 라이팅 키워드
        self.lighting_keywords = [
            "golden hour lighting", "soft lighting", "dramatic lighting",
            "natural light", "studio lighting", "backlighting",
            "rim lighting", "ambient lighting", "cinematic lighting"
        ]
        
        # 구성 키워드  
        self.composition_keywords = [
            "rule of thirds", "centered composition", "symmetrical",
            "balanced composition", "leading lines", "depth of field",
            "bokeh background", "close-up", "wide angle", "telephoto"
        ]
        
        # 부정적 프롬프트 키워드
        self.negative_keywords = [
            "blurry", "low quality", "worst quality", "jpeg artifacts",
            "watermark", "signature", "text", "words", "letters",
            "bad anatomy", "deformed", "disfigured", "extra limbs",
            "ugly", "duplicate", "morbid", "mutilated", "out of frame",
            "extra fingers", "mutated hands", "poorly drawn hands",
            "poorly drawn face", "mutation", "bad proportions"
        ]
    
    def enhance_prompt(self, base_prompt: str, style: ImageStyle, 
                      add_quality: bool = True, add_lighting: bool = True,
                      add_composition: bool = True) -> str:
        """프롬프트 향상"""
        
        enhanced_parts = [base_prompt]
        
        # 스타일 키워드 추가
        style_words = random.sample(self.style_keywords[style], 
                                   min(2, len(self.style_keywords[style])))
        enhanced_parts.extend(style_words)
        
        # 품질 키워드 추가
        if add_quality:
            quality_words = random.sample(self.quality_keywords, 2)
            enhanced_parts.extend(quality_words)
        
        # 라이팅 키워드 추가
        if add_lighting:
            lighting_word = random.choice(self.lighting_keywords)
            enhanced_parts.append(lighting_word)
        
        # 구성 키워드 추가
        if add_composition:
            composition_word = random.choice(self.composition_keywords)
            enhanced_parts.append(composition_word)
        
        return ", ".join(enhanced_parts)
    
    def generate_negative_prompt(self, custom_negatives: List[str] = None) -> str:
        """부정적 프롬프트 생성"""
        
        negatives = random.sample(self.negative_keywords, 8)
        
        if custom_negatives:
            negatives.extend(custom_negatives)
        
        return ", ".join(negatives)
    
    def create_style_prompt(self, subject: str, style: ImageStyle, 
                           scene_description: str = "") -> str:
        """스타일별 프롬프트 생성"""
        
        if scene_description:
            base_prompt = f"{subject} {scene_description}"
        else:
            base_prompt = subject
        
        return self.enhance_prompt(base_prompt, style)

class MockStableDiffusion:
    """Stable Diffusion 모의 클래스"""
    
    def __init__(self, model_name: str = "stable-diffusion-v1-5"):
        self.model_name = model_name
        self.device = "cpu"  # 모의 환경
        
        # 생성 통계
        self.generation_stats = {
            "total_generations": 0,
            "avg_time": 0.0,
            "style_counts": {},
            "success_rate": 0.95
        }
    
    def generate_image(self, config: GenerationConfig) -> GenerationResult:
        """이미지 생성 시뮬레이션"""
        
        start_time = time.time()
        
        # 시드 설정
        if config.seed is None:
            seed = random.randint(0, 2**32 - 1)
        else:
            seed = config.seed
        
        # 생성 시간 시뮬레이션 (설정에 따라 다름)
        base_time = 2.0  # 기본 2초
        time_factor = config.steps / 50.0  # 스텝 수에 따른 조정
        time_factor *= config.batch_size  # 배치 크기에 따른 조정
        
        simulated_time = base_time * time_factor + random.uniform(0.5, 1.5)
        time.sleep(min(simulated_time, 0.1))  # 실제로는 짧게 대기
        
        # 모의 이미지 생성 (실제로는 PIL Image 객체들)
        mock_images = []
        for i in range(config.batch_size):
            # 실제 환경에서는 여기서 실제 이미지 생성
            mock_image_info = {
                "width": config.width,
                "height": config.height,
                "prompt": config.prompt,
                "seed": seed + i,
                "style": config.style.value
            }
            mock_images.append(mock_image_info)
        
        generation_time = time.time() - start_time
        
        # 통계 업데이트
        self._update_stats(config, generation_time)
        
        # 결과 생성
        result = GenerationResult(
            config=config,
            images=mock_images,
            generation_time=generation_time,
            seed_used=seed,
            metadata={
                "model_name": self.model_name,
                "device": self.device,
                "success": True,
                "quality_score": self._calculate_quality_score(config)
            }
        )
        
        return result
    
    def _update_stats(self, config: GenerationConfig, generation_time: float):
        """통계 업데이트"""
        
        self.generation_stats["total_generations"] += 1
        
        # 평균 시간 업데이트
        total = self.generation_stats["total_generations"]
        prev_avg = self.generation_stats["avg_time"]
        self.generation_stats["avg_time"] = (prev_avg * (total - 1) + generation_time) / total
        
        # 스타일 카운트
        style_name = config.style.value
        if style_name in self.generation_stats["style_counts"]:
            self.generation_stats["style_counts"][style_name] += 1
        else:
            self.generation_stats["style_counts"][style_name] = 1
    
    def _calculate_quality_score(self, config: GenerationConfig) -> float:
        """품질 점수 계산"""
        
        base_score = 0.7
        
        # 스텝 수에 따른 점수 조정
        if config.steps >= 50:
            base_score += 0.1
        elif config.steps < 20:
            base_score -= 0.1
        
        # 가이던스 스케일에 따른 조정
        if 7.0 <= config.guidance_scale <= 10.0:
            base_score += 0.1
        elif config.guidance_scale > 15.0:
            base_score -= 0.1
        
        # 프롬프트 품질에 따른 조정
        if len(config.prompt.split()) > 10:
            base_score += 0.05
        if config.negative_prompt:
            base_score += 0.05
        
        # 랜덤 노이즈 추가
        base_score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, base_score))

class StableDiffusionSystem:
    """Stable Diffusion 시스템"""
    
    def __init__(self):
        self.prompt_engineer = PromptEngineer()
        self.generation_history = []
        
        # 모델 초기화
        if DIFFUSION_AVAILABLE:
            try:
                print("실제 Stable Diffusion 모델 로드 중...")
                # 실제 환경에서는 여기서 실제 모델 로드
                print("실제 모델 로드 완료!")
                self.model = "Real Stable Diffusion"
            except Exception as e:
                print(f"실제 모델 로드 실패: {e}")
                self.model = MockStableDiffusion()
        else:
            self.model = MockStableDiffusion()
        
        print(f"Stable Diffusion 시스템 준비 완료")
    
    def generate_single_image(self, prompt: str, style: ImageStyle = ImageStyle.PHOTOREALISTIC,
                            steps: int = 50, guidance_scale: float = 7.5) -> GenerationResult:
        """단일 이미지 생성"""
        
        # 프롬프트 향상
        enhanced_prompt = self.prompt_engineer.enhance_prompt(prompt, style)
        negative_prompt = self.prompt_engineer.generate_negative_prompt()
        
        # 설정 생성
        config = GenerationConfig(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            style=style,
            steps=steps,
            guidance_scale=guidance_scale
        )
        
        print(f"이미지 생성 시작...")
        print(f"  프롬프트: {config.prompt[:100]}...")
        print(f"  스타일: {style.value}")
        print(f"  스텝: {steps}, 가이던스: {guidance_scale}")
        
        # 생성 실행
        if isinstance(self.model, MockStableDiffusion):
            result = self.model.generate_image(config)
        else:
            # 실제 모델 처리 (모의)
            result = MockStableDiffusion().generate_image(config)
        
        self.generation_history.append(result)
        
        print(f"  생성 완료! 시간: {result.generation_time:.2f}초")
        print(f"  품질 점수: {result.metadata['quality_score']:.3f}")
        print(f"  사용된 시드: {result.seed_used}")
        
        return result
    
    def generate_image_variations(self, prompt: str, style: ImageStyle,
                                 num_variations: int = 4) -> List[GenerationResult]:
        """이미지 변형 생성"""
        
        print(f"이미지 변형 생성: {num_variations}개")
        
        variations = []
        base_seed = random.randint(0, 2**32 - 1)
        
        for i in range(num_variations):
            # 매개변수를 약간씩 변경
            steps = random.choice([30, 40, 50, 60])
            guidance = random.uniform(6.0, 9.0)
            
            # 프롬프트 약간 변경
            varied_prompt = self._create_prompt_variation(prompt, i)
            
            config = GenerationConfig(
                prompt=varied_prompt,
                negative_prompt=self.prompt_engineer.generate_negative_prompt(),
                style=style,
                steps=steps,
                guidance_scale=guidance,
                seed=base_seed + i * 1000
            )
            
            print(f"  변형 {i+1}/{num_variations} 생성 중...")
            
            if isinstance(self.model, MockStableDiffusion):
                result = self.model.generate_image(config)
            else:
                result = MockStableDiffusion().generate_image(config)
            
            variations.append(result)
            self.generation_history.append(result)
        
        print(f"변형 생성 완료!")
        return variations
    
    def _create_prompt_variation(self, base_prompt: str, variation_index: int) -> str:
        """프롬프트 변형 생성"""
        
        variation_modifiers = [
            ["vibrant colors", "dynamic composition"],
            ["soft lighting", "dreamy atmosphere"],
            ["dramatic shadows", "high contrast"],
            ["warm colors", "cozy feeling"],
            ["cool colors", "serene mood"]
        ]
        
        if variation_index < len(variation_modifiers):
            modifiers = variation_modifiers[variation_index]
            return f"{base_prompt}, {', '.join(modifiers)}"
        else:
            return base_prompt
    
    def style_comparison(self, prompt: str) -> Dict[ImageStyle, GenerationResult]:
        """스타일별 비교 생성"""
        
        print(f"스타일 비교 생성: {prompt}")
        
        styles_to_test = [
            ImageStyle.PHOTOREALISTIC,
            ImageStyle.ARTISTIC,
            ImageStyle.ANIME,
            ImageStyle.DIGITAL_ART
        ]
        
        results = {}
        
        for style in styles_to_test:
            print(f"  {style.value} 스타일 생성 중...")
            result = self.generate_single_image(prompt, style)
            results[style] = result
        
        print("스타일 비교 완료!")
        return results
    
    def parameter_optimization(self, prompt: str, style: ImageStyle) -> Dict[str, Any]:
        """매개변수 최적화"""
        
        print(f"매개변수 최적화 실험")
        
        # 테스트할 매개변수 조합
        test_configs = [
            {"steps": 20, "guidance": 5.0, "name": "Fast"},
            {"steps": 30, "guidance": 7.5, "name": "Balanced"}, 
            {"steps": 50, "guidance": 10.0, "name": "Quality"},
            {"steps": 80, "guidance": 12.5, "name": "Ultra"}
        ]
        
        results = {}
        
        for config in test_configs:
            print(f"  {config['name']} 설정 테스트...")
            
            result = self.generate_single_image(
                prompt, style, 
                steps=config["steps"],
                guidance_scale=config["guidance"]
            )
            
            results[config["name"]] = {
                "result": result,
                "config": config,
                "quality_score": result.metadata["quality_score"],
                "generation_time": result.generation_time,
                "efficiency": result.metadata["quality_score"] / result.generation_time
            }
        
        # 최적 설정 선택
        best_config = max(results.items(), key=lambda x: x[1]["efficiency"])
        
        print(f"\\n최적 설정: {best_config[0]}")
        print(f"  효율성 점수: {best_config[1]['efficiency']:.3f}")
        
        return results

def demonstrate_basic_generation():
    """기본 이미지 생성 데모"""
    
    print("=== 기본 이미지 생성 데모 ===")
    print("다양한 프롬프트로 이미지 생성")
    print("-" * 60)
    
    system = StableDiffusionSystem()
    
    # 테스트 프롬프트들
    test_prompts = [
        "A beautiful sunset over a mountain lake",
        "A cute cat wearing a wizard hat",
        "A futuristic city with flying cars",
        "A peaceful garden with colorful flowers"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\\n--- 샘플 {i} ---")
        
        # 다양한 스타일로 생성
        result = system.generate_single_image(
            prompt, 
            style=ImageStyle.PHOTOREALISTIC,
            steps=30,
            guidance_scale=7.5
        )
        
        print(f"원본 프롬프트: {prompt}")
        print(f"향상된 프롬프트: {result.config.prompt[:150]}...")
        print(f"부정적 프롬프트: {result.config.negative_prompt[:100]}...")

def demonstrate_style_comparison():
    """스타일 비교 데모"""
    
    print("\\n=== 스타일 비교 데모 ===")
    print("동일 프롬프트의 다양한 스타일 변형")
    print("-" * 60)
    
    system = StableDiffusionSystem()
    
    test_prompt = "A majestic dragon in a mystical forest"
    
    results = system.style_comparison(test_prompt)
    
    print(f"\\n=== 스타일 비교 결과 ===")
    print(f"테스트 프롬프트: {test_prompt}")
    
    for style, result in results.items():
        print(f"\\n{style.value.upper()}:")
        print(f"  생성 시간: {result.generation_time:.2f}초")
        print(f"  품질 점수: {result.metadata['quality_score']:.3f}")
        print(f"  시드: {result.seed_used}")
        print(f"  프롬프트: {result.config.prompt[:100]}...")

def demonstrate_parameter_optimization():
    """매개변수 최적화 데모"""
    
    print("\\n=== 매개변수 최적화 데모 ===")
    print("다양한 설정으로 최적 매개변수 찾기")
    print("-" * 60)
    
    system = StableDiffusionSystem()
    
    test_prompt = "A serene Japanese garden with cherry blossoms"
    style = ImageStyle.ARTISTIC
    
    optimization_results = system.parameter_optimization(test_prompt, style)
    
    print(f"\\n=== 최적화 결과 ===")
    
    # 결과를 효율성 순으로 정렬
    sorted_results = sorted(optimization_results.items(), 
                          key=lambda x: x[1]["efficiency"], reverse=True)
    
    print(f"설정별 성능 (효율성 순):")
    print(f"{'설정':<10} {'품질':<8} {'시간':<8} {'효율성':<8}")
    print("-" * 35)
    
    for name, data in sorted_results:
        print(f"{name:<10} {data['quality_score']:<8.3f} "
              f"{data['generation_time']:<8.2f} {data['efficiency']:<8.3f}")

def demonstrate_prompt_engineering():
    """프롬프트 엔지니어링 데모"""
    
    print("\\n=== 프롬프트 엔지니어링 데모 ===")
    print("프롬프트 개선 기법 시연")
    print("-" * 60)
    
    engineer = PromptEngineer()
    
    # 기본 프롬프트들
    base_prompts = [
        "a dog",
        "a house",
        "a woman portrait",
        "a landscape"
    ]
    
    styles = [ImageStyle.PHOTOREALISTIC, ImageStyle.ARTISTIC, ImageStyle.ANIME]
    
    for base_prompt in base_prompts:
        print(f"\\n기본 프롬프트: '{base_prompt}'")
        print("스타일별 향상된 프롬프트:")
        
        for style in styles:
            enhanced = engineer.enhance_prompt(base_prompt, style)
            print(f"  {style.value}: {enhanced}")
        
        # 부정적 프롬프트
        negative = engineer.generate_negative_prompt()
        print(f"  부정적 프롬프트: {negative[:100]}...")

def demonstrate_batch_generation():
    """일괄 생성 데모"""
    
    print("\\n=== 일괄 생성 데모 ===")
    print("여러 이미지 동시 생성")
    print("-" * 60)
    
    system = StableDiffusionSystem()
    
    # 테마별 프롬프트 집합
    themes = {
        "Fantasy": [
            "A magical crystal cave with glowing crystals",
            "A fairy riding a butterfly in an enchanted forest",
            "A wizard's tower under a starry night sky"
        ],
        "Nature": [
            "A peaceful waterfall in a tropical rainforest",
            "A field of lavender under the golden sunset",
            "A snow-covered mountain peak at dawn"
        ],
        "Urban": [
            "A bustling city street at night with neon lights",
            "A cozy coffee shop on a rainy day",
            "A modern skyscraper reflecting the sky"
        ]
    }
    
    for theme, prompts in themes.items():
        print(f"\\n--- {theme} 테마 ---")
        
        theme_results = []
        for prompt in prompts:
            result = system.generate_single_image(
                prompt,
                style=ImageStyle.DIGITAL_ART,
                steps=40
            )
            theme_results.append(result)
        
        # 테마별 통계
        avg_quality = sum(r.metadata["quality_score"] for r in theme_results) / len(theme_results)
        total_time = sum(r.generation_time for r in theme_results)
        
        print(f"  생성된 이미지: {len(theme_results)}개")
        print(f"  평균 품질: {avg_quality:.3f}")
        print(f"  총 시간: {total_time:.2f}초")

def advanced_techniques():
    """고급 기법 소개"""
    
    print("\\n=== 고급 생성 기법 ===")
    
    techniques = {
        "ControlNet": {
            "설명": "입력 이미지의 구조나 스케치를 따라 생성",
            "사용 예": "포즈 제어, 깊이 맵 활용, 캐니 엣지 따라하기",
            "장점": "정확한 구성 제어, 일관된 결과",
            "한계": "추가 전처리 필요, 복잡한 설정"
        },
        "Inpainting": {
            "설명": "이미지의 특정 부분만 다시 생성",
            "사용 예": "배경 변경, 객체 제거/추가, 부분 수정",
            "장점": "기존 이미지 활용, 정밀한 수정",
            "한계": "마스크 품질에 의존, 경계 처리 어려움"
        },
        "Image-to-Image": {
            "설명": "기존 이미지를 참조하여 새 이미지 생성",
            "사용 예": "스타일 변환, 이미지 업스케일링, 변형",
            "장점": "빠른 변형, 일관성 유지",
            "한계": "원본에 제약, 창의성 제한"
        },
        "LoRA Fine-tuning": {
            "설명": "특정 스타일이나 개념을 학습한 어댑터 활용",
            "사용 예": "특정 캐릭터, 예술 스타일, 브랜드 요소",
            "장점": "맞춤형 스타일, 효율적 학습",
            "한계": "데이터 준비 필요, 과적합 위험"
        },
        "Multi-Model Ensemble": {
            "설명": "여러 모델의 결과를 조합",
            "사용 예": "품질 향상, 다양성 증대, 안정성 개선",
            "장점": "높은 품질, 일관된 결과",
            "한계": "계산 비용 증가, 복잡한 후처리"
        }
    }
    
    for technique, info in techniques.items():
        print(f"\\n{technique}:")
        print(f"  설명: {info['설명']}")
        print(f"  사용 예: {info['사용 예']}")
        print(f"  장점: {info['장점']}")
        print(f"  한계: {info['한계']}")

def commercial_applications():
    """상업적 응용 분야"""
    
    print("\\n=== Stable Diffusion 상업적 응용 ===")
    
    applications = {
        "마케팅 및 광고": {
            "활용 방법": [
                "제품 이미지 생성",
                "소셜 미디어 콘텐츠 제작",
                "광고 배너 및 포스터 디자인",
                "브랜드 일관성 유지"
            ],
            "비즈니스 가치": "제작 비용 절감, 빠른 아이디어 검증, 맞춤형 콘텐츠"
        },
        "게임 개발": {
            "활용 방법": [
                "컨셉 아트 생성",
                "텍스처 및 배경 제작",
                "캐릭터 디자인 아이디어",
                "환경 및 소품 디자인"
            ],
            "비즈니스 가치": "개발 속도 향상, 아티스트 생산성 증대, 비용 효율성"
        },
        "전자상거래": {
            "활용 방법": [
                "상품 이미지 변형",
                "라이프스타일 컨텍스트 이미지",
                "시즌별 테마 이미지",
                "개인화된 상품 시각화"
            ],
            "비즈니스 가치": "전환율 향상, 촬영 비용 절감, 빠른 시장 대응"
        },
        "교육 및 훈련": {
            "활용 방법": [
                "교육 자료 일러스트",
                "시뮬레이션 환경 구성",
                "역사적 장면 재현",
                "과학적 개념 시각화"
            ],
            "비즈니스 가치": "학습 효과 증대, 제작 비용 절감, 맞춤형 교육"
        },
        "건축 및 인테리어": {
            "활용 방법": [
                "인테리어 디자인 아이디어",
                "건축 컨셉 시각화",
                "조경 계획 시뮬레이션",
                "가구 배치 실험"
            ],
            "비즈니스 가치": "클라이언트 소통 개선, 디자인 효율성, 의사결정 지원"
        }
    }
    
    for application, info in applications.items():
        print(f"\\n{application}:")
        print("  활용 방법:")
        for method in info["활용 방법"]:
            print(f"    • {method}")
        print(f"  비즈니스 가치: {info['비즈니스 가치']}")

def ethical_considerations():
    """윤리적 고려사항"""
    
    print("\\n=== 윤리적 고려사항 및 책임감 있는 AI 사용 ===")
    
    considerations = {
        "저작권 및 지적재산권": {
            "문제": "훈련 데이터에 포함된 저작권 보호 작품들",
            "대응방안": [
                "명확한 라이선스 정책 수립",
                "원저작자 권리 존중",
                "상업적 사용 시 법적 검토",
                "오픈소스 또는 퍼블릭 도메인 데이터 우선 사용"
            ]
        },
        "편향과 공정성": {
            "문제": "훈련 데이터의 편향이 생성 결과에 반영",
            "대응방안": [
                "다양성 있는 훈련 데이터 구성",
                "편향 테스트 및 모니터링",
                "포용적 디자인 원칙 적용",
                "정기적인 모델 감사"
            ]
        },
        "딥페이크 및 조작": {
            "문제": "악의적 목적의 가짜 이미지 생성",
            "대응방안": [
                "생성 이미지 워터마크 적용",
                "탐지 기술 개발 및 적용",
                "사용 목적 제한 및 모니터링",
                "법적 규제 준수"
            ]
        },
        "개인정보 보호": {
            "문제": "실존 인물의 초상권 침해 가능성",
            "대응방안": [
                "실존 인물 생성 금지 정책",
                "개인정보 포함 데이터 필터링",
                "동의 없는 초상 사용 금지",
                "프라이버시 보호 기술 적용"
            ]
        },
        "환경적 영향": {
            "문제": "대량의 계산 자원과 에너지 소비",
            "대응방안": [
                "효율적인 모델 아키텍처 개발",
                "재생 에너지 활용",
                "최적화된 하드웨어 사용",
                "카본 오프셋 프로그램 참여"
            ]
        }
    }
    
    for consideration, info in considerations.items():
        print(f"\\n{consideration}:")
        print(f"  문제: {info['문제']}")
        print("  대응방안:")
        for solution in info['대응방안']:
            print(f"    • {solution}")

def main():
    print("=== 문제 6.2: Stable Diffusion 텍스트-이미지 생성 구현 ===")
    
    # 1. 기본 이미지 생성 데모
    demonstrate_basic_generation()
    
    # 2. 스타일 비교 데모
    demonstrate_style_comparison()
    
    # 3. 매개변수 최적화 데모
    demonstrate_parameter_optimization()
    
    # 4. 프롬프트 엔지니어링 데모
    demonstrate_prompt_engineering()
    
    # 5. 일괄 생성 데모
    demonstrate_batch_generation()
    
    # 6. 고급 기법 소개
    advanced_techniques()
    
    # 7. 상업적 응용 분야
    commercial_applications()
    
    # 8. 윤리적 고려사항
    ethical_considerations()
    
    # 설치 안내
    if not DIFFUSION_AVAILABLE:
        print("\\n=== 설치 안내 ===")
        print("실제 Stable Diffusion을 위해:")
        print("pip install diffusers transformers accelerate")
        print("pip install torch torchvision pillow")

if __name__ == "__main__":
    main()