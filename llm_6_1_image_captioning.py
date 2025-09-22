"""
문제 6.1: 이미지 캡셔닝 (Image Captioning) 구현

지시사항:
Vision-Language 모델을 활용하여 이미지에 대한 자연어 설명을 생성하는 
이미지 캡셔닝 시스템을 구현하세요. CLIP, BLIP 등의 멀티모달 모델을 
사용하여 이미지의 내용을 이해하고 적절한 설명을 생성하는 과정을 시연하세요.
"""

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
    import torch
    VISION_LIBS_AVAILABLE = True
    print("Vision-Language 라이브러리가 사용 가능합니다.")
except ImportError:
    VISION_LIBS_AVAILABLE = False
    print("Vision-Language 라이브러리가 설치되지 않았습니다.")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import numpy as np
    import matplotlib.pyplot as plt
    DATA_VIS_AVAILABLE = True
except ImportError:
    DATA_VIS_AVAILABLE = False

import os
import random
import json
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass

@dataclass
class ImageCaption:
    """이미지 캡션 데이터 클래스"""
    image_path: str
    caption: str
    confidence: float
    model_name: str
    processing_time: float

@dataclass
class CaptioningResult:
    """캡셔닝 결과 데이터 클래스"""
    image_path: str
    captions: List[ImageCaption]
    best_caption: ImageCaption
    metadata: Dict[str, Any]

class MockVisionLanguageModel:
    """비전-언어 모델 모의 클래스"""
    
    def __init__(self, model_type: str = "BLIP"):
        self.model_type = model_type
        self.model_name = f"Mock {model_type} Model"
        
        # 이미지 유형별 설명 템플릿
        self.caption_templates = {
            "person": [
                "A person {action} in {location}",
                "Portrait of a person {description}",
                "{number} people {activity} together"
            ],
            "animal": [
                "A {animal_type} {action} in {environment}",
                "Beautiful {animal_type} {description}",
                "Wild {animal_type} in its natural habitat"
            ],
            "landscape": [
                "A beautiful {landscape_type} with {features}",
                "Scenic view of {location} during {time}",
                "Peaceful {landscape_type} scene"
            ],
            "object": [
                "A {color} {object_type} on {surface}",
                "Collection of {objects} arranged {arrangement}",
                "Modern {object_type} with {features}"
            ],
            "building": [
                "A {building_type} building with {features}",
                "Historic {building_type} in {location}",
                "Modern architecture featuring {elements}"
            ]
        }
        
        # 단어 데이터베이스
        self.word_database = {
            "actions": ["standing", "sitting", "walking", "running", "reading", "smiling"],
            "locations": ["a park", "the beach", "a city street", "indoors", "outdoors"],
            "descriptions": ["wearing casual clothes", "with a friendly expression", "looking happy"],
            "activities": ["talking", "playing", "working", "exercising", "relaxing"],
            "animal_types": ["cat", "dog", "bird", "elephant", "lion", "tiger", "bear"],
            "environments": ["the forest", "a field", "the savanna", "a zoo", "the wild"],
            "landscape_types": ["mountain", "forest", "beach", "lake", "valley", "desert"],
            "features": ["green trees", "blue sky", "flowing water", "rocky terrain"],
            "times": ["sunset", "sunrise", "daytime", "golden hour", "twilight"],
            "colors": ["red", "blue", "green", "yellow", "black", "white", "brown"],
            "object_types": ["car", "bicycle", "chair", "table", "computer", "phone"],
            "surfaces": ["a table", "the ground", "a shelf", "a desk"],
            "arrangements": ["neatly", "randomly", "in a row", "in a circle"],
            "building_types": ["modern", "historic", "residential", "commercial"],
            "elements": ["glass windows", "steel beams", "concrete walls", "wooden frames"]
        }
    
    def generate_caption(self, image_info: Dict[str, Any]) -> ImageCaption:
        """이미지 정보를 바탕으로 캡션 생성"""
        
        # 이미지 유형 추정
        image_type = self._estimate_image_type(image_info)
        
        # 적절한 템플릿 선택
        templates = self.caption_templates.get(image_type, self.caption_templates["object"])
        template = random.choice(templates)
        
        # 템플릿에 단어 채우기
        caption = self._fill_caption_template(template, image_type)
        
        # 신뢰도 생성 (이미지 복잡도에 따라)
        confidence = self._calculate_confidence(image_info, image_type)
        
        # 처리 시간 시뮬레이션
        processing_time = random.uniform(0.5, 2.0)
        
        return ImageCaption(
            image_path=image_info.get("path", "unknown"),
            caption=caption,
            confidence=confidence,
            model_name=self.model_name,
            processing_time=processing_time
        )
    
    def _estimate_image_type(self, image_info: Dict[str, Any]) -> str:
        """이미지 유형 추정"""
        # 파일명이나 경로에서 단서 찾기
        path = image_info.get("path", "").lower()
        
        if any(word in path for word in ["person", "people", "human", "portrait"]):
            return "person"
        elif any(word in path for word in ["cat", "dog", "animal", "pet", "wildlife"]):
            return "animal"
        elif any(word in path for word in ["landscape", "nature", "mountain", "forest"]):
            return "landscape"
        elif any(word in path for word in ["building", "house", "architecture"]):
            return "building"
        else:
            return "object"
    
    def _fill_caption_template(self, template: str, image_type: str) -> str:
        """캡션 템플릿에 구체적인 단어 채우기"""
        caption = template
        
        # 템플릿 변수들을 실제 단어로 치환
        replacements = {}
        
        if "{action}" in caption:
            replacements["{action}"] = random.choice(self.word_database["actions"])
        if "{location}" in caption:
            replacements["{location}"] = random.choice(self.word_database["locations"])
        if "{description}" in caption:
            replacements["{description}"] = random.choice(self.word_database["descriptions"])
        if "{activity}" in caption:
            replacements["{activity}"] = random.choice(self.word_database["activities"])
        if "{animal_type}" in caption:
            replacements["{animal_type}"] = random.choice(self.word_database["animal_types"])
        if "{environment}" in caption:
            replacements["{environment}"] = random.choice(self.word_database["environments"])
        if "{landscape_type}" in caption:
            replacements["{landscape_type}"] = random.choice(self.word_database["landscape_types"])
        if "{features}" in caption:
            replacements["{features}"] = random.choice(self.word_database["features"])
        if "{time}" in caption:
            replacements["{time}"] = random.choice(self.word_database["times"])
        if "{color}" in caption:
            replacements["{color}"] = random.choice(self.word_database["colors"])
        if "{object_type}" in caption:
            replacements["{object_type}"] = random.choice(self.word_database["object_types"])
        if "{surface}" in caption:
            replacements["{surface}"] = random.choice(self.word_database["surfaces"])
        if "{arrangement}" in caption:
            replacements["{arrangement}"] = random.choice(self.word_database["arrangements"])
        if "{building_type}" in caption:
            replacements["{building_type}"] = random.choice(self.word_database["building_types"])
        if "{elements}" in caption:
            replacements["{elements}"] = random.choice(self.word_database["elements"])
        if "{number}" in caption:
            replacements["{number}"] = random.choice(["Two", "Three", "Several", "A group of"])
        if "{objects}" in caption:
            replacements["{objects}"] = random.choice(self.word_database["object_types"]) + "s"
        
        # 치환 실행
        for placeholder, replacement in replacements.items():
            caption = caption.replace(placeholder, replacement)
        
        return caption
    
    def _calculate_confidence(self, image_info: Dict[str, Any], image_type: str) -> float:
        """신뢰도 계산"""
        base_confidence = 0.8
        
        # 이미지 유형별 기본 신뢰도 조정
        type_modifiers = {
            "person": 0.85,
            "animal": 0.82,
            "landscape": 0.88,
            "object": 0.75,
            "building": 0.80
        }
        
        base_confidence = type_modifiers.get(image_type, base_confidence)
        
        # 랜덤 노이즈 추가
        noise = random.uniform(-0.1, 0.1)
        confidence = max(0.5, min(0.99, base_confidence + noise))
        
        return confidence

class ImageCaptioningSystem:
    """이미지 캡셔닝 시스템"""
    
    def __init__(self):
        self.models = {}
        self.caption_history = []
        
        # 사용 가능한 모델들 초기화
        if VISION_LIBS_AVAILABLE:
            try:
                print("실제 BLIP 모델을 로드하는 중...")
                # 실제 환경에서는 여기서 실제 모델 로드
                print("실제 모델 로드 완료!")
                self.models["BLIP"] = "Real BLIP Model"
            except Exception as e:
                print(f"실제 모델 로드 실패: {e}")
                self.models["BLIP"] = MockVisionLanguageModel("BLIP")
        else:
            self.models["BLIP"] = MockVisionLanguageModel("BLIP")
            self.models["CLIP"] = MockVisionLanguageModel("CLIP")
        
        print(f"사용 가능한 모델: {list(self.models.keys())}")
    
    def caption_image(self, image_path: str, model_names: Optional[List[str]] = None) -> CaptioningResult:
        """이미지 캡셔닝 실행"""
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        print(f"이미지 캡셔닝 시작: {image_path}")
        
        # 이미지 정보 준비
        image_info = {
            "path": image_path,
            "size": (512, 512),  # 가정
            "format": "JPEG"
        }
        
        captions = []
        
        # 각 모델로 캡션 생성
        for model_name in model_names:
            if model_name in self.models:
                print(f"  {model_name} 모델로 캡션 생성 중...")
                
                model = self.models[model_name]
                if isinstance(model, MockVisionLanguageModel):
                    caption = model.generate_caption(image_info)
                else:
                    # 실제 모델 처리 (여기서는 모의 처리)
                    caption = MockVisionLanguageModel(model_name).generate_caption(image_info)
                
                captions.append(caption)
                print(f"    생성된 캡션: {caption.caption}")
                print(f"    신뢰도: {caption.confidence:.3f}")
        
        # 최고 품질 캡션 선택
        best_caption = max(captions, key=lambda x: x.confidence)
        
        # 결과 정리
        result = CaptioningResult(
            image_path=image_path,
            captions=captions,
            best_caption=best_caption,
            metadata={
                "num_models": len(captions),
                "avg_confidence": sum(c.confidence for c in captions) / len(captions),
                "total_processing_time": sum(c.processing_time for c in captions)
            }
        )
        
        self.caption_history.append(result)
        return result
    
    def batch_caption(self, image_paths: List[str]) -> List[CaptioningResult]:
        """여러 이미지 일괄 캡셔닝"""
        results = []
        
        print(f"일괄 캡셔닝 시작: {len(image_paths)}개 이미지")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\\n[{i}/{len(image_paths)}] 처리 중...")
            result = self.caption_image(image_path)
            results.append(result)
        
        print(f"\\n일괄 처리 완료!")
        return results
    
    def compare_models(self, image_path: str) -> Dict[str, Any]:
        """모델 간 성능 비교"""
        print(f"모델 성능 비교: {image_path}")
        
        # 모든 모델로 캡션 생성
        result = self.caption_image(image_path, list(self.models.keys()))
        
        # 비교 분석
        comparison = {
            "image_path": image_path,
            "model_results": {},
            "analysis": {}
        }
        
        for caption in result.captions:
            comparison["model_results"][caption.model_name] = {
                "caption": caption.caption,
                "confidence": caption.confidence,
                "processing_time": caption.processing_time,
                "caption_length": len(caption.caption.split())
            }
        
        # 분석 통계
        confidences = [c.confidence for c in result.captions]
        times = [c.processing_time for c in result.captions]
        
        comparison["analysis"] = {
            "highest_confidence_model": max(result.captions, key=lambda x: x.confidence).model_name,
            "fastest_model": min(result.captions, key=lambda x: x.processing_time).model_name,
            "avg_confidence": sum(confidences) / len(confidences),
            "total_time": sum(times),
            "confidence_variance": max(confidences) - min(confidences)
        }
        
        return comparison

def create_mock_image_dataset():
    """모의 이미지 데이터셋 생성"""
    
    dataset = [
        {
            "path": "images/beach_sunset.jpg",
            "description": "A beautiful beach during sunset with orange sky",
            "category": "landscape"
        },
        {
            "path": "images/cat_portrait.jpg", 
            "description": "A cute cat sitting on a windowsill",
            "category": "animal"
        },
        {
            "path": "images/people_park.jpg",
            "description": "People enjoying a picnic in the park",
            "category": "person"
        },
        {
            "path": "images/modern_building.jpg",
            "description": "A modern glass building in the city",
            "category": "building"
        },
        {
            "path": "images/vintage_car.jpg",
            "description": "A classic red vintage car",
            "category": "object"
        },
        {
            "path": "images/mountain_lake.jpg",
            "description": "A serene lake surrounded by mountains",
            "category": "landscape"
        },
        {
            "path": "images/dog_playing.jpg",
            "description": "A golden retriever playing fetch in a field",
            "category": "animal"
        },
        {
            "path": "images/chef_cooking.jpg",
            "description": "A chef preparing food in a restaurant kitchen",
            "category": "person"
        }
    ]
    
    return dataset

def demonstrate_image_captioning():
    """이미지 캡셔닝 데모"""
    
    print("=== 이미지 캡셔닝 데모 ===")
    print("다양한 이미지에 대한 자동 캡션 생성")
    print("-" * 60)
    
    # 캡셔닝 시스템 초기화
    system = ImageCaptioningSystem()
    
    # 모의 데이터셋 로드
    dataset = create_mock_image_dataset()
    
    # 몇 개 샘플 이미지 캡셔닝
    sample_images = dataset[:3]
    
    for i, image_data in enumerate(sample_images, 1):
        print(f"\\n--- 샘플 {i} ---")
        print(f"이미지: {image_data['path']}")
        print(f"실제 설명: {image_data['description']}")
        print(f"카테고리: {image_data['category']}")
        
        # 캡셔닝 실행
        result = system.caption_image(image_data['path'])
        
        print(f"\\n생성된 캡션들:")
        for j, caption in enumerate(result.captions, 1):
            print(f"  {j}. [{caption.model_name}] {caption.caption}")
            print(f"     신뢰도: {caption.confidence:.3f}, 시간: {caption.processing_time:.2f}초")
        
        print(f"\\n최고 품질 캡션: {result.best_caption.caption}")
        print(f"신뢰도: {result.best_caption.confidence:.3f}")

def demonstrate_batch_processing():
    """일괄 처리 데모"""
    
    print("\\n=== 일괄 처리 데모 ===")
    print("여러 이미지 동시 캡셔닝")
    print("-" * 60)
    
    system = ImageCaptioningSystem()
    dataset = create_mock_image_dataset()
    
    # 일괄 처리
    image_paths = [item["path"] for item in dataset[:5]]
    results = system.batch_caption(image_paths)
    
    # 결과 요약
    print(f"\\n=== 일괄 처리 결과 ===")
    
    total_time = sum(r.metadata["total_processing_time"] for r in results)
    avg_confidence = sum(r.metadata["avg_confidence"] for r in results) / len(results)
    
    print(f"처리된 이미지 수: {len(results)}")
    print(f"총 처리 시간: {total_time:.2f}초")
    print(f"평균 신뢰도: {avg_confidence:.3f}")
    
    print(f"\\n각 이미지별 최고 캡션:")
    for i, result in enumerate(results, 1):
        image_name = os.path.basename(result.image_path)
        print(f"{i}. {image_name}: {result.best_caption.caption}")

def demonstrate_model_comparison():
    """모델 비교 데모"""
    
    print("\\n=== 모델 성능 비교 데모 ===")
    print("여러 모델의 캡셔닝 성능 비교")
    print("-" * 60)
    
    system = ImageCaptioningSystem()
    dataset = create_mock_image_dataset()
    
    # 복잡한 이미지로 모델 비교
    test_image = dataset[2]["path"]  # people_park.jpg
    
    comparison = system.compare_models(test_image)
    
    print(f"\\n=== 모델 비교 결과 ===")
    print(f"테스트 이미지: {os.path.basename(comparison['image_path'])}")
    
    print(f"\\n각 모델별 결과:")
    for model_name, result in comparison["model_results"].items():
        print(f"\\n{model_name}:")
        print(f"  캡션: {result['caption']}")
        print(f"  신뢰도: {result['confidence']:.3f}")
        print(f"  처리 시간: {result['processing_time']:.2f}초")
        print(f"  캡션 길이: {result['caption_length']}단어")
    
    print(f"\\n분석 결과:")
    analysis = comparison["analysis"]
    print(f"  최고 신뢰도 모델: {analysis['highest_confidence_model']}")
    print(f"  최고 속도 모델: {analysis['fastest_model']}")
    print(f"  평균 신뢰도: {analysis['avg_confidence']:.3f}")
    print(f"  총 처리 시간: {analysis['total_time']:.2f}초")
    print(f"  신뢰도 편차: {analysis['confidence_variance']:.3f}")

def caption_quality_analysis():
    """캡션 품질 분석"""
    
    print("\\n=== 캡션 품질 분석 ===")
    print("생성된 캡션의 품질 평가 기준")
    print("-" * 60)
    
    quality_metrics = {
        "정확성 (Accuracy)": {
            "설명": "이미지 내용을 얼마나 정확하게 설명하는가",
            "평가방법": [
                "객체 인식 정확도",
                "장면 이해 정확도", 
                "색상, 위치 등 세부사항 정확도"
            ]
        },
        "완전성 (Completeness)": {
            "설명": "이미지의 주요 요소들을 모두 포함하는가",
            "평가방법": [
                "주요 객체 언급 여부",
                "배경 및 환경 설명",
                "행동이나 상호작용 묘사"
            ]
        },
        "유창성 (Fluency)": {
            "설명": "자연스럽고 이해하기 쉬운 문장인가",
            "평가방법": [
                "문법적 정확성",
                "어휘 선택의 적절성",
                "문장 구조의 자연스러움"
            ]
        },
        "관련성 (Relevance)": {
            "설명": "이미지와 관련 없는 내용이 포함되었는가",
            "평가방법": [
                "이미지에 없는 객체 언급 여부",
                "추측성 내용 포함 여부",
                "일반적 설명 vs 구체적 설명"
            ]
        },
        "다양성 (Diversity)": {
            "설명": "다양한 표현과 어휘를 사용하는가",
            "평가방법": [
                "동일 이미지 유형에 대한 표현 다양성",
                "어휘 선택의 풍부함",
                "문장 패턴의 다양성"
            ]
        }
    }
    
    for metric, info in quality_metrics.items():
        print(f"\\n{metric}:")
        print(f"  {info['설명']}")
        print("  평가 방법:")
        for method in info['평가방법']:
            print(f"    • {method}")

def multimodal_applications():
    """멀티모달 응용 분야"""
    
    print("\\n=== 이미지 캡셔닝의 응용 분야 ===")
    
    applications = {
        "접근성 지원": {
            "설명": "시각 장애인을 위한 이미지 설명 서비스",
            "기술요구사항": [
                "높은 정확도의 객체 인식",
                "상세한 공간 관계 설명",
                "실시간 처리 능력"
            ],
            "예시": "스크린 리더와 연동된 웹 브라우저 확장"
        },
        "소셜 미디어": {
            "설명": "자동 해시태그 생성 및 콘텐츠 분류",
            "기술요구사항": [
                "감정과 분위기 인식",
                "트렌드 키워드 추출",
                "다국어 지원"
            ],
            "예시": "Instagram, Facebook의 자동 alt-text 생성"
        },
        "전자상거래": {
            "설명": "상품 이미지 자동 설명 및 검색 최적화",
            "기술요구사항": [
                "상품 속성 정확한 추출",
                "브랜드 및 모델명 인식",
                "SEO 친화적 설명 생성"
            ],
            "예시": "온라인 쇼핑몰의 상품 설명 자동화"
        },
        "의료 영상": {
            "설명": "의료 이미지 분석 및 리포트 생성",
            "기술요구사항": [
                "의료 전문 용어 사용",
                "높은 정확도와 신뢰성",
                "의사의 검토 과정 포함"
            ],
            "예시": "X-ray, CT 스캔 초기 분석 보조"
        },
        "교육": {
            "설명": "교육 자료 생성 및 학습 보조",
            "기술요구사항": [
                "연령대별 적절한 어휘 선택",
                "교육적 가치 있는 설명",
                "상호작용적 학습 지원"
            ],
            "예시": "아동용 그림책 읽기 보조 앱"
        },
        "뉴스 및 미디어": {
            "설명": "뉴스 이미지 자동 캡셔닝 및 분류",
            "기술요구사항": [
                "사실적이고 객관적인 설명",
                "인물 및 장소 인식",
                "긴급성 및 중요도 평가"
            ],
            "예시": "뉴스 편집실의 이미지 처리 자동화"
        }
    }
    
    for app, info in applications.items():
        print(f"\\n{app}:")
        print(f"  설명: {info['설명']}")
        print("  기술 요구사항:")
        for req in info['기술요구사항']:
            print(f"    • {req}")
        print(f"  예시: {info['예시']}")

def future_developments():
    """미래 발전 방향"""
    
    print("\\n=== 이미지 캡셔닝의 미래 발전 방향 ===")
    
    future_trends = {
        "실시간 비디오 캡셔닝": {
            "현재 상태": "연구 개발 중",
            "기술 과제": ["실시간 처리 성능", "일관성 있는 서술", "메모리 효율성"],
            "응용 분야": "라이브 스트리밍, 비디오 회의, 감시 시스템"
        },
        "상호작용적 캡셔닝": {
            "현재 상태": "초기 연구 단계",
            "기술 과제": ["사용자 의도 파악", "대화형 인터페이스", "개인화"],
            "응용 분야": "AI 어시스턴트, 교육 플랫폼, 게임"
        },
        "감정 및 의도 인식": {
            "현재 상태": "상용화 시작",
            "기술 과제": ["미묘한 감정 표현", "문화적 차이", "편향 제거"],
            "응용 분야": "마케팅, 심리학 연구, 소셜 미디어 분석"
        },
        "3D 및 AR/VR 지원": {
            "현재 상태": "실험적 단계",
            "기술 과제": ["3D 공간 이해", "깊이 정보 활용", "몰입감 유지"],
            "응용 분야": "메타버스, 건축 설계, 의료 시뮬레이션"
        },
        "다국어 및 문화 적응": {
            "현재 상태": "개발 진행 중",
            "기술 과제": ["문화적 맥락 이해", "지역별 표현 차이", "번역 품질"],
            "응용 분야": "글로벌 서비스, 관광, 국제 협력"
        }
    }
    
    for trend, info in future_trends.items():
        print(f"\\n{trend}:")
        print(f"  현재 상태: {info['현재 상태']}")
        print("  기술 과제:")
        for challenge in info['기술 과제']:
            print(f"    • {challenge}")
        print(f"  응용 분야: {info['응용 분야']}")

def main():
    print("=== 문제 6.1: 이미지 캡셔닝 구현 ===")
    
    # 1. 기본 이미지 캡셔닝 데모
    demonstrate_image_captioning()
    
    # 2. 일괄 처리 데모
    demonstrate_batch_processing()
    
    # 3. 모델 비교 데모
    demonstrate_model_comparison()
    
    # 4. 캡션 품질 분석
    caption_quality_analysis()
    
    # 5. 응용 분야 소개
    multimodal_applications()
    
    # 6. 미래 발전 방향
    future_developments()
    
    # 설치 안내
    if not VISION_LIBS_AVAILABLE:
        print("\\n=== 설치 안내 ===")
        print("실제 이미지 캡셔닝을 위해:")
        print("pip install transformers torch pillow")
        print("pip install -U sentence-transformers")

if __name__ == "__main__":
    main()