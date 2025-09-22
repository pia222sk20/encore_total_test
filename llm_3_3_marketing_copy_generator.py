"""
문제 3.3: 마케팅 카피 생성기

지시사항:
GPT 모델을 활용해 다양한 마케팅 카피를 자동 생성하는 시스템을 구축하세요. 
제품명과 특징을 입력으로 받아 매력적인 마케팅 문구, 슬로건, 제품 설명을 
생성하는 기능을 구현하세요.
"""

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    print("Transformers 라이브러리가 사용 가능합니다.")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers 라이브러리가 설치되지 않았습니다.")

import random
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Product:
    """제품 정보 데이터 클래스"""
    name: str
    category: str
    features: List[str]
    target_audience: str
    price_range: str

class MarketingCopyGenerator:
    """마케팅 카피 생성기"""
    
    def __init__(self):
        # 마케팅 카피 템플릿들
        self.slogan_templates = {
            "aspirational": [
                "Transform your {category} experience with {product}",
                "Discover the future of {category}",
                "Elevate your lifestyle with {product}",
                "Where innovation meets {category}",
                "Unleash the power of {product}"
            ],
            "benefit_focused": [
                "Get {feature} like never before",
                "Experience {feature} that changes everything",
                "{product} - Because you deserve {feature}",
                "Revolutionary {feature} in every {product}",
                "The {category} that delivers {feature}"
            ],
            "emotional": [
                "Fall in love with {product}",
                "Your perfect {category} companion",
                "Life is better with {product}",
                "Make every moment count with {product}",
                "The {category} that understands you"
            ]
        }
        
        self.description_starters = [
            "Introducing the revolutionary",
            "Experience the next generation of",
            "Discover the ultimate",
            "Meet your new favorite",
            "The game-changing",
            "Prepare to be amazed by"
        ]
        
        self.feature_connectors = [
            "that delivers",
            "designed for",
            "engineered to provide",
            "crafted with",
            "built for those who value"
        ]
        
        self.call_to_action = [
            "Order now and experience the difference!",
            "Get yours today - limited time offer!",
            "Join thousands of satisfied customers!",
            "Don't wait - transform your life today!",
            "Experience the revolution - order now!",
            "Make the switch - you'll thank yourself later!"
        ]
        
        # 업계별 키워드
        self.industry_keywords = {
            "technology": ["innovative", "smart", "advanced", "cutting-edge", "intelligent", "seamless"],
            "beauty": ["radiant", "luxurious", "glamorous", "flawless", "stunning", "rejuvenating"],
            "fitness": ["powerful", "dynamic", "energizing", "transformative", "peak", "ultimate"],
            "food": ["delicious", "fresh", "authentic", "gourmet", "artisanal", "satisfying"],
            "fashion": ["stylish", "elegant", "trendy", "sophisticated", "chic", "timeless"],
            "home": ["comfortable", "cozy", "modern", "elegant", "functional", "beautiful"]
        }
    
    def create_slogan(self, product: Product, style: str = "aspirational") -> List[str]:
        """슬로건 생성"""
        templates = self.slogan_templates.get(style, self.slogan_templates["aspirational"])
        slogans = []
        
        for template in templates[:3]:  # 상위 3개 템플릿 사용
            # 템플릿 변수 치환
            slogan = template.format(
                product=product.name,
                category=product.category,
                feature=random.choice(product.features) if product.features else "excellence"
            )
            slogans.append(slogan)
        
        return slogans
    
    def create_product_description(self, product: Product, length: str = "medium") -> str:
        """제품 설명 생성"""
        # 업계 키워드 선택
        category_lower = product.category.lower()
        keywords = []
        for industry, words in self.industry_keywords.items():
            if industry in category_lower or category_lower in industry:
                keywords = words
                break
        
        if not keywords:
            keywords = ["exceptional", "premium", "high-quality", "outstanding", "remarkable"]
        
        # 설명 구성
        starter = random.choice(self.description_starters)
        connector = random.choice(self.feature_connectors)
        keyword = random.choice(keywords)
        
        # 기본 설명
        description = f"{starter} {product.name}, the {keyword} {product.category} {connector}"
        
        # 특징 추가
        if product.features:
            if len(product.features) == 1:
                description += f" {product.features[0]}."
            elif len(product.features) == 2:
                description += f" {product.features[0]} and {product.features[1]}."
            else:
                features_text = ", ".join(product.features[:-1]) + f", and {product.features[-1]}"
                description += f" {features_text}."
        
        # 길이에 따른 추가 내용
        if length == "long":
            description += f" Perfect for {product.target_audience}, this {product.category} "
            description += f"combines innovation with practicality. "
            if product.price_range:
                description += f"Available at an {product.price_range} price point, "
            description += f"the {product.name} is your gateway to a better experience."
        
        return description
    
    def create_ad_copy(self, product: Product) -> Dict[str, str]:
        """완전한 광고 카피 생성"""
        headline = f"Discover the Power of {product.name}"
        
        # 부제목
        if product.features:
            main_feature = product.features[0]
            subheading = f"Revolutionary {product.category} with {main_feature}"
        else:
            subheading = f"The Ultimate {product.category} Experience"
        
        # 본문
        body = self.create_product_description(product, "long")
        
        # 행동 유도
        cta = random.choice(self.call_to_action)
        
        return {
            "headline": headline,
            "subheading": subheading,
            "body": body,
            "call_to_action": cta
        }

def create_marketing_generator():
    """마케팅 생성기 생성"""
    
    if TRANSFORMERS_AVAILABLE:
        try:
            print("GPT-2 모델 로딩 중...")
            generator = pipeline('text-generation', model='gpt2')
            print("AI 마케팅 생성기 로딩 완료!")
            return generator, "AI Generator"
        except Exception as e:
            print(f"AI 모델 로딩 실패: {e}")
            print("템플릿 기반 생성기를 사용합니다.")
            return MarketingCopyGenerator(), "Template Generator"
    else:
        print("Transformers가 설치되지 않아 템플릿 기반 생성기를 사용합니다.")
        return MarketingCopyGenerator(), "Template Generator"

def generate_with_ai(generator, prompt: str) -> str:
    """AI 모델로 마케팅 카피 생성"""
    try:
        result = generator(
            prompt,
            max_length=80,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        generated_text = result[0]['generated_text']
        # 프롬프트 제거하고 생성된 부분만 반환
        new_text = generated_text[len(prompt):].strip()
        return new_text
    
    except Exception as e:
        return f"AI 생성 오류: {e}"

def demo_product_examples():
    """다양한 제품 예시로 데모"""
    
    print("=== 마케팅 카피 생성기 데모 ===")
    
    # 샘플 제품들
    products = [
        Product(
            name="EcoClean Pro",
            category="cleaning device",
            features=["eco-friendly formula", "99% bacteria removal", "wireless operation"],
            target_audience="environmentally conscious families",
            price_range="affordable"
        ),
        Product(
            name="FitTracker X1",
            category="fitness wearable",
            features=["24/7 heart monitoring", "waterproof design", "7-day battery life"],
            target_audience="fitness enthusiasts",
            price_range="premium"
        ),
        Product(
            name="GlowSerum Elite",
            category="beauty product",
            features=["anti-aging formula", "natural ingredients", "instant results"],
            target_audience="beauty-conscious individuals",
            price_range="luxury"
        )
    ]
    
    generator, generator_type = create_marketing_generator()
    
    for i, product in enumerate(products, 1):
        print(f"\n{'='*20} 제품 {i}: {product.name} {'='*20}")
        print(f"카테고리: {product.category}")
        print(f"주요 특징: {', '.join(product.features)}")
        print(f"타겟 고객: {product.target_audience}")
        print(f"가격대: {product.price_range}")
        
        if generator_type == "AI Generator":
            # AI 기반 생성
            print(f"\n--- AI 생성 결과 ---")
            
            # 슬로건 생성
            slogan_prompt = f"Create a catchy slogan for {product.name}, a {product.category}: "
            ai_slogan = generate_with_ai(generator, slogan_prompt)
            print(f"AI 슬로건: {ai_slogan}")
            
            # 제품 설명 생성
            desc_prompt = f"Write a marketing description for {product.name}, a {product.category} with {product.features[0]}: "
            ai_description = generate_with_ai(generator, desc_prompt)
            print(f"AI 설명: {ai_description}")
        
        else:
            # 템플릿 기반 생성
            print(f"\n--- 템플릿 기반 생성 결과 ---")
            
            # 다양한 스타일의 슬로건
            for style in ["aspirational", "benefit_focused", "emotional"]:
                slogans = generator.create_slogan(product, style)
                print(f"\n{style.replace('_', ' ').title()} 슬로건:")
                for j, slogan in enumerate(slogans, 1):
                    print(f"  {j}. {slogan}")
            
            # 제품 설명
            description = generator.create_product_description(product, "medium")
            print(f"\n제품 설명:\n  {description}")
            
            # 완전한 광고 카피
            ad_copy = generator.create_ad_copy(product)
            print(f"\n완전한 광고 카피:")
            print(f"  헤드라인: {ad_copy['headline']}")
            print(f"  부제목: {ad_copy['subheading']}")
            print(f"  본문: {ad_copy['body']}")
            print(f"  행동유도: {ad_copy['call_to_action']}")

def interactive_copy_generator():
    """대화형 카피 생성기"""
    
    print("\n=== 대화형 마케팅 카피 생성기 ===")
    print("제품 정보를 입력하면 맞춤형 마케팅 카피를 생성해드립니다!")
    
    # 사용자 입력 시뮬레이션 (실제 환경에서는 input() 사용)
    sample_inputs = [
        {
            "name": "SmartMug Pro",
            "category": "smart drinkware",
            "features": ["temperature control", "app connectivity", "leak-proof design"],
            "target": "tech-savvy professionals",
            "price": "premium"
        },
        {
            "name": "ZenPillow",
            "category": "sleep accessory",
            "features": ["memory foam", "cooling gel", "ergonomic design"],
            "target": "people with sleep issues",
            "price": "mid-range"
        }
    ]
    
    generator, generator_type = create_marketing_generator()
    
    for i, input_data in enumerate(sample_inputs, 1):
        print(f"\n--- 예시 입력 {i} ---")
        print(f"제품명: {input_data['name']}")
        print(f"카테고리: {input_data['category']}")
        print(f"주요 특징: {', '.join(input_data['features'])}")
        print(f"타겟 고객: {input_data['target']}")
        print(f"가격대: {input_data['price']}")
        
        # Product 객체 생성
        product = Product(
            name=input_data['name'],
            category=input_data['category'],
            features=input_data['features'],
            target_audience=input_data['target'],
            price_range=input_data['price']
        )
        
        print(f"\n생성된 마케팅 카피:")
        
        if generator_type == "Template Generator":
            # 슬로건 생성
            slogans = generator.create_slogan(product, "aspirational")
            print(f"슬로건: {slogans[0]}")
            
            # 제품 설명 생성
            description = generator.create_product_description(product, "medium")
            print(f"설명: {description}")
            
            # 광고 카피 생성
            ad_copy = generator.create_ad_copy(product)
            print(f"광고 헤드라인: {ad_copy['headline']}")

def marketing_best_practices():
    """마케팅 카피 모범 사례"""
    
    print("\n=== 마케팅 카피 작성 모범 사례 ===")
    
    best_practices = {
        "헤드라인 작성": [
            "명확하고 간결한 메시지 전달",
            "혜택이나 결과를 강조",
            "감정적 연결 시도",
            "궁금증 유발하는 요소 포함",
            "타겟 고객에게 직접적으로 말하기"
        ],
        "제품 설명": [
            "특징(Feature)보다 혜택(Benefit) 강조",
            "구체적인 수치나 결과 제시",
            "고객의 문제점 해결 방법 명시",
            "사회적 증거나 신뢰성 요소 포함",
            "명확한 가치 제안 제시"
        ],
        "행동 유도": [
            "명확하고 구체적인 다음 단계 제시",
            "긴급성이나 희소성 활용",
            "위험 요소 제거 (환불 보장 등)",
            "간단하고 실행하기 쉬운 행동",
            "혜택 재강조"
        ],
        "언어 사용": [
            "타겟 고객의 언어 수준에 맞추기",
            "긍정적이고 활동적인 표현 사용",
            "전문 용어보다 일반적인 언어",
            "감정을 자극하는 형용사 활용",
            "읽기 쉬운 구조와 리듬"
        ]
    }
    
    for category, practices in best_practices.items():
        print(f"\n{category}:")
        for practice in practices:
            print(f"  • {practice}")

def industry_specific_tips():
    """업계별 마케팅 팁"""
    
    print("\n=== 업계별 마케팅 카피 팁 ===")
    
    industry_tips = {
        "기술/전자제품": {
            "핵심 키워드": ["혁신적", "스마트", "효율적", "최신", "고성능"],
            "강조점": "기술적 우위와 사용 편의성",
            "주의사항": "과도한 기술 용어 사용 피하기"
        },
        "뷰티/화장품": {
            "핵심 키워드": ["빛나는", "젊어지는", "자연스러운", "럭셔리", "완벽한"],
            "강조점": "즉각적인 결과와 장기적인 혜택",
            "주의사항": "과장된 효과 주장 피하기"
        },
        "패션/의류": {
            "핵심 키워드": ["스타일리시", "트렌디", "편안한", "우아한", "세련된"],
            "강조점": "스타일과 개성 표현",
            "주의사항": "트렌드 변화에 민감하게 대응"
        },
        "식품/음료": {
            "핵심 키워드": ["신선한", "맛있는", "건강한", "프리미엄", "authentic"],
            "강조점": "맛과 건강, 품질",
            "주의사항": "건강 효능 과대 광고 피하기"
        },
        "피트니스/헬스": {
            "핵심 키워드": ["강력한", "변화", "에너지", "성과", "목표"],
            "강조점": "결과와 변화, 동기 부여",
            "주의사항": "비현실적인 결과 약속 피하기"
        }
    }
    
    for industry, tips in industry_tips.items():
        print(f"\n{industry}:")
        print(f"  핵심 키워드: {', '.join(tips['핵심 키워드'])}")
        print(f"  강조점: {tips['강조점']}")
        print(f"  주의사항: {tips['주의사항']}")

def a_b_testing_guide():
    """A/B 테스트 가이드"""
    
    print("\n=== 마케팅 카피 A/B 테스트 가이드 ===")
    
    print("테스트할 요소들:")
    test_elements = [
        "헤드라인의 길이와 톤",
        "혜택 vs 특징 강조",
        "감정적 vs 논리적 어필",
        "긴급성 포함 여부",
        "가격 정보 노출 방식",
        "행동 유도 문구의 강도",
        "이미지와 텍스트 조합"
    ]
    
    for element in test_elements:
        print(f"  • {element}")
    
    print("\n테스트 방법:")
    methods = [
        "단일 요소씩 변경하여 테스트",
        "충분한 샘플 크기 확보",
        "통계적 유의성 확인",
        "다양한 시간대와 요일 테스트",
        "타겟 그룹별 반응 차이 분석"
    ]
    
    for method in methods:
        print(f"  • {method}")

def ethical_considerations():
    """윤리적 고려사항"""
    
    print("\n=== 마케팅에서의 윤리적 고려사항 ===")
    
    considerations = [
        "과장 광고나 허위 정보 금지",
        "취약 계층 타겟팅 시 주의",
        "개인정보 수집과 사용의 투명성",
        "문화적 감수성 고려",
        "사회적 책임 의식",
        "환경적 영향 고려",
        "공정한 경쟁 원칙 준수"
    ]
    
    print("주요 윤리 원칙:")
    for consideration in considerations:
        print(f"  • {consideration}")
    
    print("\n권장 실천 방안:")
    practices = [
        "사실에 기반한 마케팅 메시지",
        "고객의 실제 필요 충족에 집중",
        "투명하고 정직한 커뮤니케이션",
        "다양성과 포용성 존중",
        "지속 가능한 비즈니스 모델 추구"
    ]
    
    for practice in practices:
        print(f"  • {practice}")

def main():
    print("=== 문제 3.3: 마케팅 카피 생성기 ===")
    
    # 1. 제품 예시 데모
    demo_product_examples()
    
    # 2. 대화형 생성기
    interactive_copy_generator()
    
    # 3. 마케팅 모범 사례
    marketing_best_practices()
    
    # 4. 업계별 팁
    industry_specific_tips()
    
    # 5. A/B 테스트 가이드
    a_b_testing_guide()
    
    # 6. 윤리적 고려사항
    ethical_considerations()
    
    # 설치 안내
    if not TRANSFORMERS_AVAILABLE:
        print("\n=== 설치 안내 ===")
        print("AI 기반 마케팅 카피 생성을 위해:")
        print("pip install transformers torch")

if __name__ == "__main__":
    main()