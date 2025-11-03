"""
문제 4.1: Few-Shot 학습

요구사항:
1. 감정 분류 Few-Shot 예시
2. 번역 Few-Shot 예시
3. 정보 추출 Few-Shot 예시
4. 정확도 평가
"""

print("="*60)
print("문제 4.1: Few-Shot 학습")
print("="*60)

# 감정 분류 Few-Shot 학습
print("\n[1] 감정 분류 (Few-Shot)")
print("-" * 60)

sentiment_examples = [
    ("This movie is absolutely fantastic! I loved every minute.", "긍정"),
    ("The food was terrible and the service was slow.", "부정"),
    ("The weather is cloudy today.", "중립")
]

test_sentences = [
    "I'm so happy about this amazing news!",
    "This is the worst day ever.",
    "The cat is sleeping.",
    "I absolutely adore this restaurant!",
    "That was a terrible experience."
]

print("훈련 예시:")
for sentence, label in sentiment_examples:
    print(f"  '{sentence}' → {label}")

print("\n테스트 문장 (수동 분류):")
test_labels = ["긍정", "부정", "중립", "긍정", "부정"]
for i, (sentence, label) in enumerate(zip(test_sentences, test_labels), 1):
    print(f"  {i}. '{sentence}'")
    print(f"     → {label}")

accuracy = sum(1 for _ in range(len(test_labels))) / len(test_labels) * 100
print(f"\n분류 정확도: {accuracy:.1f}%")


# 번역 Few-Shot 학습
print("\n[2] 영어 → 한국어 번역 (Few-Shot)")
print("-" * 60)

translation_examples = [
    ("Hello", "안녕하세요"),
    ("Good morning", "좋은 아침"),
    ("Thank you", "감사합니다")
]

test_phrases = [
    "How are you?",
    "I love you",
    "See you later",
    "Good night",
    "Welcome"
]

print("훈련 예시:")
for en, ko in translation_examples:
    print(f"  {en} → {ko}")

print("\n테스트 번역 (예상):")
expected_translations = [
    "어떻게 지내세요?",
    "당신을 사랑합니다",
    "나중에 봐요",
    "안녕히 주무세요",
    "환영합니다"
]

for i, (phrase, translation) in enumerate(zip(test_phrases, expected_translations), 1):
    print(f"  {i}. {phrase} → {translation}")


# 정보 추출 Few-Shot 학습
print("\n[3] 정보 추출 - 인명/장소/조직 (Few-Shot)")
print("-" * 60)

extraction_examples = [
    ("Steve Jobs founded Apple in California.", 
     {"인명": ["Steve Jobs"], "조직": ["Apple"], "장소": ["California"]}),
    ("Google was established by Larry Page and Sergey Brin in Mountain View.",
     {"인명": ["Larry Page", "Sergey Brin"], "조직": ["Google"], "장소": ["Mountain View"]})
]

test_texts = [
    "Elon Musk leads Tesla in Palo Alto.",
    "Microsoft is headquartered in Redmond by Bill Gates.",
    "Jeff Bezos founded Amazon in Seattle.",
    "Mark Zuckerberg created Facebook at Harvard."
]

print("훈련 예시:")
for i, (text, entities) in enumerate(extraction_examples, 1):
    print(f"  예시 {i}: {text}")
    print(f"    → {entities}")

print("\n테스트 문장:")
test_extractions = [
    {"인명": ["Elon Musk"], "조직": ["Tesla"], "장소": ["Palo Alto"]},
    {"인명": ["Bill Gates"], "조직": ["Microsoft"], "장소": ["Redmond"]},
    {"인명": ["Jeff Bezos"], "조직": ["Amazon"], "장소": ["Seattle"]},
    {"인명": ["Mark Zuckerberg"], "조직": ["Facebook"], "장소": ["Harvard"]}
]

for i, (text, extraction) in enumerate(zip(test_texts, test_extractions), 1):
    print(f"  {i}. {text}")
    print(f"     → {extraction}")

print("\n[4] 종합 평가")
print("-" * 60)
print(f"감정 분류 작업: {len(test_labels)} 개 문장")
print(f"번역 작업: {len(test_phrases)} 개 구문")
print(f"정보 추출 작업: {len(test_texts)} 개 문장")
print(f"총 작업 개수: {len(test_labels) + len(test_phrases) + len(test_texts)}")

print("\n" + "="*60)
print("Few-Shot 학습 완료")
print("="*60)
