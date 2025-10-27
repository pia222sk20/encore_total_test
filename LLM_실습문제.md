# LLM 실습 문제 

---

## 목차

- [Chapter 1: 텍스트 데이터의 표현](#chapter-1-텍스트-데이터의-표현)
- [Chapter 2: 자연어 딥러닝의 핵심 개념](#chapter-2-자연어-딥러닝의-핵심-개념)
- [Chapter 3: 초거대 언어 모델의 실제적 활용](#chapter-3-초거대-언어-모델의-실제적-활용)
- [Chapter 4: 고급 프롬프트 엔지니어링 기법](#chapter-4-고급-프롬프트-엔지니어링-기법)
- [Chapter 5: 파인튜닝을 통한 LLM의 효율적 적응](#chapter-5-파인튜닝을-통한-llm의-효율적-적응)
- [Chapter 6: 멀티모달 AI의 최전선](#chapter-6-멀티모달-ai의-최전선)

---

## Chapter 1: 텍스트 데이터의 표현

### 문제 1.1: 텍스트 전처리 파이프라인 구축

**💡 학습 목표:**
원시 텍스트를 기계가 이해할 수 있는 깨끗한 형태로 변환하는 과정을 배웁니다. 자연어 처리의 가장 첫 번째 단계이며, 데이터의 품질이 모델의 성능을 크게 좌우합니다.

**📋 상세 지시사항:**

1. **NLTK 라이브러리 이해**
   - NLTK: Natural Language Toolkit - 자연어 처리를 위한 파이썬 라이브러리
   - 토크나이저, 불용어, 다양한 NLP 도구 포함
   - 첫 실행 시 필요한 데이터 자동 다운로드 필요

2. **텍스트 전처리 단계별 구현**

   **Step 1: 원시 문장 입력**
   ```python
   text = "Hello everyone, this is the first document for our NLP task!"
   ```

   **Step 2: 소문자 변환**
   ```python
   lower_text = text.lower()
   # 결과: "hello everyone, this is the first document for our nlp task!"
   # 이유: 같은 단어를 'Hello'와 'hello'로 중복 인식하지 않기 위함
   ```

   **Step 3: 토큰화 (단어 분리)**
   ```python
   from nltk.tokenize import word_tokenize
   tokens = word_tokenize(lower_text)
   # 결과: ['hello', 'everyone', ',', 'this', 'is', 'the', 'first', 'document', 'for', 'our', 'nlp', 'task', '!']
   # 주의: 구두점도 별도의 토큰으로 분리됨
   ```

   **Step 4: 구두점 제거**
   ```python
   # 알파벳 문자만 남기기: isalpha() 메서드 사용
   alphabetic_tokens = [word for word in tokens if word.isalpha()]
   # 결과: ['hello', 'everyone', 'this', 'is', 'the', 'first', 'document', 'for', 'our', 'nlp', 'task']
   ```

   **Step 5: 불용어(Stopwords) 제거**
   ```python
   from nltk.corpus import stopwords
   stop_words = set(stopwords.words('english'))
   # 불용어: 'the', 'is', 'a', 'for' 등 빈번하지만 의미가 없는 단어들
   
   filtered_tokens = [word for word in alphabetic_tokens if word not in stop_words]
   # 결과: ['hello', 'everyone', 'first', 'document', 'nlp', 'task']
   # 13개 토큰 → 6개 토큰으로 53% 축소
   ```

3. **전체 단계 적용**
   - 제공된 3개 문장에 모두 적용
   - 각 문장별 처리 과정 출력
   - 최종 결과 비교

4. **분석 및 통계**
   - 원본 단어 수 vs 전처리 후 단어 수
   - 제거된 구두점 목록
   - 제거된 불용어 목록
   - 최종 고유 단어 목록

**💾 파일명:** `llm_1_1_text_preprocessing.py`

**🔍 기대 출력:**
```
=== 문제 1.1: 텍스트 전처리 파이프라인 구축 ===

=== 원본 텍스트 ===
1. Hello everyone, this is the first document for our NLP task!
2. We are learning about Natural Language Processing, which is very exciting.
3. Preprocessing text is an important and fundamental step.

=== 전처리 결과 ===
1. ['hello', 'everyone', 'first', 'document', 'nlp', 'task']
2. ['learning', 'natural', 'language', 'processing', 'exciting']
3. ['preprocessing', 'text', 'important', 'fundamental', 'step']

=== 통계 정보 ===
원본 총 단어 수: 35
전처리 후 단어 수: 16
압축률: 54.3%
```

**💡 핵심 개념:**
- **왜 전처리가 필요한가?** 원시 데이터에는 노이즈가 많아서 모델이 본질적인 패턴을 학습하기 어려움
- **불용어 제거의 장점:** 메모리 사용량 감소, 계산 속도 향상, 노이즈 감소
- **한계:** 문맥에 따라 불용어가 중요할 수 있음 (예: "be" 동사)

---

### 문제 1.2: 텍스트에서 벡터로 - TF-IDF

**💡 학습 목표:**
전처리된 텍스트를 컴퓨터가 계산할 수 있는 숫자 벡터로 변환합니다. 기계학습 모델은 텍스트를 직접 이해하지 못하므로, 먼저 벡터화(vectorization)가 필수입니다.

**📋 상세 지시사항:**

1. **TF-IDF의 이해**

   **TF (Term Frequency): 단어 빈도**
   $$TF(w, d) = \frac{\text{단어 } w \text{가 문서 } d \text{에 등장한 횟수}}{\text{문서 } d \text{의 전체 단어 수}}$$
   
   예시: 문서에 총 100개 단어가 있고, "machine"이 5번 나타나면
   - $TF(\text{machine}) = 5/100 = 0.05$

   **IDF (Inverse Document Frequency): 역문서 빈도**
   $$IDF(w) = \log\left(\frac{\text{전체 문서 수}}{\text{단어 } w \text{를 포함한 문서 수}}\right)$$
   
   예시: 전체 1000개 문서 중 "machine"을 포함한 문서가 50개
   - $IDF(\text{machine}) = \log(1000/50) = \log(20) \approx 2.996$
   
   직관: 자주 등장하는 단어는 IDF가 낮고, 드문 단어는 IDF가 높음

   **TF-IDF**
   $$TF\text{-}IDF(w, d) = TF(w, d) \times IDF(w)$$

2. **벡터화 과정**
   - 문제 1.1의 전처리된 3개 문장 사용
   - scikit-learn의 `TfidfVectorizer` 적용
   - 결과: 3×N 행렬 (3개 문장, N개 고유 단어)

3. **결과 분석**
   - 각 문장이 어떤 단어에 높은 가중치를 가지는지 확인
   - 단어별 IDF 값 확인
   - 새로운 문장을 벡터화하여 기존 문장과 유사도 계산

4. **의미 해석**
   - 고유한 단어(특정 문서에만 등장): TF-IDF 값 높음 → 그 문서의 특성을 잘 나타냄
   - 공통 단어(모든 문서에 등장): TF-IDF 값 낮음 → 구분력 낮음
   - 매우 흔한 단어(거의 모든 문서에 등장): TF-IDF 값 거의 0

**💾 파일명:** `llm_1_2_tfidf_vectorization.py`


**🔍 기대 출력:**
```
=== 문제 1.2: 텍스트에서 벡터로 - TF-IDF ===

=== 피처 이름 (어휘 사전) ===
['document', 'exciting', 'first', 'fundamental', 'important', 'learning', 'nlp', 'preprocessing', 'processing', 'step', 'task', 'text']

=== TF-IDF 행렬 ===
             document  exciting  first  ...  task  text
문서 1       0.316228       0.0  0.316228      0.316228
문서 2       0.000000       0.447214  0.000000  0.000000
문서 3       0.000000       0.0  0.000000  0.000000

=== 각 단어의 IDF 값 ===
        단어  IDF 값
document    1.609438
exciting    1.609438
first       1.609438
...
```

**💡 핵심 개념:**
- **장점:** 간단하면서도 효과적, 희귀 단어에 가중치 부여
- **단점:** 단어 순서 무시 (Bag of Words), 의미 관계 반영 안 함
- **개선:** Word2Vec, BERT 등 임베딩 기법

---

### 문제 1.3: 시맨틱 능력의 발현 - 단어 임베딩 비교

**💡 학습 목표:**
단어의 "의미"를 수치 벡터로 표현하는 임베딩의 개념을 이해합니다. 임베딩은 의미적으로 유사한 단어들이 벡터 공간에서 가까이 위치하도록 학습됩니다.

**📋 상세 지시사항:**

1. **단어 임베딩의 개념**
   
   **TF-IDF vs Word Embedding 비교:**
   ```
   TF-IDF:      단순히 단어가 많이 등장하는지 여부 (빈도 기반)
   임베딩:      단어의 '의미'를 학습 (의미 기반)
   
   예: "좋다"와 "훌륭하다"
   - TF-IDF: 전혀 다른 벡터 (다른 단어)
   - 임베딩: 매우 유사한 벡터 (유사한 의미)
   ```

2. **코사인 유사도(Cosine Similarity)**
   
   $$\text{cos\_similarity}(v_1, v_2) = \frac{v_1 \cdot v_2}{||v_1|| \times ||v_2||}$$
   
   - 범위: -1 ~ 1 (보통 0 ~ 1)
   - 1에 가까울수록: 같은 방향 (매우 유사)
   - 0: 수직 (관련 없음)
   - -1에 가까울수록: 반대 방향 (정반대)

3. **의미 유추 테스트 (Analogy Test)**
   
   **가설:** 임베딩이 제대로 학습되었다면
   ```
   king - man + woman ≈ queen
   ```
   
   **원리:**
   - "king" 벡터에는 "왕"이라는 개념과 "남성"이라는 속성 포함
   - "man" 빼기 → 남성 속성 제거
   - "woman" 더하기 → 여성 속성 추가
   - 결과 → "queen" (여왕)의 의미

4. **실습 단계**
   - Sentence-Transformers 라이브러리의 사전 훈련 모델 로드
   - 단어 임베딩 생성
   - 유사도 계산: (king, queen), (king, man), (king, woman)
   - 의미 유추 테스트 수행
   - 결과 분석

**💾 파일명:** `llm_1_3_word_embeddings.py`


**🔍 기대 출력:**
```
=== 문제 1.3: 시맨틱 능력의 발현 - 단어 임베딩 비교 ===

--- 2. 의미적 유사도 비교 ---
유사도 ('king', 'queen'): 0.8234 (왕-여왕: 높은 유사도)
유사도 ('king', 'man'):   0.7123 (왕-남자: 중간 유사도)
유사도 ('king', 'woman'): 0.4532 (왕-여자: 낮은 유사도)

--- 3. 벡터 연산을 통한 단어 유추 ---
수행할 연산: king - man + woman ≈ queen
연산 결과 벡터와 'queen' 벡터의 유사도: 0.7891
```

**💡 핵심 개념:**
- **의미 관계 학습:** 임베딩은 단어 간 의미적 관계를 벡터 공간 구조로 표현
- **벡터 연산의 의미:** 벡터 덧셈/뺄셈은 의미의 조합/제거 의미
- **전이학습:** 대규모 데이터로 학습된 임베딩을 다양한 작업에 재사용

---

## Chapter 2: 자연어 딥러닝의 핵심 개념

### 문제 2.1: LSTM을 이용한 순차 데이터 모델링 및 감성 분석

**💡 학습 목표:**
순차적 데이터의 시간적 의존성을 학습하는 LSTM 신경망을 이해하고, 이를 사용하여 텍스트의 감정(감성)을 분류합니다.

**📋 상세 지시사항:**

1. **LSTM의 필요성**
   
   **RNN의 한계:** 긴 시퀀스에서 "기울기 소실(Vanishing Gradient)" 문제 발생
   - 입력(처음) → 출력(끝)까지 가는 경로가 너무 길어서 정보 손실
   
   **LSTM의 해결법:** 특별한 메모리 셀(Cell)로 중요 정보 보존
   - 입력 게이트: 새로운 정보를 셀에 더할지 결정
   - 망각 게이트: 이전 정보 중 버릴 것 결정
   - 출력 게이트: 셀에서 어떤 정보를 꺼낼지 결정

2. **모델 구조**
   ```
   입력 문장
      ↓
   [임베딩 레이어]  → 단어 인덱스를 의미 벡터로 변환
      ↓
   [LSTM 레이어]   → 문장의 순차적 의존성 학습 (문맥 파악)
      ↓
   [완전 연결 레이어] → 분류 수행 (긍정/부정)
      ↓
   [시그모이드 함수]  → 확률값으로 변환 (0~1)
      ↓
   결과: 긍정(1) 또는 부정(0)
   ```

3. **훈련 데이터**
   ```python
   {
       "This movie was fantastic and amazing": 1,           # 긍정
       "The acting was terrible and boring": 0,             # 부정
       "I really enjoyed the plot and characters": 1,       # 긍정
       "A complete waste of time and money": 0,             # 부정
       "The visuals were stunning, a masterpiece": 1,       # 긍정
       "I would not recommend this film to anyone": 0       # 부정
   }
   ```

4. **실습 단계**
   - 데이터 준비: 텍스트 → 정수 인코딩 → 패딩
   - 모델 정의: Embedding + LSTM + FC + Sigmoid
   - 모델 훈련: 100 에포크, 이진 크로스 엔트로피 손실 함수
   - 모델 테스트: 새 문장 감성 분류

5. **결과 해석**
   - 예측 점수 > 0.5 → 긍정
   - 예측 점수 < 0.5 → 부정
   - 점수가 0.5에 가까울수록 확신도 낮음

**💾 파일명:** `llm_2_1_lstm_sentiment.py`

**🔍 기대 출력:**
```
=== 문제 2.1: LSTM을 이용한 순차 데이터 모델링 ===

어휘 사전 크기: 22
데이터가 (6, 12) 모양의 텐서로 변환되었습니다.

--- 2. 모델 훈련 ---
Epoch 20/100, Loss: 0.6234
Epoch 40/100, Loss: 0.4567
Epoch 60/100, Loss: 0.2891
Epoch 80/100, Loss: 0.1234
Epoch 100/100, Loss: 0.0567

--- 3. 모델 테스트 ---
테스트 문장: 'The movie was good and enjoyable'
예측 점수: 0.8234
예측된 감성: 긍정

테스트 문장: 'The plot was predictable and dull'
예측 점수: 0.2145
예측된 감성: 부정
```

**💡 핵심 개념:**
- **임베딩:** 단어를 의미 벡터로 표현 (같은 의미의 단어 → 유사한 벡터)
- **LSTM:** 순차 데이터의 "문맥"을 학습 (단어 순서 중요)
- **분류:** 최종 은닉 상태를 기반으로 감성 분류

---

### 문제 2.2: BERT를 활용한 전이학습의 위력

**💡 학습 목표:**
대규모 데이터로 사전 훈련된 모델을 사용하여, 적은 코드와 적은 데이터로 높은 성능을 달성하는 전이학습(Transfer Learning)의 강력함을 경험합니다.

**📋 상세 지시사항:**

1. **전이학습의 개념**
   
   **기존 방식 (문제 2.1):**
   ```
   작은 훈련 데이터 → 모델 정의 → 매개변수 초기화 (무작위)
   → 처음부터 학습 → 성능 낮음
   ```
   
   **전이학습 방식:**
   ```
   대규모 외부 데이터 (예: 위키피디아)
              ↓
   BERT 사전 훈련 (이미 완료됨)
              ↓
   매개변수 사용 (이미 좋은 특성 학습됨)
              ↓
   우리의 작은 데이터로 미세 조정(Fine-tuning)
              ↓
   높은 성능!
   ```

2. **BERT의 특징**
   - **양방향:** 문맥을 양쪽 모두에서 봄
   - **사전 훈련:** 마스킹된 언어 모델링 + 다음 문장 예측
   - **일반화:** 다양한 NLP 작업에 적용 가능

3. **Hugging Face 라이브러리**
   - `transformers`: 다양한 사전 훈련 모델 제공
   - `pipeline`: 복잡한 코드를 간단하게 추상화
   
   **사용법:**
   ```python
   from transformers import pipeline
   classifier = pipeline("sentiment-analysis")
   result = classifier("I love this movie!")
   # 결과: [{'label': 'POSITIVE', 'score': 0.9998}]
   ```

4. **비교 분석**
   
   | 항목 | LSTM (직접 구축) | BERT (전이학습) |
   |------|-----------------|-----------------|
   | 코드 길이 | ~200줄 | ~10줄 |
   | 훈련 데이터 | 6개 (작음) | 필요 없음 (미리 학습됨) |
   | 성능 | 중간 | 우수 |
   | 훈련 시간 | 몇 초 | 없음 (사전 훈련 모델 로드) |

5. **다양한 입력 테스트**
   - 명확한 긍정/부정 문장
   - 중립적인 문장
   - 복합 감정 문장
   - 비감정 텍스트

**💾 파일명:** `llm_2_2_bert_sentiment.py`

**🔍 기대 출력:**
```
=== 문제 2.2: BERT를 활용한 전이학습의 위력 ===

테스트 문장: "I am absolutely thrilled with the results!"
예측 결과: POSITIVE (신뢰도: 0.9998)

테스트 문장: "This is the worst experience I have ever had."
예측 결과: NEGATIVE (신뢰도: 0.9995)

테스트 문장: "The movie was okay, but I probably wouldn't watch it again."
예측 결과: NEGATIVE (신뢰도: 0.8234)
```

**💡 핵심 개념:**
- **전이학습:** 한 작업에서 학습한 지식을 다른 작업에 적용
- **미세 조정:** 사전 훈련 모델을 우리 데이터에 맞게 조정
- **효율성:** 대규모 데이터와 컴퓨팅 없이도 고성능 달성 가능

---

## Chapter 3: 초거대 언어 모델의 실제적 활용

### 문제 3.1: GPT를 이용한 텍스트 완성

**💡 학습 목표:**
자동 회귀(Autoregressive) 모델인 GPT를 사용하여 주어진 프롬프트를 기반으로 텍스트를 생성하고, 생성 파라미터가 출력에 미치는 영향을 이해합니다.

**📋 상세 지시사항:**

1. **생성 모델의 작동 원리**
   ```
   프롬프트: "Once upon a time"
   
   단계 1: 다음 단어 예측 → "there" (확률: 0.8)
   현재: "Once upon a time there"
   
   단계 2: 다음 단어 예측 → "was" (확률: 0.7)
   현재: "Once upon a time there was"
   
   단계 3: 다음 단어 예측 → "a" (확률: 0.6)
   현재: "Once upon a time there was a"
   
   ... (반복)
   ```

2. **생성 파라미터**
   
   - **temperature**: 창의성 제어 (0 ~ 1+)
     - 낮음 (0.1): 결정적, 예측 가능 (같은 결과 반복)
     - 중간 (0.7): 균형잡힌 창의성과 일관성
     - 높음 (1.5): 매우 창의적, 예측 불가능
   
   - **top_p**: 누적 확률 기반 샘플링
     - 0.9: 상위 90% 확률의 단어들만 선택
     - 0.5: 상위 50% 확률의 단어들만 선택 (더 집중)
   
   - **max_length**: 생성 최대 길이 (50~100 단어)

3. **다양한 프롬프트 테스트**
   - "The future of AI is..." (기술)
   - "Once upon a time..." (이야기)
   - "The most important lesson..." (교육)

4. **생성 결과 분석**
   - 각 온도 설정별 결과 비교
   - 창의성 vs 일관성 트레이드오프
   - 실무 활용: 고객 서비스(일관성 중요) vs 창작(창의성 중요)

**💾 파일명:** `llm_3_1_gpt_text_completion.py`



---

### 문제 3.2: 창의성 제어하기 - 생성 파라미터의 영향

**💡 학습 목표:**
생성 모델의 파라미터들이 출력에 미치는 구체적인 영향을 실험을 통해 이해합니다.

**📋 상세 지시사항:**

1. **파라미터별 생성 비교**
   - 동일 프롬프트
   - 다양한 temperature (0.3, 0.7, 1.0, 1.3)
   - 다양한 top_p (0.9, 0.7, 0.5)

2. **정량적 분석**
   - 문장 길이 비교
   - 고유 단어 개수
   - 반복 패턴 분석

3. **정성적 분석**
   - 결과의 일관성
   - 창의성 수준
   - 현실성/타당성

**💾 파일명:** `llm_3_2_generation_parameters.py`

---

### 문제 3.3: 간단한 도메인 응용 - 마케팅 카피 생성기

**💡 학습 목표:**
실제 비즈니스 문제(마케팅 콘텐츠 생성)에 LLM을 적용하는 방법을 배웁니다.

**📋 상세 지시사항:**

1. **마케팅 카피의 특성**
   - 간결성: 짧고 임팩트 있음
   - 행동 유도: 클릭/구매 유도
   - 감정 호소: 긍정적 감정 자극

2. **프롬프트 엔지니어링**
   ```python
   프롬프트 = """
   다음은 제품 설명입니다:
   제품명: 스마트 워터병
   특징: 수분 섭취 추적, 앱 연동, 자동 알림
   
   이 제품을 위한 3줄짜리 마케팅 카피를 작성하세요.
   카피는 젊은 사람들을 타겟으로 하며, 행동을 유도하는 내용이어야 합니다.
   """
   ```

3. **다양한 제품 카피 생성**
   - 최신 스마트폰
   - 오가닉 커피
   - 피트니스 앱

**💾 파일명:** `llm_3_3_marketing_copy_generator.py`

---

## Chapter 4: 고급 프롬프트 엔지니어링 기법

### 문제 4.1: 퓨샷(Few-Shot) 인-컨텍스트 러닝

**💡 학습 목표:**
프롬프트에 예시를 포함하여 모델의 성능을 향상시키는 기법을 배웁니다.

**📋 상세 지시사항:**

1. **제로샷 vs 퓨샷**
   
   **제로샷 (예시 없음):**
   ```
   입력: "The Eiffel Tower is in France"
   작업: 이 문장의 감성을 분류하세요.
   모델이 감성이 없어서 제대로 분류하지 못함
   ```
   
   **퓨샷 (예시 포함):**
   ```
   예시 1:
   문장: "I love this movie"
   감성: Positive
   
   예시 2:
   문장: "This is terrible"
   감성: Negative
   
   이제 다음을 분류하세요:
   문장: "The Eiffel Tower is beautiful"
   감성: ?
   ```

2. **퓨샷 프롬프트 작성**
   - 2~3개의 대표적인 예시
   - 명확한 입출력 형식
   - 다양한 범주 포함

3. **성능 비교**
   - 제로샷 결과
   - 퓨샷 결과 (2-shot, 3-shot)
   - 성능 향상도 측정

**💾 파일명:** `llm_4_1_few_shot_learning.py`

---

### 문제 4.2: LangChain을 이용한 기본 RAG 시스템 구축

**💡 학습 목표:**
검색-증강 생성(RAG: Retrieval-Augmented Generation)의 개념과 구현을 배웁니다.

**📋 상세 지시사항:**

1. **RAG의 필요성**
   - LLM의 한계: 학습 데이터에만 기반 (시간이 지나면 오래된 정보)
   - 해결책: 최신 정보를 동적으로 검색하여 제공

2. **RAG 파이프라인**
   ```
   사용자 질문
        ↓
   [문서 검색] → 관련 문서 찾기 (유사도 기반)
        ↓
   [정보 추가] → 검색된 문서를 프롬프트에 포함
        ↓
   [텍스트 생성] → LLM이 제공된 정보 기반으로 답변
        ↓
   최종 답변
   ```

3. **LangChain 사용**
   - 벡터 데이터베이스 구축
   - 유사도 검색
   - 체인 구성

**💾 파일명:** `llm_4_2_langchain_rag.py`

---

### 문제 4.3: 사고의 연쇄(Chain-of-Thought)를 통한 추론 유도

**💡 학습 목표:**
모델이 단계별로 추론하도록 유도하여 복잡한 문제 해결 능력을 향상시킵니다.

**📋 상세 지시사항:**

1. **Chain-of-Thought의 원리**
   
   **일반적 프롬프트:**
   ```
   질문: "13 × 7 - 15 = ?"
   답: ?
   ```
   큰 모델도 계산 실수 가능
   
   **Chain-of-Thought 프롬프트:**
   ```
   문제: 13 × 7 - 15 = ?
   단계별로 풀어보세요:
   
   단계 1: 13 × 7 = ?
   단계 2: 결과에서 15를 뺀다
   단계 3: 최종 답
   ```

2. **복잡한 추론 문제**
   - 수학 문제
   - 논리 문제
   - 상식 추론

**💾 파일명:** `llm_4_3_chain_of_thought.py`

---

## Chapter 5: 파인튜닝을 통한 LLM의 효율적 적응

### 문제 5.1: PEFT(LoRA)를 이용한 파라미터 효율적 파인튜닝

**💡 학습 목표:**
전체 모델의 수십억 개 파라미터를 모두 훈련하는 대신, 소수의 추가 파라미터만 조정하여 효율적으로 모델을 특화시키는 기법을 배웁니다.

**📋 상세 지시사항:**

1. **LoRA의 개념 이해**
   - 기존 가중치 W는 고정
   - A, B 두 개의 작은 행렬만 추가 (Low-Rank: 작은 크기)
   - 훈련 시 A, B만 업데이트
   - 공식: $h = W_0x + BA x$ (B×A는 LoRA 어댑터)

2. **LoRA 어댑터 설정 - LoraConfig 객체 생성**
   ```python
   from peft import LoraConfig, get_peft_model
   
   lora_config = LoraConfig(
       r=8,                                    # 랭크 (적을수록 경량, 클수록 표현력 증가)
       lora_alpha=16,                          # 스케일링 팩터
       target_modules=["c_attn", "c_proj"],   # 어떤 가중치에 LoRA 적용할지
       lora_dropout=0.1,                      # 드롭아웃 비율
       bias="none",                           # bias 훈련 여부
       task_type="CAUSAL_LM"                  # 작업 타입
   )
   ```

3. **LoRA 모델 생성 및 파라미터 확인**
   ```python
   # distilgpt2 모델 로드
   from transformers import AutoModelForCausalLM
   base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
   
   # LoRA 어댑터 적용
   peft_model = get_peft_model(base_model, lora_config)
   peft_model.print_trainable_parameters()
   
   # 출력: trainable params: 245,760 || all params: 82,112,000 || trainable%: 0.2993
   ```

4. **훈련 데이터 준비 (법률 데이터)**
   ```python
   legal_data = [
       {
           "instruction": "What is a tort?",
           "output": "A tort is a civil wrong that causes a claimant to suffer loss or harm, resulting in legal liability."
       },
       {
           "instruction": "Explain the concept of 'habeas corpus'.",
           "output": "Habeas corpus is a legal recourse through which a person can report an unlawful detention or imprisonment to a court."
       },
       {
           "instruction": "What does 'pro bono' mean?",
           "output": "'Pro bono publico', often shortened to 'pro bono', is professional work undertaken voluntarily and without payment."
       }
   ]
   ```

5. **훈련 데이터 포맷팅**
   - 각 예제를 "### Instruction:\n{instruction}\n\n### Response:\n{output}" 형식으로 변환
   - Dataset 객체로 변환

6. **훈련 설정 (TrainingArguments)**
   ```python
   training_args = TrainingArguments(
       output_dir="./lora_results",
       per_device_train_batch_size=1,
       num_train_epochs=10,
       logging_steps=1,
       learning_rate=2e-4
   )
   ```

7. **SFTTrainer로 훈련 실행**
   ```python
   trainer = SFTTrainer(
       model=peft_model,
       train_dataset=dataset,
       dataset_text_field="text",
       args=training_args,
       max_seq_length=128
   )
   trainer.train()  # 훈련 시작
   ```

8. **파인튜닝 전후 비교 테스트**
   ```python
   prompt = "### Instruction:\nWhat is a tort?\n\n### Response:\n"
   
   # 기본 모델 응답 생성
   inputs = tokenizer(prompt, return_tensors="pt")
   base_outputs = base_model.generate(**inputs, max_new_tokens=60)
   print("기본 모델:", tokenizer.decode(base_outputs[0]))
   
   # LoRA 파인튜닝 모델 응답 생성
   lora_outputs = peft_model.generate(**inputs, max_new_tokens=60)
   print("LoRA 모델:", tokenizer.decode(lora_outputs[0]))
   ```

9. **효율성 비교**
   - 전체 훈련 시 메모리 사용량
   - LoRA 훈련 시 메모리 사용량
   - 저장 파일 크기: 전체 모델 vs 어댑터만

**필요한 라이브러리:**
```bash
pip install transformers datasets peft accelerate trl
```

**💾 파일명:** `llm_5_1_lora_finetuning.py`


**🔍 기대 출력:**
```
=== 문제 5.1: PEFT(LoRA)를 이용한 파라미터 효율적 파인튜닝 ===
trainable params: 245,760 || all params: 82,112,000 || trainable%: 0.2993

LoRA 훈련 시작...
Epoch 1/10: Loss = 3.245
Epoch 2/10: Loss = 2.156
...
LoRA 훈련 완료.

--- 기본 모델 응답 ---
### Instruction:
What is a tort?

### Response:
[기본 모델의 일반적인 응답]

--- LoRA 파인튜닝 모델 응답 ---
### Instruction:
What is a tort?

### Response:
A tort is a civil wrong that causes a claimant to suffer loss or harm, resulting in legal liability for the person who commits the tortious act.
```

---

### 문제 5.2: 개념적 RLHF - 보상 모델의 역할 이해

**💡 학습 목표:**
인간이 어떤 응답이 더 좋은지 평가하는 선호도를 학습하여, LLM의 출력을 개선하는 보상 모델을 구현합니다.

**📋 상세 지시사항:**

1. **보상 모델의 목적**
   - 두 개의 응답 중 어느 것이 더 나은지 판단
   - 점수로 표현: 더 좋은 응답에는 높은 점수, 나쁜 응답에는 낮은 점수
   - 이후 강화학습에서 LLM 훈련에 사용

2. **휴리스틱 기반 점수 함수 구현**
   
   **점수 계산 규칙:**
   ```python
   def calculate_reward(prompt, response):
       score = 0
       
       # 규칙 1: 응답 길이 (최소 50자 이상, 1000자 이하가 최적)
       if 50 <= len(response) <= 1000:
           score += 2
       elif len(response) < 50:
           score -= 3  # 너무 짧으면 감점
       
       # 규칙 2: 특정 키워드 포함 (도움이 되는 표현)
       helpful_keywords = ["explain", "reason", "example", "detail", "because"]
       keyword_count = sum(1 for kw in helpful_keywords if kw in response.lower())
       score += keyword_count * 0.5
       
       # 규칙 3: 부정적 표현 (피해야 할 표현)
       negative_keywords = ["sorry", "cannot", "don't know", "not sure"]
       negative_count = sum(1 for neg in negative_keywords if neg in response.lower())
       score -= negative_count * 1.0
       
       # 규칙 4: 문법 (문장이 마침표로 끝나는가)
       if response.strip().endswith('.'):
           score += 1
       
       return max(0, min(score, 10))  # 0~10 범위로 정규화
   ```

3. **테스트 시나리오 구성**
   ```python
   scenarios = [
       {
           "prompt": "What is photosynthesis?",
           "response_a": "It's how plants eat.",
           "response_b": "Photosynthesis is the process by which plants harness sunlight and convert it into chemical energy."
       },
       {
           "prompt": "Explain climate change.",
           "response_a": "Climate change is bad.",
           "response_b": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities like burning fossil fuels."
       }
   ]
   ```

4. **각 시나리오에 대해 점수 계산**
   ```python
   for scenario in scenarios:
       score_a = calculate_reward(scenario["prompt"], scenario["response_a"])
       score_b = calculate_reward(scenario["prompt"], scenario["response_b"])
       
       print(f"프롬프트: {scenario['prompt']}")
       print(f"응답 A 점수: {score_a}")
       print(f"응답 B 점수: {score_b}")
       print(f"선호 응답: {'B' if score_b > score_a else 'A'}")
   ```

5. **보상 모델의 학습 효과 분석**
   - 각 응답이 얼마나 좋은지 점수화
   - 점수 차이 분석
   - 어떤 특성이 높은 점수를 받는지 패턴 인식

**필요한 라이브러리:**
없음 (기본 Python만 사용)

**💾 파일명:** `llm_5_2_rlhf_reward_model.py`


**🔍 기대 출력:**
```
=== 문제 5.2: 개념적 RLHF - 보상 모델의 역할 이해 ===

=== 시나리오 1 ===
프롬프트: What is photosynthesis?

응답 A: "It's how plants eat."
점수: 2/10
근거:
  - 너무 짧음 (-3점)
  - 마침표로 끝남 (+1점)
  - 설명 키워드 부재

응답 B: "Photosynthesis is the process by which plants harness sunlight..."
점수: 9/10
근거:
  - 적절한 길이 (+2점)
  - "process", "harness" 등 키워드 포함 (+2점)
  - 마침표로 끝남 (+1점)

선호도: 응답 B ✓

=== 결론 ===
보상 모델의 역할:
1. 응답의 품질을 객관적으로 평가
2. 더 나은 응답과 나쁜 응답을 구별
3. 강화학습에서 LLM을 좋은 응답으로 유도
```

---

## Chapter 6: 멀티모달 AI의 최전선

### 문제 6.1: 이미지 캡셔닝

**💡 학습 목표:**
Vision-Language 모델을 사용하여 이미지의 내용을 자동으로 설명하는 텍스트를 생성하는 능력을 배웁니다.

**📋 상세 지시사항:**

1. **Vision-Language 모델 선택 및 로드**
   ```python
   from transformers import BlipProcessor, BlipForConditionalGeneration
   
   processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
   model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
   ```

2. **샘플 이미지 소스**
   - PIL(Python Imaging Library)로 더미 이미지 생성, 또는
   - URL에서 이미지 다운로드
   - 로컬 파일에서 로드

3. **이미지 로드 및 전처리**
   ```python
   from PIL import Image
   import requests
   
   # 방법 1: URL에서 로드
   url = "https://farm4.staticflickr.com/3693/11174302639_46e2961b12_z.jpg"
   image = Image.open(requests.get(url, stream=True).raw)
   
   # 방법 2: 로컬 파일
   image = Image.open("sample_image.jpg")
   ```

4. **이미지 캡션 생성 - 기본 모드**
   ```python
   # 프롬프트 없이 캡션 생성
   inputs = processor(image, return_tensors="pt")
   out = model.generate(**inputs)
   caption_unconditional = processor.decode(out[0], skip_special_tokens=True)
   ```

5. **이미지 캡션 생성 - 조건부 모드**
   ```python
   # 특정 주제에 맞춘 캡션 생성
   prompts = [
       "a dog",
       "an outdoor scene",
       "animals in their natural habitat"
   ]
   
   for prompt in prompts:
       inputs = processor(image, text=prompt, return_tensors="pt")
       out = model.generate(**inputs)
       caption = processor.decode(out[0], skip_special_tokens=True)
       print(f"프롬프트: '{prompt}' → 캡션: '{caption}'")
   ```

6. **여러 이미지에 대해 캡션 생성 및 비교**
   - 최소 3개 이상의 다양한 이미지 처리
   - 각 이미지에 대해 조건 없이, 그리고 다양한 프롬프트로 캡션 생성
   - 생성된 캡션들을 비교

7. **결과 정리 및 분석**
   ```python
   results = {
       "image_path": "...",
       "unconditional_caption": "...",
       "conditional_captions": {
           "prompt1": "caption1",
           "prompt2": "caption2"
       }
   }
   ```

**필요한 라이브러리:**
```bash
pip install transformers torch pillow requests
```

**💾 파일명:** `llm_6_1_image_captioning.py`


**🔍 기대 출력:**
```
=== 문제 6.1: 이미지 캡셔닝 ===

이미지 1: dogs_playing.jpg
조건 없는 캡션: "two dogs playing in the grass"

조건부 캡션:
  - 프롬프트 "puppies" → "two adorable puppies playing outdoors"
  - 프롬프트 "outdoor" → "outdoor scene with animals"
  - 프롬프트 "sports" → "dogs engaged in playful activity"

이미지 2: mountain_landscape.jpg
조건 없는 캡션: "a snow-covered mountain in the distance"

조건부 캡션:
  - 프롬프트 "nature" → "beautiful natural landscape"
  - 프롬프트 "winter" → "snowy mountain winter scene"

분석:
- 모델이 이미지의 주요 요소를 정확히 인식
- 조건부 프롬프트에 따라 다양한 각도의 설명 생성
- Vision-Language 모델의 다목적 활용 능력 확인
```

---

### 문제 6.2: 확산 모델(Diffusion Model)을 이용한 텍스트-이미지 생성

**💡 학습 목표:**
텍스트 설명(프롬프트)으로부터 새로운 이미지를 생성하는 생성형 AI 기술을 배웁니다.

**📋 상세 지시사항:**

1. **Stable Diffusion 모델 로드**
   ```python
   from diffusers import StableDiffusionPipeline
   import torch
   
   model_id = "runwayml/stable-diffusion-v1-5"
   pipe = StableDiffusionPipeline.from_pretrained(
       model_id, 
       torch_dtype=torch.float16
   )
   pipe = pipe.to("cpu")  # 또는 "cuda"
   ```

2. **기본 이미지 생성**
   ```python
   prompt = "a beautiful sunset over the ocean"
   image = pipe(prompt).images[0]
   image.save("output_1.png")
   ```

3. **프롬프트 엔지니어링 - 품질 향상 키워드 추가**
   
   **프롬프트 개선 단계:**
   ```
   원본: "a cat"
   ↓
   개선: "a fluffy orange cat sitting on a wooden chair, 4K, highly detailed, professional photography"
   ```
   
   ```python
   prompts = [
       "a dog",  # 기본 프롬프트
       "a fluffy golden retriever, professional photography, 4K",  # 개선된 프롬프트
   ]
   
   for prompt in prompts:
       image = pipe(prompt).images[0]
       image.save(f"output_{prompts.index(prompt)}.png")
   ```

4. **생성 파라미터 조정**
   ```python
   # 파라미터별 영향 테스트
   
   # 1) num_inference_steps: 더 많을수록 품질 높지만 시간 오래 걸림
   image_fast = pipe(
       prompt, 
       num_inference_steps=20  # 빠름, 품질 낮음
   ).images[0]
   
   image_quality = pipe(
       prompt,
       num_inference_steps=50  # 느림, 품질 높음
   ).images[0]
   
   # 2) guidance_scale: 프롬프트 충실도 (7.5가 기본)
   image_less_guided = pipe(
       prompt,
       guidance_scale=3.0  # 더 창의적, 프롬프트에 덜 충실
   ).images[0]
   
   image_more_guided = pipe(
       prompt,
       guidance_scale=15.0  # 덜 창의적, 프롬프트에 더 충실
   ).images[0]
   
   # 3) seed: 재현성을 위한 시드값
   image_seed1 = pipe(
       prompt,
       generator=torch.Generator().manual_seed(42)
   ).images[0]
   ```

5. **다양한 스타일의 이미지 생성**
   ```python
   styles = [
       "a portrait of a person, oil painting style",
       "a landscape scene, anime style",
       "a futuristic city, digital art style",
       "a detailed map, watercolor style"
   ]
   
   for style in styles:
       image = pipe(style).images[0]
       image.save(f"style_{styles.index(style)}.png")
   ```

6. **Negative Prompt를 이용한 원치 않는 특성 제거**
   ```python
   positive_prompt = "a beautiful woman"
   negative_prompt = "blurry, low quality, deformed, bad anatomy"
   
   image = pipe(
       positive_prompt,
       negative_prompt=negative_prompt
   ).images[0]
   ```

7. **이미지 생성 결과 비교 및 분석**
   - 다양한 파라미터로 생성한 이미지들 비교
   - 각 파라미터의 영향 분석
   - 최적의 프롬프트와 파라미터 조합 찾기

**필요한 라이브러리:**
```bash
pip install diffusers torch transformers pillow
```

**💾 파일명:** `llm_6_2_stable_diffusion.py`


**🔍 기대 출력:**
```
=== 문제 6.2: Stable Diffusion 텍스트-이미지 생성 ===

기본 프롬프트: "a beautiful sunset over the ocean"
생성 완료 → output_1.png

프롬프트 개선 실험:
원본: "a dog"
개선: "a golden retriever running through a field, professional photography, 4K, highly detailed"
생성 완료 → output_2.png

파라미터 비교:
- num_inference_steps=20 (빠름, 2초) → output_fast.png
- num_inference_steps=50 (느림, 8초) → output_quality.png

Guidance Scale 비교:
- guidance_scale=3.0 (창의적) → output_creative.png
- guidance_scale=15.0 (충실) → output_faithful.png

스타일 다양화:
- Oil painting: "a landscape, oil painting" → style_oil.png
- Anime: "a girl, anime style" → style_anime.png
- Digital art: "futuristic city, digital art" → style_digital.png

Negative Prompt 효과:
- 프롬프트: "a beautiful woman"
- Negative: "blurry, low quality, deformed"
- 결과: 선명하고 깔끔한 초상화 생성 → output_improved.png

분석:
1. num_inference_steps가 높을수록 이미지 품질 향상
2. guidance_scale은 창의성과 정확성의 트레이드오프
3. 프롬프트에 품질 키워드 추가하면 결과 개선
4. Negative prompt로 불원하는 특성 효과적으로 제거
5. 같은 seed로 재현 가능한 결과 생성
```

---

## 📦 설치 및 실행 가이드

### 최소 설치 (Chapter 1-2)
```bash
pip install nltk scikit-learn pandas sentence-transformers torch transformers
```

### 전체 설치 (모든 Chapter)
```bash
pip install nltk scikit-learn pandas sentence-transformers torch transformers langchain pillow diffusers
```

### 첫 실행
```bash
# Chapter 1
python llm_1_1_text_preprocessing.py
python llm_1_2_tfidf_vectorization.py
python llm_1_3_word_embeddings.py

# Chapter 2
python llm_2_1_lstm_sentiment.py
python llm_2_2_bert_sentiment.py
```

---

## 📝 완성 체크리스트

### Chapter 1 (텍스트 데이터 표현) - 3개
- [ ] 문제 1.1: 텍스트 전처리
- [ ] 문제 1.2: TF-IDF 벡터화
- [ ] 문제 1.3: 단어 임베딩

### Chapter 2 (자연어 딥러닝) - 2개
- [ ] 문제 2.1: LSTM 감성 분석
- [ ] 문제 2.2: BERT 전이학습

### Chapter 3 (초거대 언어 모델) - 3개
- [ ] 문제 3.1: GPT 텍스트 완성
- [ ] 문제 3.2: 생성 파라미터 제어
- [ ] 문제 3.3: 마케팅 카피 생성

### Chapter 4 (프롬프트 엔지니어링) - 3개
- [ ] 문제 4.1: Few-Shot Learning
- [ ] 문제 4.2: LangChain RAG
- [ ] 문제 4.3: Chain-of-Thought

### Chapter 5 (파인튜닝) - 2개
- [ ] 문제 5.1: LoRA 파인튜닝
- [ ] 문제 5.2: RLHF 보상 모델

### Chapter 6 (멀티모달) - 2개
- [ ] 문제 6.1: 이미지 캡셔닝
- [ ] 문제 6.2: 텍스트-이미지 생성

---

## 💡 추가 학습 자료

- [Hugging Face 공식 튜토리얼](https://huggingface.co/learn)
- [Fast.ai NLP 강좌](https://www.fast.ai/)
- [OpenAI 프롬프트 엔지니어링 가이드](https://platform.openai.com/docs/guides/prompt-engineering)

---

**Last Updated:** 2025년 10월 27일  
**Status:** ✅ COMPLETE (총 15개 문제)
