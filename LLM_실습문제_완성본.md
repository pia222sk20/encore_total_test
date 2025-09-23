# LLM 실습 문제 완성본

## 목차
- [Chapter 1: 텍스트 데이터의 표현](#chapter-1-텍스트-데이터의-표현)
  - [문제 1.1: 텍스트 전처리 파이프라인 구축](#문제-11-텍스트-전처리-파이프라인-구축)
  - [문제 1.2: 텍스트에서 벡터로 - TF-IDF](#문제-12-텍스트에서-벡터로---tf-idf)
  - [문제 1.3: 시맨틱 능력의 발현 - 단어 임베딩 비교](#문제-13-시맨틱-능력의-발현---단어-임베딩-비교)
- [Chapter 2: 자연어 딥러닝의 핵심 개념](#chapter-2-자연어-딥러닝의-핵심-개념)
  - [문제 2.1: LSTM을 이용한 순차 데이터 모델링 및 감성 분석](#문제-21-lstm을-이용한-순차-데이터-모델링-및-감성-분석)
  - [문제 2.2: BERT를 활용한 전이학습의 위력](#문제-22-bert를-활용한-전이학습의-위력)
- [Chapter 3: 초거대 언어 모델의 실제적 활용](#chapter-3-초거대-언어-모델의-실제적-활용)
  - [문제 3.1: GPT를 이용한 텍스트 완성](#문제-31-gpt를-이용한-텍스트-완성)
  - [문제 3.2: 창의성 제어하기 - 생성 파라미터의 영향](#문제-32-창의성-제어하기---생성-파라미터의-영향)
  - [문제 3.3: 간단한 도메인 응용 - 마케팅 카피 생성기](#문제-33-간단한-도메인-응용---마케팅-카피-생성기)
- [Chapter 4: 고급 프롬프트 엔지니어링 기법 마스터](#chapter-4-고급-프롬프트-엔지니어링-기법-마스터)
  - [문제 4.1: 퓨샷(Few-Shot) 인-컨텍스트 러닝](#문제-41-퓨샷few-shot-인-컨텍스트-러닝)
  - [문제 4.2: LangChain을 이용한 기본 RAG 시스템 구축](#문제-42-langchain을-이용한-기본-rag-시스템-구축)
  - [문제 4.3: 사고의 연쇄(Chain-of-Thought)를 통한 추론 유도](#문제-43-사고의-연쇄chain-of-thought를-통한-추론-유도)
- [Chapter 5: 파인튜닝을 통한 LLM의 효율적 적응](#chapter-5-파인튜닝을-통한-llm의-효율적-적응)
  - [문제 5.1: PEFT(LoRA)를 이용한 파라미터 효율적 파인튜닝](#문제-51-peftlora를-이용한-파라미터-효율적-파인튜닝)
  - [문제 5.2: 개념적 RLHF - 보상 모델의 역할 이해](#문제-52-개념적-rlhf---보상-모델의-역할-이해)
- [Chapter 6: 멀티모달 AI의 최전선 탐험](#chapter-6-멀티모달-ai의-최전선-탐험)
  - [문제 6.1: 이미지 캡셔닝](#문제-61-이미지-캡셔닝)
  - [문제 6.2: 확산 모델(Diffusion Model)을 이용한 텍스트-이미지 생성](#문제-62-확산-모델diffusion-model을-이용한-텍스트-이미지-생성)

---

## Chapter 1: 텍스트 데이터의 표현

### 문제 1.1: 텍스트 전처리 파이프라인 구축

- **문제:** 원시 텍스트를 기계가 이해할 수 있는 형태로 가공하는 기본적인 전처리 과정을 구현합니다.
- **지시어:** 구두점, 대소문자, 불용어(stopwords)가 포함된 원시 텍스트 문장 리스트가 주어졌을 때, NLTK 라이브러리를 사용하여 토큰화, 소문자 변환, 구두점 제거, 불용어 제거를 수행하는 Python 함수를 작성하세요.
- **제공데이터:**
  ```python
  raw_texts = [
      "Hello everyone, this is the first document for our NLP task!",
      "We are learning about Natural Language Processing, which is very exciting.",
      "Preprocessing text is an important and fundamental step."
  ]
  ```
- **필요한 라이브러리:**
  ```bash
  pip install nltk
  ```
- **소스코드 내용:**
  ```python
  import nltk
  from nltk.corpus import stopwords
  from nltk.tokenize import word_tokenize
  import string

  def download_nltk_data():
      """NLTK 데이터 다운로드 (최초 1회 실행 필요)"""
      try:
          stopwords.words('english')
      except LookupError:
          print("Downloading NLTK stopwords...")
          nltk.download('stopwords')
      
      try:
          nltk.data.find('tokenizers/punkt')
      except LookupError:
          print("Downloading NLTK punkt tokenizer...")
          nltk.download('punkt')

  def preprocess_text(text_list):
      """
      텍스트 리스트를 받아 전처리 파이프라인을 적용하는 함수.
      :param text_list: 원시 텍스트 문자열의 리스트
      :return: 전처리된 토큰들의 리스트 (문장별로 나뉨)
      """
      stop_words = set(stopwords.words('english'))
      
      processed_texts = []
      analysis_details = []
      
      for text in text_list:
          details = {}
          details['original'] = text
          
          # 1. 소문자 변환
          lower_text = text.lower()
          details['lower'] = lower_text
          
          # 2. 토큰화
          tokens = word_tokenize(lower_text)
          details['tokenized'] = tokens
          
          # 3. 구두점 제거 및 불용어 제거
          processed_tokens = []
          removed_punctuations = []
          removed_stopwords = []
          
          for word in tokens:
              if word.isalpha():
                  if word not in stop_words:
                      processed_tokens.append(word)
                  else:
                      removed_stopwords.append(word)
              else:
                  if word in string.punctuation:
                      removed_punctuations.append(word)
          
          details['removed_punctuations'] = removed_punctuations
          details['removed_stopwords'] = removed_stopwords
          details['final'] = processed_tokens
          
          processed_texts.append(processed_tokens)
          analysis_details.append(details)
          
      return processed_texts, analysis_details

  def main():
      """메인 실행 함수"""
      print("=== 문제 1.1: 텍스트 전처리 파이프라인 구축 ===")
      
      # NLTK 데이터 다운로드
      download_nltk_data()
      
      raw_texts = [
          "Hello everyone, this is the first document for our NLP task!",
          "We are learning about Natural Language Processing, which is very exciting.",
          "Preprocessing text is an important and fundamental step."
      ]
      
      print("\n=== 원본 텍스트 ===")
      for i, text in enumerate(raw_texts):
          print(f"{i+1}. {text}")
          
      processed_data, details = preprocess_text(raw_texts)
      
      print("\n=== 전처리 결과 ===")
      for i, tokens in enumerate(processed_data):
          print(f"{i+1}. {tokens}")
          
      print("\n=== 전처리 과정 상세 분석 ===")
      total_original_words = 0
      total_processed_words = 0
      
      for i, detail in enumerate(details):
          print(f"\n문장 {i+1}: '{detail['original']}'")
          print(f"  1단계 (소문자): {detail['lower']}")
          print(f"  2단계 (토큰화): {detail['tokenized']}")
          if detail['removed_punctuations']:
              print(f"  제거된 구두점: {detail['removed_punctuations']}")
          if detail['removed_stopwords']:
              print(f"  제거된 불용어: {detail['removed_stopwords']}")
          print(f"  최종 결과: {detail['final']}")
          total_original_words += len(detail['tokenized'])
          total_processed_words += len(detail['final'])

      print("\n=== 통계 정보 ===")
      compression_rate = (total_original_words - total_processed_words) / total_original_words * 100
      unique_words = sorted(list(set(token for sentence in processed_data for token in sentence)))
      
      print(f"원본 총 단어 수: {total_original_words}")
      print(f"전처리 후 단어 수: {total_processed_words}")
      print(f"제거된 단어 수: {total_original_words - total_processed_words}")
      print(f"압축률: {compression_rate:.1f}%")
      print(f"고유 단어 수: {len(unique_words)}")
      print(f"어휘 사전: {unique_words}")

      print("\n=== 해설 ===")
      print("1. lower(): 대소문자 통일로 'JUMPS'와 'jumps'를 같은 단어로 인식")
      print("2. word_tokenize(): 문장을 단어 단위로 분리")
      print("3. isalpha(): 구두점과 숫자 제거")
      print("4. stopwords 제거: 'the', 'a', 'is' 등 빈번하지만 의미가 적은 단어 제거")
      print("5. 이 과정으로 분석에 불필요한 노이즈를 제거하고 핵심 단어만 추출")

  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_1_1_text_preprocessing.py`

---

### 문제 1.2: 텍스트에서 벡터로 - TF-IDF

- **문제:** 전처리된 텍스트를 컴퓨터가 계산할 수 있는 숫자 형태의 벡터로 변환합니다.
- **지시어:** 문제 1.1에서 전처리된 텍스트를 사용하여, scikit-learn의 TfidfVectorizer를 적용해 문장들을 TF-IDF 행렬로 변환하세요. 벡터라이저가 학습한 피처(단어) 이름들과 변환된 TF-IDF 행렬을 출력하여 텍스트가 어떻게 수치 데이터로 표현되는지 확인하세요.
- **제공데이터:** 문제 1.1의 `preprocessed_data` 결과 사용
- **필요한 라이브러리:**
  ```bash
  pip install scikit-learn pandas
  ```
- **소스코드 내용:**
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  import pandas as pd
  import numpy as np

  # 문제 1.1의 전처리된 데이터 (예시)
  preprocessed_data = [
      ['hello', 'everyone', 'first', 'document', 'nlp', 'task'],
      ['learning', 'natural', 'language', 'processing', 'exciting'],
      ['preprocessing', 'text', 'important', 'fundamental', 'step']
  ]

  def tfidf_vectorization(preprocessed_data):
      """
      전처리된 토큰 리스트를 받아 TF-IDF 행렬로 변환하는 함수.
      :param preprocessed_data: 토큰들의 리스트 (문장별로 나뉨)
      :return: TF-IDF 행렬(DataFrame), 피처 이름, 벡터라이저 객체
      """
      # TfidfVectorizer는 토큰화된 리스트가 아닌, 공백으로 구분된 문자열을 입력으로 받음
      corpus = [' '.join(tokens) for tokens in preprocessed_data]
      
      # TfidfVectorizer 객체 생성 및 학습
      vectorizer = TfidfVectorizer()
      tfidf_matrix = vectorizer.fit_transform(corpus)
      
      # 피처 이름 (단어 사전) 확인
      feature_names = vectorizer.get_feature_names_out()
      
      # TF-IDF 행렬을 DataFrame으로 변환
      df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), 
                              columns=feature_names, 
                              index=[f"문서 {i+1}" for i in range(len(corpus))])
      
      return df_tfidf, feature_names, vectorizer

  def main():
      """메인 실행 함수"""
      print("=== 문제 1.2: 텍스트에서 벡터로 - TF-IDF ===")
      
      df_tfidf, feature_names, vectorizer = tfidf_vectorization(preprocessed_data)
      
      print("\n=== 피처 이름 (어휘 사전) ===")
      print(feature_names)
      
      print("\n=== TF-IDF 행렬 ===")
      print(df_tfidf)
      
      print("\n=== TF-IDF 의미 분석 ===")
      print("TF-IDF는 '단어 빈도(TF)'와 '역문서 빈도(IDF)'의 곱입니다.")
      print("  - TF (Term Frequency): 한 문서 내에서 특정 단어가 얼마나 자주 등장하는가.")
      print("  - IDF (Inverse Document Frequency): 특정 단어가 전체 문서들 중 얼마나 희귀한가.")
      
      # 특정 단어의 IDF 값 확인
      idf_values = vectorizer.idf_
      idf_df = pd.DataFrame({'단어': feature_names, 'IDF 값': idf_values})
      print("\n--- 각 단어의 IDF 값 (희귀도) ---")
      print(idf_df)
      
      print("\n--- 결과 해석 ---")
      print("  - 'document'는 문서 1에만 등장하므로 높은 TF-IDF 값을 가집니다.")
      print("  - 모든 문서에 공통으로 등장하는 단어가 있다면, 그 단어의 IDF는 낮아져 TF-IDF 값도 낮아집니다.")
      print("  - 이 행렬은 각 문서를 16차원의 벡터로 표현한 것입니다.")

      # 새로운 문장 변환 테스트
      print("\n=== 새로운 문장 벡터화 테스트 ===")
      new_sentence = "A new exciting document about NLP"
      new_corpus = [' '.join(token for token in new_sentence.lower().split() if token.isalpha())]
      new_tfidf_matrix = vectorizer.transform(new_corpus)
      
      new_df_tfidf = pd.DataFrame(new_tfidf_matrix.toarray(), columns=feature_names, index=["새로운 문장"])
      print(f"테스트 문장: '{new_sentence}'")
      print("벡터화 결과:")
      print(new_df_tfidf)
      print("\n해석: 'exciting', 'document', 'nlp' 단어가 기존 어휘 사전에 있어 해당 차원에서 값을 가집니다.")

  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_1_2_tfidf_vectorization.py`

---

### 문제 1.3: 시맨틱 능력의 발현 - 단어 임베딩 비교

- **문제:** 단어의 '의미'를 벡터 공간에 표현하는 단어 임베딩의 개념을 이해합니다.
- **지시어:** `sentence-transformers` 라이브러리를 사용하여 사전 훈련된 언어 모델을 로드하세요. ("king", "queen") 쌍과 ("king", "man") 쌍에 대해 각각 코사인 유사도를 계산하고 비교하세요. 마지막으로, "king" - "man" + "woman" 벡터 연산을 수행한 결과가 "queen"의 벡터와 얼마나 유사한지 확인하여, 단어 임베딩이 어떻게 의미적 관계를 학습하는지 확인하세요.
- **제공데이터:** `sentence-transformers` 라이브러리에서 제공하는 사전 훈련 모델 (`all-MiniLM-L6-v2`)
- **필요한 라이브러리:**
  ```bash
  pip install sentence-transformers torch
  ```
- **소스코드 내용:**
  ```python
  try:
      import torch
      from sentence_transformers import SentenceTransformer, util
      SENTENCE_TRANSFORMERS_AVAILABLE = True
      print("Sentence-transformers 라이브러리가 사용 가능합니다.")
  except ImportError:
      SENTENCE_TRANSFORMERS_AVAILABLE = False
      print("Sentence-transformers 라이브러리가 설치되지 않았습니다.")

  import numpy as np

  class MockSentenceTransformer:
      """SentenceTransformer가 없을 때 사용할 모의 모델"""
      def __init__(self):
          self.model_name = "Mock Model"
          # 간단한 단어-벡터 매핑
          self.mock_embeddings = {
              'king': np.array([0.9, 0.1, 0.8, 0.2]),
              'queen': np.array([0.8, 0.2, 0.9, 0.1]),
              'man': np.array([0.7, 0.3, 0.1, 0.9]),
              'woman': np.array([0.6, 0.4, 0.2, 0.8]),
              'apple': np.array([0.1, 0.9, 0.3, 0.7]),
              'banana': np.array([0.2, 0.8, 0.4, 0.6])
          }
          print(f"{self.model_name}을(를) 사용합니다.")

      def encode(self, sentences, convert_to_tensor=False):
          embeddings = [self.mock_embeddings.get(s, np.random.rand(4)) for s in sentences]
          if convert_to_tensor:
              return torch.tensor(np.array(embeddings), dtype=torch.float32)
          return np.array(embeddings)

  class MockUtil:
      """sentence_transformers.util의 모의 클래스"""
      def cos_sim(self, tensor_a, tensor_b):
          # PyTorch 텐서를 NumPy 배열로 변환
          if isinstance(tensor_a, torch.Tensor):
              tensor_a = tensor_a.numpy()
          if isinstance(tensor_b, torch.Tensor):
              tensor_b = tensor_b.numpy()
          
          # 코사인 유사도 계산
          dot_product = np.dot(tensor_a, tensor_b)
          norm_a = np.linalg.norm(tensor_a)
          norm_b = np.linalg.norm(tensor_b)
          
          if norm_a == 0 or norm_b == 0:
              return torch.tensor([[0.0]])
              
          similarity = dot_product / (norm_a * norm_b)
          return torch.tensor([[similarity]])

  def main():
      """메인 실행 함수"""
      print("=== 문제 1.3: 시맨틱 능력의 발현 - 단어 임베딩 비교 ===")

      if SENTENCE_TRANSFORMERS_AVAILABLE:
          model = SentenceTransformer('all-MiniLM-L6-v2')
          st_util = util
          print("실제 SentenceTransformer 모델을 사용합니다.")
      else:
          model = MockSentenceTransformer()
          st_util = MockUtil()

      # 1. 단어 임베딩 생성
      words = ['king', 'queen', 'man', 'woman']
      word_embeddings = model.encode(words, convert_to_tensor=True)
      embeddings_dict = {word: emb for word, emb in zip(words, word_embeddings)}
      print(f"\n임베딩된 단어: {list(embeddings_dict.keys())}")
      print(f"임베딩 벡터 차원: {word_embeddings.shape[1]}")

      # 2. 코사인 유사도 비교
      print("\n--- 2. 의미적 유사도 비교 ---")
      cosine_kq = st_util.cos_sim(embeddings_dict['king'], embeddings_dict['queen'])
      cosine_km = st_util.cos_sim(embeddings_dict['king'], embeddings_dict['man'])
      cosine_kw = st_util.cos_sim(embeddings_dict['king'], embeddings_dict['woman'])
      
      print(f"유사도 ('king', 'queen'): {cosine_kq.item():.4f} (왕-여왕: 성별 관계)")
      print(f"유사도 ('king', 'man'):   {cosine_km.item():.4f} (왕-남자: 종류 관계)")
      print(f"유사도 ('king', 'woman'): {cosine_kw.item():.4f} (왕-여자: 관련성 낮음)")

      # 3. 벡터 연산을 통한 유추(Analogy) 테스트
      print("\n--- 3. 벡터 연산을 통한 단어 유추 ---")
      print("수행할 연산: king - man + woman ≈ queen")
      
      result_vector = embeddings_dict['king'] - embeddings_dict['man'] + embeddings_dict['woman']
      analogy_similarity = st_util.cos_sim(result_vector, embeddings_dict['queen'])
      
      print(f"\n연산 결과 벡터와 'queen' 벡터의 유사도: {analogy_similarity.item():.4f}")
      
      print("\n=== 해설 ===")
      print("1. 단어 임베딩은 단어의 '의미'를 다차원 벡터 공간에 표현합니다.")
      print("2. 코사인 유사도는 두 벡터가 가리키는 방향이 얼마나 유사한지를 측정하며, -1에서 1 사이의 값을 가집니다.")
      print("3. 'king'과 'queen'의 유사도가 'king'과 'man'의 유사도보다 높게 나타나는 것은, 모델이 '왕위'라는 핵심 속성을 더 중요하게 학습했음을 보여줍니다.")
      print("4. 'king - man + woman' 연산은 '왕' 벡터에서 '남성성'을 빼고 '여성성'을 더하는 것과 같습니다.")
      print("5. 이 결과가 'queen' 벡터와 높은 유사도를 보이는 것은, 임베딩 공간이 '성별'이라는 의미적 축을 성공적으로 학습했음을 의미합니다.")

  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_1_3_word_embeddings.py`

---

## Chapter 2: 자연어 딥러닝의 핵심 개념

### 문제 2.1: LSTM을 이용한 순차 데이터 모델링 및 감성 분석

- **문제:** 순차적인 데이터의 패턴을 학습하는 RNN의 한 종류인 LSTM을 이해하고, 이를 이용해 텍스트의 감성을 분석하는 모델을 구축합니다.
- **지시어:** PyTorch를 사용하여 간단한 LSTM 기반의 신경망을 구축하고, 주어진 영화 리뷰 데이터셋으로 감성 분류(긍정/부정) 모델을 훈련시키세요. 모델은 단어 인덱스의 시퀀스를 입력받아 이진 분류 결과를 출력해야 합니다.
- **제공데이터:**
  ```python
  train_data = {
      "This movie was fantastic and amazing": 1,
      "The acting was terrible and the story was boring": 0,
      "I really enjoyed the plot and the characters": 1,
      "A complete waste of time and money": 0,
      "The visuals were stunning, a true masterpiece": 1,
      "I would not recommend this film to anyone": 0
  }
  ```
- **필요한 라이브러리:**
  ```bash
  pip install torch
  ```
- **소스코드 내용:**
  ```python
  try:
      import torch
      import torch.nn as nn
      from torch.utils.data import Dataset, DataLoader
      PYTORCH_AVAILABLE = True
      print("PyTorch 라이브러리가 사용 가능합니다.")
  except ImportError:
      PYTORCH_AVAILABLE = False
      print("PyTorch 라이브러리가 설치되지 않았습니다.")

  from collections import Counter
  import numpy as np

  # --- Mock Objects for when PyTorch is not available ---
  class MockModule:
      def __init__(self):
          self.training = True
      def train(self):
          self.training = True
      def eval(self):
          self.training = False
      def __call__(self, *args, **kwargs):
          return self.forward(*args, **kwargs)

  class MockSentimentLSTM(MockModule):
      def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
          super().__init__()
          self.vocab_size = vocab_size
          self.positive_words = {'fantastic', 'amazing', 'enjoyed', 'stunning', 'masterpiece', 'good'}
          self.negative_words = {'terrible', 'boring', 'waste', 'not'}
          print("모의 LSTM 모델을 사용합니다.")

      def forward(self, x, int_to_word):
          # 입력 x는 정수 인코딩된 텐서
          scores = []
          for seq in x:
              score = 0
              words = [int_to_word.get(i.item(), '') for i in seq if i.item() != 0]
              for word in words:
                  if word in self.positive_words:
                      score += 1
                  if word in self.negative_words:
                      score -= 1
              
              # Sigmoid-like scaling
              prob = 1 / (1 + np.exp(-score))
              scores.append(prob)
          return torch.tensor(scores, dtype=torch.float32).view(-1, 1)

  def main():
      """메인 실행 함수"""
      print("=== 문제 2.1: LSTM을 이용한 순차 데이터 모델링 및 감성 분석 ===")
      
      train_data = {
          "This movie was fantastic and amazing": 1,
          "The acting was terrible and the story was boring": 0,
          "I really enjoyed the plot and the characters": 1,
          "A complete waste of time and money": 0,
          "The visuals were stunning, a true masterpiece": 1,
          "I would not recommend this film to anyone": 0
      }

      # --- 1. 데이터 준비 ---
      print("\n--- 1. 데이터 준비 ---")
      texts = list(train_data.keys())
      labels = list(train_data.values())

      words = ' '.join(texts).lower().split()
      word_counts = Counter(words)
      vocab = sorted(word_counts, key=word_counts.get, reverse=True)
      word_to_int = {word: i + 1 for i, word in enumerate(vocab)}
      int_to_word = {i: word for word, i in word_to_int.items()}
      vocab_size = len(word_to_int) + 1

      print(f"어휘 사전 크기: {len(vocab)}")
      
      text_ints = [[word_to_int[word] for word in text.lower().split()] for text in texts]
      
      seq_length = max(len(x) for x in text_ints)
      features = torch.zeros((len(text_ints), seq_length), dtype=torch.long)
      for i, row in enumerate(text_ints):
          features[i, -len(row):] = torch.tensor(row)[:seq_length]
      
      labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
      print(f"데이터가 {features.shape} 모양의 텐서로 변환되었습니다.")

      # --- 2. 모델 정의 및 훈련 ---
      print("\n--- 2. 모델 훈련 ---")
      embedding_dim = 50
      hidden_dim = 64
      output_dim = 1
      n_layers = 2

      if PYTORCH_AVAILABLE:
          class SentimentLSTM(nn.Module):
              def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
                  super().__init__()
                  self.embedding = nn.Embedding(vocab_size, embedding_dim)
                  self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=0.5)
                  self.dropout = nn.Dropout(0.5)
                  self.fc = nn.Linear(hidden_dim, output_dim)
                  self.sigmoid = nn.Sigmoid()

              def forward(self, x):
                  embedded = self.embedding(x)
                  lstm_out, _ = self.lstm(embedded)
                  lstm_out = lstm_out[:, -1, :]
                  out = self.dropout(lstm_out)
                  out = self.fc(out)
                  sig_out = self.sigmoid(out)
                  return sig_out

          model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers)
          criterion = nn.BCELoss()
          optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
          
          epochs = 100
          for epoch in range(epochs):
              model.train()
              optimizer.zero_grad()
              output = model(features)
              loss = criterion(output, labels_tensor)
              loss.backward()
              optimizer.step()
              if (epoch + 1) % 20 == 0:
                  print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
      else:
          model = MockSentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers)
          print("PyTorch가 없어 실제 훈련을 건너뜁니다.")

      # --- 3. 모델 테스트 ---
      print("\n--- 3. 모델 테스트 ---")
      model.eval()
      
      test_texts = [
          "The movie was good and enjoyable",
          "The plot was predictable and dull"
      ]
      
      for test_text in test_texts:
          test_int = [word_to_int.get(word, 0) for word in test_text.lower().split()]
          test_feature = torch.zeros((1, seq_length), dtype=torch.long)
          test_feature[0, -len(test_int):] = torch.tensor(test_int)
          
          if PYTORCH_AVAILABLE:
              with torch.no_grad():
                  prediction = model(test_feature)
          else:
              prediction = model(test_feature, int_to_word)

          print(f"\n테스트 문장: '{test_text}'")
          print(f"예측 점수: {prediction.item():.4f}")
          print(f"예측된 감성: {'긍정' if prediction.item() > 0.5 else '부정'}")

      print("\n=== 해설 ===")
      print("1. 임베딩 레이어: 단어 인덱스를 밀집 벡터로 변환하여 단어의 의미를 학습합니다.")
      print("2. LSTM 레이어: 단어 벡터의 시퀀스를 처리하여 문장의 순차적 정보를 학습합니다. 과거의 정보를 '기억'하여 문맥을 파악합니다.")
      print("3. 완전 연결 레이어(FC): LSTM의 최종 출력을 받아 이진 분류(긍정/부정)를 위한 최종 점수를 계산합니다.")
      print("4. 시그모이드 함수: 최종 점수를 0과 1 사이의 확률 값으로 변환합니다.")

  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_2_1_lstm_sentiment.py`

---

### 문제 2.2: BERT를 활용한 전이학습의 위력

- **문제:** 대규모 데이터로 사전 훈련된(pre-trained) 모델을 사용하여, 적은 데이터와 간단한 코드로 높은 성능을 달성하는 전이학습(transfer learning)의 개념을 이해합니다.
- **지시어:** 문제 2.1과 동일한 감성 분류 작업을, 직접 모델을 구축하는 대신 Hugging Face `transformers` 라이브러리의 사전 훈련된 BERT 모델과 `pipeline` API를 사용하여 수행하세요. 코드의 간결성과 (개념적인) 성능을 LSTM 모델과 비교해보세요.
- **제공데이터:**
  ```python
  test_texts = [
      "I am absolutely thrilled with the results!",
      "This is the worst experience I have ever had.",
      "The movie was okay, but I probably wouldn't watch it again.",
      "Despite the long wait, the food was incredibly delicious.",
      "The weather today is sunny and pleasant.",
      "I'm feeling a bit down today.",
      "This document contains the quarterly financial report."
  ]
  ```
- **필요한 라이브러리:**
  ```bash
  pip install transformers torch
  ```
- **소스코드 내용:**
  ```python
  try:
      from transformers import pipeline
      TRANSFORMERS_AVAILABLE = True
      print("Transformers 라이브러리가 사용 가능합니다.")
  except ImportError:
      TRANSFORMERS_AVAILABLE = False
      print("Transformers 라이브러리가 설치되지 않았습니다.")

  import random

  class MockSentimentPipeline:
      """transformers.pipeline이 없을 때 사용할 모의 파이프라인"""
      def __init__(self):
          self.model_name = "Mock BERT"
          self.positive_keywords = ['thrilled', 'delicious', 'pleasant', 'fantastic', 'amazing', 'enjoyed']
          self.negative_keywords = ['worst', 'terrible', 'boring', 'down', 'waste']
          print(f"{self.model_name}을(를) 사용합니다.")

      def __call__(self, texts):
          results = []
          for text in texts:
              text_lower = text.lower()
              score = 0.5  # Neutral base
              pos_count = sum(1 for word in self.positive_keywords if word in text_lower)
              neg_count = sum(1 for word in self.negative_keywords if word in text_lower)
              
              if pos_count > neg_count:
                  label = 'POSITIVE'
                  score = min(0.99, 0.7 + pos_count * 0.1)
              elif neg_count > pos_count:
                  label = 'NEGATIVE'
                  score = min(0.99, 0.7 + neg_count * 0.1)
              else:
                  # Simple sentiment for neutral-like cases
                  if 'okay' in text_lower and "wouldn't" in text_lower:
                      label = 'NEGATIVE'
                      score = 0.85
                  else:
                      label = 'POSITIVE' if random.random() > 0.5 else 'NEGATIVE'
                      score = random.uniform(0.5, 0.7)
              
              results.append({'label': label, 'score': score})
          return results

  def main():
      """메인 실행 함수"""
      print("=== 문제 2.2: BERT를 활용한 전이학습의 위력 ===")
      
      test_texts = [
          "I am absolutely thrilled with the results!",
          "This is the worst experience I have ever had.",
          "The movie was okay, but I probably wouldn't watch it again.",
          "Despite the long wait, the food was incredibly delicious.",
          "The weather today is sunny and pleasant.",
          "I'm feeling a bit down today.",
          "This document contains the quarterly financial report.",
          "This movie was fantastic and amazing",
          "The acting was terrible and the story was boring",
          "I really enjoyed the plot and the characters",
          "A complete waste of time and money"
      ]

      print("\n=== BERT 감성 분석 결과 ===")
      if TRANSFORMERS_AVAILABLE:
          print("Hugging Face BERT 모델 로딩 중...")
          # 모델을 명시하지 않으면 기본 모델(distilbert-base-uncased-finetuned-sst-2-english)이 로드됨
          sentiment_pipeline = pipeline("sentiment-analysis", device=-1) # device=-1 for CPU
          print("모델 로딩 완료!")
          print("사용 모델: Real BERT")
      else:
          sentiment_pipeline = MockSentimentPipeline()

      results = sentiment_pipeline(test_texts)
      
      print("------------------------------------------------------------")
      total_score = 0
      positive_count = 0
      high_confidence_count = 0

      for text, result in zip(test_texts, results):
          label = result['label']
          score = result['score']
          total_score += score
          if label == 'POSITIVE':
              positive_count += 1
          if score > 0.8:
              high_confidence_count += 1
          
          print(f"Text: {text}")
          print(f"  -> Prediction: {label}")
          print(f"  -> Confidence: {score:.4f}")
          print(f"  -> Interpretation: {'High confidence' if score > 0.8 else 'Low confidence'}\n")

      print("=== 분석 통계 ===")
      print(f"총 분석 문장 수: {len(test_texts)}개")
      print(f"긍정 예측: {positive_count}개")
      print(f"부정 예측: {len(test_texts) - positive_count}개")
      print(f"고신뢰도 예측 (>0.8): {high_confidence_count}개")
      print(f"평균 신뢰도: {total_score / len(test_texts):.4f}")

      print("\n=== BERT vs LSTM 접근 방식 비교 ===")
      print(f"{'측면':<18}{'LSTM (직접 구현)':<30}{'BERT (Pipeline)':<20}")
      print("-" * 70)
      print(f"{'코드 복잡성':<18}{'높음 (모델 구조, 훈련 루프 등)':<30}{'매우 낮음 (몇 줄)':<20}")
      print(f"{'훈련 시간':<18}{'중간 (소규모 데이터)':<30}{'없음 (사전 훈련됨)':<20}")
      print(f"{'성능':<18}{'중간 (데이터에 따라)':<30}{'높음 (대규모 사전 훈련)':<20}")
      print(f"{'메모리 사용량':<18}{'낮음':<30}{'높음':<20}")
      print(f"{'전이학습':<18}{'제한적 (임베딩만)':<30}{'완전함 (전체 모델)':<20}")
      print(f"{'설명가능성':<18}{'상대적으로 높음':<30}{'낮음 (블랙박스)':<20}")

      print("\n=== 상세 비교 ===")
      print("\n1. 코드 복잡성:")
      print("   LSTM: 데이터 전처리, 모델 정의, 훈련 루프, 최적화 등 모든 과정을 직접 구현")
      print("   BERT: pipeline('sentiment-analysis') 한 줄로 완성")
      
      print("\n2. 성능:")
      print("   LSTM: 작은 데이터셋에서는 제한적, 도메인 특화 학습 가능")
      print("   BERT: 대규모 텍스트로 사전 훈련되어 일반적으로 우수한 성능")

      print("\n3. 실용성:")
      print("   LSTM: 연구용, 교육용, 특수한 요구사항이 있는 경우")
      print("   BERT: 실제 서비스, 빠른 프로토타이핑, 일반적인 NLP 작업")

      print("\n4. 전이학습:")
      print("   LSTM: 단어 임베딩 정도만 전이 가능")
      print("   BERT: 언어의 전반적 이해가 전이됨")

      print("\n=== BERT 모델 변형들 ===")
      print("BERT-base: 12층, 768차원, 110M 파라미터 - 원조 BERT")
      print("DistilBERT: 6층, 768차원, 66M 파라미터 - BERT 지식 증류, 60% 빠름")
      print("RoBERTa: BERT 개선버전 - 더 많은 데이터, 더 긴 훈련")
      print("ALBERT: 파라미터 공유로 경량화, 더 깊은 네트워크")
      print("DeBERTa: 분리된 어텐션으로 성능 개선")

      print("\n=== 감성 분석 특화 모델들 ===")
      print("cardiffnlp/twitter-roberta-base-sentiment: 트위터 데이터로 파인튜닝")
      print("nlptown/bert-base-multilingual-uncased-sentiment: 다국어 지원")
      print("ProsusAI/finbert: 금융 도메인 특화")

      print("\n=== 실용적인 사용 팁 ===")
      print("1. 빠른 프로토타이핑: pipeline API 사용")
      print("2. 높은 성능 필요: 특화된 모델 선택 또는 파인튜닝")
      print("3. 메모리 제약: DistilBERT 같은 경량 모델 사용")
      print("4. 특정 도메인: 도메인 특화 모델이나 자체 파인튜닝")
      print("5. 다국어: multilingual 모델 사용")
      print("6. 실시간 서비스: 모델 최적화 및 캐싱 전략 필요")

      print("\n=== 실제 서비스 고려사항 ===")
      print("• 모델 크기와 추론 속도의 트레이드오프")
      print("• GPU 메모리 요구사항")
      print("• 배치 처리로 효율성 향상")
      print("• 모델 버전 관리 및 업데이트 전략")
      print("• A/B 테스트를 통한 성능 검증")

  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_2_2_bert_sentiment.py`

---

## Chapter 3: 초거대 언어 모델의 실제적 활용

### 문제 3.1: GPT를 이용한 텍스트 완성

- **문제:** LLM의 가장 기본적인 능력인 '다음 단어 예측'을 활용하여, 주어진 문맥(프롬프트)에 이어지는 자연스러운 문장을 생성합니다.
- **지시어:** GPT 모델(Hugging Face `transformers`의 GPT-2 또는 유사한 모델)을 사용하여 간단한 텍스트 완성 작업을 수행하세요. 주어진 프롬프트에 대해 여러 개의 완성 결과를 생성하고 비교해보세요.
- **제공데이터:**
  ```python
  prompts = [
      "Once upon a time, in a distant land",
      "The future of artificial intelligence",
      "Starting a successful business requires",
      "Climate change is a global challenge that",
      "The most important lesson I learned was"
  ]
  ```
- **필요한 라이브러리:**
  ```bash
  pip install transformers torch
  ```
- **소스코드 내용:**
  ```python
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

      def get_template(self, prompt):
          prompt_lower = prompt.lower()
          if "business" in prompt_lower or "lesson" in prompt_lower:
              return self.business_templates
          elif "intelligence" in prompt_lower or "climate" in prompt_lower:
              return self.tech_templates
          else:
              return self.story_templates

      def generate(self, input_ids, max_length, num_return_sequences, **kwargs):
          # 실제 generate 메서드와 유사한 출력을 만들기 위해 리스트의 리스트를 반환
          prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
          templates = self.get_template(prompt_text)
          
          outputs = []
          for _ in range(num_return_sequences):
              completion = random.choice(templates)
              full_text = prompt_text + completion
              # max_length에 맞춰 텍스트 자르기
              encoded = tokenizer.encode(full_text, max_length=max_length, truncation=True)
              outputs.append(encoded)
          return torch.tensor(outputs)

  class MockTokenizer:
      """GPT 토크나이저 모의 객체"""
      def __init__(self):
          self.vocab = {'<|endoftext|>': 0}
          self.eos_token_id = 0
          self.pad_token_id = 0

      def encode(self, text, max_length=None, truncation=False):
          # 간단한 단어 기반 인코딩
          tokens = re.findall(r'\w+|\W+', text)
          encoded = [hash(token) % 1000 for token in tokens]
          if truncation and max_length and len(encoded) > max_length:
              encoded = encoded[:max_length]
          return encoded

      def decode(self, token_ids, skip_special_tokens=True):
          # 실제 디코딩은 불가능하므로, 이 예제에서는 사용되지 않음
          # generate 함수에서 직접 텍스트를 조합하므로, 이 함수는 형식만 맞춤
          return "Decoded text from mock tokenizer"

      def __call__(self, text, return_tensors=None):
          encoded = self.encode(text)
          if return_tensors == "pt":
              return {'input_ids': torch.tensor([encoded])}
          return {'input_ids': [encoded]}

  def demonstrate_text_completion(prompts, model, tokenizer, device):
      """주어진 프롬프트로 텍스트 완성을 시연하는 함수"""
      print("=== GPT 텍스트 완성 데모 ===")
      
      if TRANSFORMERS_AVAILABLE:
          print(f"GPT-2 모델 로딩 중...")
          model.to(device)
          print(f"Device set to use {device}")
          print("GPT-2 모델 로딩 완료!")
          print(f"사용 모델: Real GPT-2")
      else:
          print(f"사용 모델: {model.model_name}")
      
      print("------------------------------------------------------------")

      for i, prompt in enumerate(prompts):
          print(f"\n=== 프롬프트 {i+1}: {prompt} ===")
          inputs = tokenizer(prompt, return_tensors="pt")
          
          if TRANSFORMERS_AVAILABLE:
              inputs = {k: v.to(device) for k, v in inputs.items()}
              outputs = model.generate(
                  **inputs,
                  max_length=50,
                  num_return_sequences=3,
                  pad_token_id=tokenizer.eos_token_id,
                  no_repeat_ngram_size=2,
                  early_stopping=True
              )
              completions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
          else:
              # Mock 모델은 텍스트를 직접 반환
              completions = model.generate(prompt, num_return_sequences=3)

          for j, completion in enumerate(completions):
              print(f"\n완성 {j+1}:\n  {completion}")

  def explore_generation_parameters(model, tokenizer, device):
      """생성 파라미터의 영향을 탐색하는 함수"""
      print("\n=== 생성 파라미터 탐색 ===")
      prompt = "Artificial intelligence will"
      
      if TRANSFORMERS_AVAILABLE:
          print("실제 GPT-2 모델을 사용한 파라미터 실험:")
          inputs = tokenizer(prompt, return_tensors="pt")
          inputs = {k: v.to(device) for k, v in inputs.items()}

          params = {
              "Temperature = 0.1": {"temperature": 0.1, "top_k": 50},
              "Temperature = 0.7": {"temperature": 0.7, "top_k": 50},
              "Temperature = 1.2": {"temperature": 1.2, "top_k": 50},
          }

          for name, config in params.items():
              print(f"\n--- {name} ---")
              outputs = model.generate(
                  **inputs,
                  max_length=40,
                  num_return_sequences=2,
                  do_sample=True,
                  pad_token_id=tokenizer.eos_token_id,
                  **config
              )
              results = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
              for k, result in enumerate(results):
                  print(f"결과 {k+1}: {result.replace(prompt, '').strip()}")
      else:
          print("모의 모델을 사용한 파라미터 실험 (개념 설명):")
          print("\n--- Temperature = 0.1 (결정적) ---")
          print("  결과 1: create a new form of life that is more intelligent than humans.")
          print("  결과 2: create a new form of life that is more intelligent than humans.")
          print("\n--- Temperature = 0.7 (균형적) ---")
          print("  결과 1: change the world in profound ways, from healthcare to transportation.")
          print("  결과 2: automate many tasks, freeing up humans for more creative endeavors.")
          print("\n--- Temperature = 1.2 (창의적) ---")
          print("  결과 1: dream of electric sheep and write poetry about the stars.")
          print("  결과 2: eventually ask the question: what is the meaning of my existence?")

      print("\n=== 생성 파라미터 설명 ===")
      print("\nTEMPERATURE:")
      print("  설명: 생성의 랜덤성 조절")
      print("  낮은 값 (0.1-0.3): 더 예측 가능하고 보수적인 텍스트")
      print("  중간 값 (0.7-0.9): 창의적이면서도 일관성 있는 텍스트")
      print("  높은 값 (1.0+): 매우 창의적이지만 때로는 일관성 없는 텍스트")
      
      print("\nMAX_LENGTH:")
      print("  설명: 생성할 최대 토큰 수")
      print("  짧은 길이: 간결한 완성")
      print("  긴 길이: 상세한 완성")

      print("\nTOP_K:")
      print("  설명: 각 단계에서 고려할 상위 k개 토큰")
      print("  낮은 값: 더 예측 가능한 선택")
      print("  높은 값: 더 다양한 선택")

      print("\nTOP_P:")
      print("  설명: 누적 확률이 p에 도달할 때까지의 토큰들 고려")
      print("  낮은 값: 높은 확률 토큰들만 사용")
      print("  높은 값: 더 많은 토큰 후보 고려")

  def practical_applications(model, tokenizer, device):
      """창의적, 실용적 활용 예시를 보여주는 함수"""
      print("\n=== 창작 활용 예시 ===")
      
      scenarios = {
          "공상과학 장르": ("In the year 2150, humans discovered", 60),
          "미스터리 장르": ("The detective noticed something strange about the room", 60),
          "로맨스 장르": ("Their eyes met across the crowded coffee shop and", 60)
      }
      
      if TRANSFORMERS_AVAILABLE:
          generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if device=='cuda' else -1)
      else:
          # Mock pipeline
          def mock_generator(prompt, max_length):
              templates = model.get_template(prompt)
              return [{'generated_text': prompt + random.choice(templates)}]
          generator = mock_generator

      for genre, (prompt, max_len) in scenarios.items():
          print(f"\n--- {genre} ---")
          print(f"상황: {prompt.split(',')[0]}")
          print(f"프롬프트: {prompt}")
          
          if TRANSFORMERS_AVAILABLE:
              result = generator(prompt, max_length=max_len, num_return_sequences=1, no_repeat_ngram_size=2)
              print("생성 결과:\n ", result[0]['generated_text'])
          else:
              result = generator(prompt, max_length=max_len)
              print("생성 결과:\n ", result[0]['generated_text'])

      print("\n=== 실용적 활용 방안 ===")
      print("\n콘텐츠 제작:")
      print("  • 블로그 포스트 초안 작성")
      print("  • 소셜 미디어 캡션 생성")
      print("  • 제품 설명 자동 생성")
      print("  • 이메일 템플릿 작성")

      print("\n교육:")
      print("  • 예시 문장 생성")
      print("  • 문제 출제 도움")
      print("  • 학습 자료 보완")
      print("  • 언어 학습 연습")

      print("\n비즈니스:")
      print("  • 보고서 요약 초안")
      print("  • 제안서 아이디어 생성")
      print("  • 브레인스토밍 지원")
      print("  • 고객 응답 템플릿")

      print("\n개발:")
      print("  • 코드 주석 생성")
      print("  • 문서 작성 지원")
      print("  • API 설명 생성")
      print("  • 테스트 케이스 아이디어")

  def limitations_and_best_practices():
      """LLM의 한계점과 모범 사례를 설명하는 함수"""
      print("\n=== 한계점과 고려사항 ===")
      print("주요 한계점:")
      print("  • 사실성 검증 필요 - 생성된 내용이 항상 정확하지 않음")
      print("  • 편향성 - 훈련 데이터의 편향이 결과에 반영될 수 있음")
      print("  • 일관성 - 긴 텍스트에서 논리적 일관성 유지 어려움")
      print("  • 저작권 - 훈련 데이터와 유사한 내용 생성 가능성")
      print("  • 윤리적 사용 - 악의적 목적으로 오용될 수 있음")

      print("\n모범 사례:")
      print("  • 생성된 내용은 항상 검토하고 수정하기")
      print("  • 중요한 정보는 별도로 사실 확인하기")
      print("  • 다양한 프롬프트로 여러 결과 비교하기")
      print("  • 최종 사용자에게 AI 생성임을 명시하기")
      print("  • 윤리적 가이드라인 준수하기")

  def main():
      """메인 실행 함수"""
      print("=== 문제 3.1: GPT를 이용한 텍스트 완성 ===")
      
      global tokenizer
      if TRANSFORMERS_AVAILABLE:
          model = GPT2LMHeadModel.from_pretrained('gpt2')
          tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
          tokenizer.pad_token = tokenizer.eos_token
          device = "cuda" if torch.cuda.is_available() else "cpu"
      else:
          model = MockGPTModel()
          tokenizer = MockTokenizer()
          device = "cpu"

      prompts = [
          "Once upon a time, in a distant land",
          "The future of artificial intelligence",
          "Starting a successful business requires",
          "Climate change is a global challenge that",
          "The most important lesson I learned was"
      ]
      
      # 텍스트 완성 시연
      if TRANSFORMERS_AVAILABLE:
          demonstrate_text_completion_real(prompts, model, tokenizer, device)
      else:
          demonstrate_text_completion_mock(prompts, model, tokenizer)

      # 생성 파라미터 탐색
      explore_generation_parameters(model, tokenizer, device)

      # 창의적, 실용적 활용 예시
      practical_applications(model, tokenizer, device)

      # 한계점 및 모범 사례
      limitations_and_best_practices()

  def demonstrate_text_completion_real(prompts, model, tokenizer, device):
      """실제 모델을 사용한 텍스트 완성"""
      print("=== GPT 텍스트 완성 데모 ===")
      print(f"GPT-2 모델 로딩 중...")
      model.to(device)
      print(f"Device set to use {device}")
      print("GPT-2 모델 로딩 완료!")
      print(f"사용 모델: Real GPT-2")
      print("------------------------------------------------------------")

      for i, prompt in enumerate(prompts):
          print(f"\n=== 프롬프트 {i+1}: {prompt} ===")
          inputs = tokenizer(prompt, return_tensors="pt")
          inputs = {k: v.to(device) for k, v in inputs.items()}
          
          outputs = model.generate(
              **inputs,
              max_new_tokens=256,
              num_return_sequences=3,
              pad_token_id=tokenizer.eos_token_id,
              no_repeat_ngram_size=2,
              early_stopping=True
          )
          completions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

          for j, completion in enumerate(completions):
              print(f"\n완성 {j+1}:\n  {completion}")

  def demonstrate_text_completion_mock(prompts, model, tokenizer):
      """모의 모델을 사용한 텍스트 완성"""
      print("=== GPT 텍스트 완성 데모 ===")
      print(f"사용 모델: {model.model_name}")
      print("------------------------------------------------------------")

      for i, prompt in enumerate(prompts):
          print(f"\n=== 프롬프트 {i+1}: {prompt} ===")
          templates = model.get_template(prompt)
          for j in range(3):
              completion = random.choice(templates)
              full_text = prompt + completion
              print(f"\n완성 {j+1}:\n  {full_text}")

  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_3_1_gpt_text_completion.py`

---

### 문제 3.2: 창의성 제어하기 - 생성 파라미터의 영향

- **문제:** LLM의 출력 결과물의 스타일과 다양성을 조절하는 주요 파라미터들의 역할을 이해합니다.
- **지시어:** 문제 3.1과 동일한 모델과 프롬프트를 사용하되, `generate` 함수의 파라미터를 변경하며 여러 번 텍스트를 생성하세요. 다음 네 가지 경우의 결과를 비교하고 각 파라미터가 생성 결과에 미치는 영향을 분석하세요.
  1. 기본 설정 (Greedy Search)
  2. 높은 `temperature` (예: 1.5)
  3. 낮은 `temperature` (예: 0.5)
  4. `top_p` 샘플링 (Nucleus Sampling, 예: 0.90)
- **제공데이터:** 프롬프트 - `"The best way to learn a new programming language is"`
- **필요한 라이브러리:**
  ```bash
  pip install transformers torch
  ```
- **소스코드 내용:**
  ```python
  try:
      from transformers import pipeline
      import torch
      TRANSFORMERS_AVAILABLE = True
      print("Transformers 라이브러리가 사용 가능합니다.")
  except ImportError:
      TRANSFORMERS_AVAILABLE = False
      print("Transformers 라이브러리가 설치되지 않았습니다.")

  import random

  class MockGenerator:
      """생성 파라미터의 개념을 설명하기 위한 모의 생성기"""
      def __init__(self):
          self.prompt = "The best way to learn a new programming language is"
          self.base_completions = [
              " to build a project you are passionate about.",
              " by starting with the fundamentals and practicing consistently.",
              " to immerse yourself in its community and read others' code.",
              " by following a structured course and doing all the exercises."
          ]
          self.creative_completions = [
              " to dream in its syntax and think in its logic.",
              " to teach it to someone else, even if you are just a beginner.",
              " to fail, debug, and fail again, until it becomes second nature.",
              " to understand its philosophy, not just its commands."
          ]

      def generate(self, config_name, config):
          print(f"\n===== {config_name} =====")
          
          if "Greedy" in config_name:
              # 항상 가장 확률 높은 것만 선택
              print("  결과 1: " + self.prompt + self.base_completions[0])
              print("  결과 2: " + self.prompt + self.base_completions[0])
              print("  -> 해석: 가장 예측 가능한, 일관된 결과만 생성됩니다.")
          elif "Low Temperature" in config_name:
              # 예측 가능하지만 약간의 다양성 허용
              results = random.sample(self.base_completions, 2)
              print("  결과 1: " + self.prompt + results[0])
              print("  결과 2: " + self.prompt + results[1])
              print("  -> 해석: 안정적이고 논리적인 문장 내에서 약간의 다양성을 보입니다.")
          elif "High Temperature" in config_name:
              # 창의적이고 예상치 못한 결과
              results = random.sample(self.base_completions + self.creative_completions, 2)
              print("  결과 1: " + self.prompt + results[0])
              print("  결과 2: " + self.prompt + results[1])
              print("  -> 해석: 더 창의적이고 독특하지만, 때로는 문맥에서 벗어날 수 있습니다.")
          elif "Top-p" in config_name:
              # 확률 분포의 상위 p% 내에서만 샘플링
              results = random.sample(self.base_completions, 2)
              print("  결과 1: " + self.prompt + results[0])
              print("  결과 2: " + self.prompt + results[1])
              print("  -> 해석: Temperature와 유사하게 다양성을 조절하지만, 문맥에서 크게 벗어나는 것을 방지합니다.")

  def main():
      """메인 실행 함수"""
      print("=== 문제 3.2: 창의성 제어하기 - 생성 파라미터의 영향 ===")
      
      prompt = "The best way to learn a new programming language is"
      
      if TRANSFORMERS_AVAILABLE:
          print("\n=== 실제 모델을 사용한 파라미터별 생성 결과 비교 ===")
          generator = pipeline('text-generation', model='distilgpt2', device=-1)
          
          generation_configs = {
              "1. Default (Greedy Search)": {"max_length": 50, "num_return_sequences": 2},
              "2. High Temperature (1.5)": {"max_length": 50, "temperature": 1.5, "do_sample": True, "num_return_sequences": 2},
              "3. Low Temperature (0.5)": {"max_length": 50, "temperature": 0.5, "do_sample": True, "num_return_sequences": 2},
              "4. Top-p Sampling (0.90)": {"max_length": 50, "top_p": 0.90, "do_sample": True, "num_return_sequences": 2}
          }

          for name, config in generation_configs.items():
              print(f"\n===== {name} =====")
              config['pad_token_id'] = generator.tokenizer.eos_token_id
              responses = generator(prompt, **config)
              for i, response in enumerate(responses):
                  print(f"  결과 {i+1}: {response['generated_text']}")
      else:
          print("\n=== 모의 생성을 통한 파라미터 개념 설명 ===")
          mock_generator = MockGenerator()
          generation_configs = {
              "1. Default (Greedy Search)": {},
              "2. Low Temperature (0.5)": {},
              "3. High Temperature (1.5)": {},
              "4. Top-p Sampling (0.90)": {}
          }
          for name, config in generation_configs.items():
              mock_generator.generate(name, config)

      print("\n\n=== 파라미터 해설 ===")
      print("\n[ Temperature ]")
      print("  - 역할: 다음 단어 선택의 무작위성 조절. 확률 분포를 얼마나 '뾰족하게' 또는 '평평하게' 만들지 결정.")
      print("  - 낮은 값 (e.g., 0.2): 확률이 가장 높은 단어가 선택될 가능성이 매우 높음. -> 일관성 있고 예측 가능한 텍스트.")
      print("  - 높은 값 (e.g., 1.0+): 확률이 낮은 단어도 선택될 가능성이 생김. -> 창의적이고 다양한 텍스트, 때로는 비문 생성.")
      
      print("\n[ Top-k Sampling ]")
      print("  - 역할: 다음 단어 후보를 확률 순으로 상위 k개로 제한.")
      print("  - 낮은 값 (e.g., 10): 매우 한정된, 안전한 단어들 중에서만 선택.")
      print("  - 높은 값 (e.g., 100): 더 다양한 단어 후보군을 고려.")
      
      print("\n[ Top-p (Nucleus) Sampling ]")
      print("  - 역할: 다음 단어 후보들의 누적 확률이 p를 넘지 않는 최소한의 단어 집합으로 제한.")
      print("  - 예시 (p=0.9): 확률이 높은 순서대로 단어들의 확률을 더해가다가, 합이 0.9가 되는 지점까지만 후보로 인정.")
      print("  - 장점: 문맥에 따라 후보 단어의 개수가 동적으로 변함. Top-k보다 유연함.")
      
      print("\n[ do_sample=True ]")
      print("  - 역할: 이 옵션을 켜야 Temperature, Top-k, Top-p 등의 샘플링 기법이 활성화됨.")
      print("  - do_sample=False (기본값): Greedy Search. 항상 가장 확률 높은 단어만 선택.")

      print("\n=== 언제 어떤 파라미터를 사용할까? ===")
      print("  - **정보 요약, 번역, 사실 기반 질의응답:** 낮은 Temperature (0.1 ~ 0.4). 정확성이 중요.")
      print("  - **창의적인 글쓰기, 브레인스토밍, 대화형 챗봇:** 높은 Temperature (0.7 ~ 1.2) 또는 Top-p (0.9 ~ 0.95). 다양성과 창의성이 중요.")
      print("  - **균형 잡힌 결과:** 중간 Temperature (0.5 ~ 0.7) 또는 Top-k와 Top-p를 함께 사용.")

  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_3_2_generation_parameters.py`

---

### 문제 3.3: 간단한 도메인 응용 - 마케팅 카피 생성기

- **문제:** LLM을 특정 목적에 맞게 활용하는 방법을 배웁니다. 제품 정보를 입력하면 그럴듯한 마케팅 문구를 생성하는 간단한 애플리케이션을 만듭니다.
- **지시어:** 제품 이름과 핵심 기능 리스트를 입력받아, 이를 구조화된 프롬프트로 조합한 뒤 LLM을 사용하여 짧고 매력적인 마케팅 문구를 생성하는 Python 함수를 작성하세요.
- **제공데이터:**
  - `product_name = "QuantumLeap Smartwatch"`
  - `features = ["AI-powered fitness tracking", "5-day battery life", "Holographic display"]`
- **필요한 라이브러리:**
  ```bash
  pip install transformers torch
  ```
- **소스코드 내용:**
  ```python
  try:
      from transformers import pipeline
      import torch
      TRANSFORMERS_AVAILABLE = True
      print("Transformers 라이브러리가 사용 가능합니다.")
  except ImportError:
      TRANSFORMERS_AVAILABLE = False
      print("Transformers 라이브러리가 설치되지 않았습니다.")

  import random

  class MockMarketingGenerator:
      """마케팅 카피 생성을 위한 모의 생성기"""
      def __init__(self):
          self.openings = [
              "Unleash the future with",
              "Experience the next generation of technology with",
              "Introducing the revolutionary",
              "Step into tomorrow with"
          ]
          self.closings = [
              "Get yours today and redefine your world.",
              "Don't just keep up with the future, own it.",
              "The ultimate device for the modern achiever.",
              "Available now for a limited time."
          ]

      def generate(self, product_name, features):
          feature_sentence = f"Featuring {features[0]}, an incredible {features[1]}, and a stunning {features[2]}."
          opening = random.choice(self.openings)
          closing = random.choice(self.closings)
          
          return f"{opening} {product_name}! {feature_sentence} {closing}"

  def generate_marketing_copy(product_name, features, generator):
      """
      제품 정보로 마케팅 문구를 생성하는 함수.
      
      :param product_name: 제품 이름 (str)
      :param features: 제품 기능 리스트 (list of str)
      :param generator: 실제 또는 모의 생성기
      :return: 생성된 마케팅 문구 (str)
      """
      feature_string = "\n- ".join(features)
      
      prompt = f"""
  Generate a short, catchy, and persuasive marketing description for a new product.

  Product Name: {product_name}
  Key Features:
  - {feature_string}

  Marketing Description:
  """
      
      print("\n--- 생성에 사용된 프롬프트 ---")
      print(prompt)
      
      if TRANSFORMERS_AVAILABLE:
          responses = generator(
              prompt, 
              max_length=150, 
              num_return_sequences=1,
              pad_token_id=generator.tokenizer.eos_token_id,
              no_repeat_ngram_size=2,
              temperature=0.8,
              top_p=0.95
          )
          generated_text = responses[0]['generated_text']
          # "Marketing Description:" 이후의 텍스트만 추출
          marketing_copy = generated_text.split("Marketing Description:")[1].strip()
      else:
          marketing_copy = generator.generate(product_name, features)
      
      return marketing_copy

  def main():
      """메인 실행 함수"""
      print("=== 문제 3.3: 간단한 도메인 응용 - 마케팅 카피 생성기 ===")
      
      if TRANSFORMERS_AVAILABLE:
          print("실제 모델을 사용하여 마케팅 카피를 생성합니다.")
          generator = pipeline('text-generation', model='distilgpt2', device=-1)
      else:
          print("모의 생성기를 사용하여 마케팅 카피를 생성합니다.")
          generator = MockMarketingGenerator()

      # 시나리오 1: 스마트워치
      product_name_1 = "QuantumLeap Smartwatch"
      features_1 = ["AI-powered fitness tracking", "5-day battery life", "Holographic display"]
      copy_1 = generate_marketing_copy(product_name_1, features_1, generator)
      print(f"\n===== '{product_name_1}' 마케팅 문구 =====")
      print(copy_1)

      # 시나리오 2: 커피 머신
      product_name_2 = "AromaMax Pro"
      features_2 = ["Bean-to-cup freshness", "Customizable brew strength", "Self-cleaning function"]
      copy_2 = generate_marketing_copy(product_name_2, features_2, generator)
      print(f"\n===== '{product_name_2}' 마케팅 문구 =====")
      print(copy_2)

      # 시나리오 3: 무선 이어폰
      product_name_3 = "EchoBuds Air"
      features_3 = ["Crystal-clear audio", "Active noise cancellation", "All-day comfort fit"]
      copy_3 = generate_marketing_copy(product_name_3, features_3, generator)
      print(f"\n===== '{product_name_3}' 마케팅 문구 =====")
      print(copy_3)

      print("\n\n=== 프롬프트 엔지니어링 해설 ===")
      print("1. **역할 부여 (Role-playing):** 'Generate a ... marketing description' 프롬프트를 통해 LLM에게 '마케터' 역할을 부여했습니다.")
      print("2. **구조화된 입력:** 'Product Name', 'Key Features'와 같이 명확한 레이블을 사용하여 정보를 구조적으로 제공했습니다.")
      print("3. **출력 형식 지정:** 'Marketing Description:'으로 프롬프트를 마무리하여, LLM이 이어서 내용을 채우도록 유도했습니다.")
      print("4. **품질 요구:** 'short, catchy, and persuasive'와 같은 형용사를 사용하여 원하는 결과물의 스타일을 명시했습니다.")
      print("=> 이처럼 원하는 결과에 맞게 프롬프트를 설계하는 과정을 '프롬프트 엔지니어링'이라고 합니다.")

  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_3_3_marketing_copy_generator.py`

---

## Chapter 4: 고급 프롬프트 엔지니어링 기법 마스터

### 문제 4.1: 퓨샷(Few-Shot) 인-컨텍스트 러닝

- **문제:** 모델에게 작업을 직접 지시하는 대신, 몇 개의 예시(shots)를 보여줌으로써 모델이 스스로 작업의 패턴을 학습하고 새로운 문제에 적용하게 하는 '인-컨텍스트 러닝'을 실습합니다.
- **지시어:** 새로운 영화 리뷰의 감성을 분류하는 프롬프트를 작성하세요. 이 프롬프트는 모델에게 작업을 직접 지시하는 대신, 긍정 리뷰 예시 하나와 부정 리뷰 예시 하나를 먼저 보여준 후, 분류해야 할 새로운 리뷰를 제시해야 합니다. 이러한 '퓨샷(few-shot)' 방식을 통해 모델이 추가적인 훈련(fine-tuning) 없이도 주어진 예시를 보고 작업을 학습하도록 유도하세요.
- **제공데이터:**
  - 예시: `{"review": "A masterpiece!", "sentiment": "Positive"}`, `{"review": "Utterly boring.", "sentiment": "Negative"}`
  - 분류 대상: `{"review": "It was a decent film, with some flaws."}`
- **필요한 라이브러리:**
  ```bash
  pip install transformers torch
  ```
- **소스코드 내용:**
  ```python
  try:
      from transformers import pipeline
      import torch
      TRANSFORMERS_AVAILABLE = True
      print("Transformers 라이브러리가 사용 가능합니다.")
  except ImportError:
      TRANSFORMERS_AVAILABLE = False
      print("Transformers 라이브러리가 설치되지 않았습니다.")

  import random

  class MockFewShotClassifier:
      """퓨샷 학습을 시뮬레이션하는 모의 분류기"""
      def __init__(self):
          self.positive_keywords = ['masterpiece', 'fantastic', 'amazing', 'enjoyed', 'stunning']
          self.negative_keywords = ['boring', 'flaws', 'terrible', 'waste']

      def classify(self, new_review):
          review_lower = new_review.lower()
          pos_count = sum(1 for word in self.positive_keywords if word in review_lower)
          neg_count = sum(1 for word in self.negative_keywords if word in review_lower)
          
          if pos_count > neg_count:
              return "Positive"
          elif neg_count > pos_count:
              return "Negative"
          else:
              return "Neutral" # 실제 LLM은 더 미묘한 판단을 할 수 있음

  def create_few_shot_prompt(examples, new_input):
      """퓨샷 프롬프트를 생성하는 함수"""
      prompt = "Classify the sentiment of the movie review. The sentiment can be Positive or Negative.\n\n"
      for ex in examples:
          prompt += f"Review: \"{ex['review']}\"\nSentiment: {ex['sentiment']}\n\n"
      prompt += f"Review: \"{new_input}\"\nSentiment:"
      return prompt

  def main():
      """메인 실행 함수"""
      print("=== 문제 4.1: 퓨샷(Few-Shot) 인-컨텍스트 러닝 ===")
      
      examples = [
          {"review": "A masterpiece!", "sentiment": "Positive"},
          {"review": "Utterly boring.", "sentiment": "Negative"}
      ]
      
      reviews_to_classify = [
          "It was a decent film, with some flaws.",
          "An absolute triumph of cinema.",
          "I almost fell asleep halfway through.",
          "The plot was confusing, but the acting was superb."
      ]

      if TRANSFORMERS_AVAILABLE:
          print("\n=== 실제 모델을 사용한 퓨샷 분류 ===")
          generator = pipeline('text-generation', model='distilgpt2', device=-1)
          
          for new_review in reviews_to_classify:
              prompt = create_few_shot_prompt(examples, new_review)
              
              responses = generator(
                  prompt,
                  max_new_tokens=5,
                  num_return_sequences=1,
                  pad_token_id=generator.tokenizer.eos_token_id
              )
              
              full_text = responses[0]['generated_text']
              # 'Sentiment:'의 마지막 등장 이후 텍스트 추출
              prediction = full_text.split('Sentiment:')[-1].strip().split('\n')[0]
              
              print(f"\n리뷰: '{new_review}'")
              print(f"  -> 예측된 감성: {prediction}")

      else:
          print("\n=== 모의 분류기를 사용한 퓨샷 개념 시뮬레이션 ===")
          mock_classifier = MockFewShotClassifier()
          for new_review in reviews_to_classify:
              prompt = create_few_shot_prompt(examples, new_review)
              print("\n--- 생성된 프롬프트 ---")
              print(prompt)
              prediction = mock_classifier.classify(new_review)
              print(f"----------------------")
              print(f"리뷰: '{new_review}'")
              print(f"  -> 예측된 감성: {prediction}")

      print("\n\n=== 퓨샷 러닝 해설 ===")
      print("1. **인-컨텍스트 러닝(In-Context Learning):** 모델의 가중치를 변경하는 '파인튜닝'과 달리, 프롬프트 내에 예시를 제공하여 모델이 해당 '문맥' 안에서 작업을 학습하도록 하는 방식입니다.")
      print("2. **샷(Shot)의 의미:** 제공되는 예시의 개수를 의미합니다.")
      print("   - **제로샷 (Zero-Shot):** 예시 없이 작업 설명만으로 요청 (e.g., 'Translate this to French.')")
      print("   - **원샷 (One-Shot):** 하나의 예시를 제공.")
      print("   - **퓨샷 (Few-Shot):** 여러 개의 예시를 제공 (본 문제처럼).")
      print("3. **장점:**")
      print("   - **빠른 적용:** 별도의 훈련 과정 없이 다양한 작업에 모델을 빠르게 적용할 수 있습니다.")
      print("   - **데이터 효율성:** 소수의 예시만으로도 모델의 행동을 유도할 수 있습니다.")
      print("4. **핵심 원리:** 모델은 주어진 예시의 패턴(입력 -> 출력 형식)을 파악하고, 새로운 입력에 대해서도 동일한 패턴을 따르려고 합니다.")

  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_4_1_few_shot_learning.py`

---

### 문제 4.2: LangChain을 이용한 기본 RAG 시스템 구축

- **문제:** LLM이 학습하지 않은 최신 정보나 특정 도메인 지식에 대해 답변하게 하는 RAG(검색 증강 생성) 기술의 기본 원리를 이해합니다.
- **지시어:** LangChain 라이브러리를 사용하여 간단한 Retrieval-Augmented Generation (RAG) 시스템을 구축하세요. 주어진 텍스트 문서를 기반으로 소규모 벡터 저장소(vector store)를 생성한 후, 해당 문서의 정보로만 답변할 수 있는 질문을 하여 LLM이 외부 지식을 참조해 답변하도록 만드세요.
- **제공데이터:**
  - 문서: `"Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass more than two and a half times that of all the other planets in the Solar System combined. Jupiter's largest moon is Ganymede."`
  - 질문: `"What is the name of Jupiter's largest moon?"`
- **필요한 라이브러리:**
  ```bash
  pip install langchain langchain-community sentence-transformers faiss-cpu
  # HuggingFace Hub 모델 사용 시:
  # pip install huggingface_hub
  ```
- **소스코드 내용:**
  ```python
  try:
      from langchain.text_splitter import RecursiveCharacterTextSplitter
      from langchain.embeddings import HuggingFaceEmbeddings
      from langchain.vectorstores import FAISS
      from langchain.chains import RetrievalQA
      from langchain.llms import HuggingFacePipeline
      from langchain.document_loaders import TextLoader
      from langchain.docstore.document import Document
      LANGCHAIN_AVAILABLE = True
      print("LangChain 라이브러리가 사용 가능합니다.")
  except ImportError:
      LANGCHAIN_AVAILABLE = False
      print("LangChain 라이브러리가 설치되지 않았습니다.")

  try:
      import faiss
      FAISS_AVAILABLE = True
  except ImportError:
      FAISS_AVAILABLE = False

  import json
  import os
  import random

  class MockRAGSystem:
      """LangChain이 없을 때 사용할 모의 RAG 시스템"""
      def __init__(self, documents):
          self.documents = documents
          self.knowledge_base = " ".join([doc.page_content for doc in documents])
          print("모의 RAG 시스템을 사용합니다.")

      def answer_question(self, question):
          question_lower = question.lower()
          # 간단한 키워드 기반 검색
          if "jupiter" in question_lower and "moon" in question_lower:
              if "ganymede" in self.knowledge_base.lower():
                  return "Ganymede", self.documents
          if "ai" in question_lower:
              return "AI is the simulation of human intelligence in machines.", self.documents
          if "machine learning" in question_lower and "deep learning" in question_lower:
              return "Deep learning is a subset of machine learning.", self.documents
          
          # 관련 정보가 없을 경우
          return "I don't have information about that.", []

      def retrieve_documents(self, query):
          # 간단한 검색 시뮬레이션
          return self.documents[:3] # 항상 상위 3개 문서 반환

  def setup_real_rag_system(documents):
      """실제 LangChain RAG 시스템을 설정하는 함수"""
      if not LANGCHAIN_AVAILABLE or not FAISS_AVAILABLE:
          raise ImportError("LangChain 또는 FAISS가 설치되지 않았습니다.")
      
      try:
          text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
          split_docs = text_splitter.split_documents(documents)
          
          embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
          vectorstore = FAISS.from_documents(split_docs, embeddings)
          
          # 실제 LLM 대신 간단한 응답 함수 사용 (API 키 불필요)
          def simple_llm(context, question):
              return f"Based on the retrieved context, the answer to '{question}' is likely found within: '{context[:100]}...'"

          retriever = vectorstore.as_retriever()
          
          # RetrievalQA 대신 수동으로 구현
          def qa_chain(query):
              retrieved_docs = retriever.get_relevant_documents(query)
              context = " ".join([doc.page_content for doc in retrieved_docs])
              answer = simple_llm(context, query)
              return {"result": answer, "source_documents": retrieved_docs}

          return qa_chain, retriever
      except Exception as e:
          print(f"실제 RAG 설정 실패: {e}")
          return None, None

  def main():
      """메인 실행 함수"""
      print("=== 문제 4.2: LangChain을 활용한 RAG 시스템 ===")
      
      # 1. 문서 데이터 준비
      docs_content = {
          "AI_intro.txt": "Artificial intelligence (AI) is the simulation of human intelligence in machines. Key technologies include machine learning and deep learning.",
          "ML_basics.txt": "Machine learning is a subset of AI where systems learn from data. It includes supervised, unsupervised, and reinforcement learning.",
          "DL_guide.txt": "Deep learning uses multi-layered neural networks to learn complex patterns. It excels in image recognition, NLP, and speech recognition.",
          "Transformer_arch.txt": "The Transformer architecture, based on self-attention, is the foundation for most modern large language models like BERT and GPT.",
          "Jupiter_info.txt": "Jupiter is the fifth planet from the Sun and the largest in the Solar System. Jupiter's largest moon is Ganymede."
      }
      
      documents = [Document(page_content=content, metadata={"source": name}) for name, content in docs_content.items()]

      # RAG 시스템 설정
      rag_system = None
      retriever = None
      system_type = "Mock RAG"

      if LANGCHAIN_AVAILABLE and FAISS_AVAILABLE:
          print("\n실제 RAG 시스템을 설정합니다...")
          try:
              qa_chain, retriever = setup_real_rag_system(documents)
              if qa_chain:
                  rag_system = qa_chain
                  system_type = "Real RAG"
                  print("RAG 시스템 설정 완료!")
              else:
                  raise ValueError("QA Chain 설정 실패")
          except Exception as e:
              print(f"실제 RAG 시스템 설정 중 오류 발생: {e}")
              rag_system = MockRAGSystem(documents)
              retriever = rag_system # Mock retriever
      else:
          rag_system = MockRAGSystem(documents)
          retriever = rag_system # Mock retriever

      print(f"사용 시스템: {system_type}")

      # --- 질의 응답 데모 ---
      print("\n=== RAG 시스템 질의 응답 데모 ===")
      print("문서 데이터베이스에서 관련 정보를 검색하여 답변합니다.")
      print("------------------------------------------------------------")
      
      questions = [
          "What is artificial intelligence?",
          "What is the difference between machine learning and deep learning?",
          "Tell me about the Transformer architecture",
          "What is the process of natural language processing?",
          "What are the characteristics of reinforcement learning?",
          "What is the name of Jupiter's largest moon?"
      ]

      for q in questions:
          print(f"\n--- 질문: {q} ---")
          try:
              if system_type == "Real RAG":
                  response = rag_system(q)
                  answer = response['result']
                  source_docs = response['source_documents']
              else:
                  answer, source_docs = rag_system.answer_question(q)
              
              print(f"답변: {answer}")
              if source_docs:
                  print("\n참조 문서들:")
                  for doc in source_docs:
                      print(f"  - {doc.metadata['source']}: {doc.page_content[:80]}...")
          except Exception as e:
              print(f"오류 발생: {e}")

      # --- 검색 품질 분석 ---
      print("\n=== 검색 품질 분석 ===")
      print("다양한 질문 유형에 대한 검색 성능 평가")
      print("------------------------------------------------------------")

      search_queries = {
          "정의 질문": ["What is the definition of AI?", "What is deep learning?"],
          "비교 질문": ["Difference between machine learning and deep learning?", "Relationship between BERT and Transformer?"],
          "방법 질문": ["How does natural language processing work?", "How does reinforcement learning learn?"],
          "응용 질문": ["What are the applications of computer vision?", "Where are Transformers used?"]
      }

      for q_type, queries in search_queries.items():
          print(f"\n--- {q_type} ---")
          for query in queries:
              print(f"\n질문: {query}")
              try:
                  if system_type == "Real RAG":
                      retrieved_docs = retriever.get_relevant_documents(query)
                  else:
                      retrieved_docs = retriever.retrieve_documents(query)
                  
                  print(f"검색된 문서 수: {len(retrieved_docs)}")
                  if retrieved_docs:
                      print(f"최상위 문서: {retrieved_docs[0].metadata['source']}")
                      
                      # 키워드 겹침 분석 (간단하게)
                      query_words = set(query.lower().split())
                      content_words = set(retrieved_docs[0].page_content.lower().split())
                      overlap = len(query_words.intersection(content_words))
                      print(f"키워드 겹침: {overlap}개")

              except Exception as e:
                  print(f"검색 중 오류: {e}")

      # --- RAG 시스템 구성 요소 설명 ---
      print("\n=== RAG 시스템 구성 요소 ===")
      components = [
          ("1. 문서 로더 (Document Loader)", "다양한 형식의 문서를 시스템에 로드", "PDF, TXT, HTML, 웹페이지 등", "TextLoader, PyPDFLoader, WebBaseLoader"),
          ("2. 텍스트 분할기 (Text Splitter)", "긴 문서를 검색 가능한 청크로 분할", "문장 단위, 단락 단위, 토큰 수 기반", "RecursiveCharacterTextSplitter, TokenTextSplitter"),
          ("3. 임베딩 모델 (Embedding Model)", "텍스트를 벡터로 변환하여 의미적 유사도 계산", "sentence-transformers, OpenAI embeddings", "HuggingFaceEmbeddings, OpenAIEmbeddings"),
          ("4. 벡터 스토어 (Vector Store)", "임베딩 벡터를 저장하고 유사도 검색 수행", "FAISS, Chroma, Pinecone, Weaviate", "FAISS, Chroma, Pinecone"),
          ("5. 검색기 (Retriever)", "쿼리에 대해 관련 문서 청크 검색", "유사도 검색, 하이브리드 검색", "VectorStoreRetriever, MultiQueryRetriever"),
          ("6. 언어 모델 (Language Model)", "검색된 정보를 바탕으로 최종 답변 생성", "GPT, Claude, Llama", "OpenAI, HuggingFacePipeline"),
          ("7. 체인 (Chain)", "전체 RAG 파이프라인을 연결하고 조율", "RetrievalQA, ConversationalRetrievalChain", "RetrievalQA, RetrievalQAWithSourcesChain")
      ]
      for title, role, example, lc_class in components:
          print(f"\n{title}:")
          print(f"  역할: {role}")
          print(f"  예시: {example}")
          print(f"  LangChain: {lc_class}")

      # --- RAG 시스템 최적화 기법 ---
      print("\n=== RAG 시스템 최적화 기법 ===")
      optimizations = [
          ("청크 크기 최적화", "문서 분할 시 청크 크기를 작업에 맞게 조정", "100-500 토큰 범위에서 실험", "너무 작으면 컨텍스트 부족, 너무 크면 노이즈 증가"),
          ("하이브리드 검색", "키워드 검색과 벡터 검색을 결합", "BM25 + 벡터 유사도의 가중 평균", "정확한 매칭과 의미적 유사도 모두 활용"),
          ("쿼리 확장", "사용자 쿼리를 의미적으로 확장하여 검색 향상", "동의어, 관련 용어, 다른 표현 방식 추가", "과도한 확장은 노이즈 증가 가능"),
          ("재순위화 (Re-ranking)", "초기 검색 결과를 더 정교한 모델로 재순위화", "Cross-encoder 모델 사용", "계산 비용 증가하지만 정확도 향상"),
          ("메타데이터 필터링", "문서의 메타데이터를 활용한 검색 범위 제한", "날짜, 카테고리, 저자 등으로 필터링", "관련성 높은 문서에 집중 가능"),
          ("문맥 압축", "검색된 문서에서 관련 부분만 추출", "요약 모델이나 문장 선택 알고리즘 사용", "토큰 수 절약과 중요 정보 보존의 균형")
      ]
      for title, desc, method, consideration in optimizations:
          print(f"\n{title}:")
          print(f"  설명: {desc}")
          print(f"  방법: {method}")
          print(f"  고려사항: {consideration}")

      # --- RAG 시스템 평가 메트릭 ---
      print("\n=== RAG 시스템 평가 메트릭 ===")
      metrics = {
          "검색 품질": ["Recall@K", "Precision@K", "MRR", "NDCG"],
          "생성 품질": ["BLEU", "ROUGE", "BERTScore", "Faithfulness"],
          "종합 평가": ["End-to-End Accuracy", "Human Evaluation", "Response Time", "Cost Efficiency"]
      }
      for category, items in metrics.items():
          print(f"\n{category}:")
          for item in items:
              print(f"  {item}")

      # --- RAG 시스템 실제 활용 사례 ---
      print("\n=== RAG 시스템 실제 활용 사례 ===")
      use_cases = [
          ("고객 지원", "FAQ, 매뉴얼 기반 자동 응답", "제품 문서, 과거 상담 기록", "24/7 지원, 일관된 답변 품질"),
          ("법률 검색", "판례, 법령 기반 법률 조언", "법률 문서, 판례집", "빠른 법령 검색, 관련 판례 찾기"),
          ("의료 진단 지원", "의학 문헌 기반 진단 보조", "의학 논문, 진료 가이드라인", "최신 연구 반영, 진단 정확도 향상"),
          ("교육", "교육 자료 기반 질의응답", "교과서, 강의 노트, 참고서", "개인화된 학습 지원, 즉시 피드백"),
          ("연구 지원", "논문 데이터베이스 기반 연구 도움", "학술 논문, 연구 보고서", "관련 연구 빠른 발견, 연구 동향 파악"),
          ("기업 지식 관리", "내부 문서 기반 정보 제공", "내부 문서, 프로세스 매뉴얼", "지식 공유 효율화, 업무 생산성 향상")
      ]
      for use, purpose, data, effect in use_cases:
          print(f"\n{use}:")
          print(f"  용도: {purpose}")
          print(f"  데이터: {data}")
          print(f"  효과: {effect}")

      if not LANGCHAIN_AVAILABLE or not FAISS_AVAILABLE:
          print("\n=== 설치 안내 ===")
          print("실제 RAG 시스템을 사용하려면:")
          print("pip install langchain sentence-transformers faiss-cpu")

  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_4_2_langchain_rag.py`

---

### 문제 4.3: 사고의 연쇄(Chain-of-Thought)를 통한 추론 유도

- **문제:** 복잡한 추론이 필요한 문제에 대해, LLM이 중간 과정을 생략하고 틀린 답을 내놓는 경우를 방지합니다.
- **지시어:** 간단한 다단계 연산이 필요한 응용 문제를 LLM에게 풀어보게 하세요. 첫 번째 시도에서는 최종 답변만 요구하고, 두 번째 시도에서는 프롬프트에 "Let's think step by step."이라는 문구를 추가하여 모델이 문제 해결 과정을 단계별로 서술하도록 유도하세요. 두 결과의 정확성과 출력 내용을 비교 분석하세요.
- **제공데이터:**
  - 문제: `"A farmer has 15 apples. He sells 3 to his neighbor and then buys 5 more. He then divides the apples equally among his 4 children. How many apples does each child get?"`
- **필요한 라이브러리:**
  ```bash
  pip install transformers torch
  ```
- **소스코드 내용:**
  ```python
  try:
      from transformers import pipeline
      import torch
      TRANSFORMERS_AVAILABLE = True
      print("Transformers 라이브러리가 사용 가능합니다.")
  except ImportError:
      TRANSFORMERS_AVAILABLE = False
      print("Transformers 라이브러리가 설치되지 않았습니다.")

  class MockCoTModel:
      """사고의 연쇄를 시뮬레이션하는 모의 모델"""
      def __init__(self):
          self.problem = "A farmer has 15 apples. He sells 3 to his neighbor and then buys 5 more. He then divides the apples equally among his 4 children. How many apples does each child get?"
          self.direct_answer = "Answer: 4.25" # 일반적인 LLM이 저지를 수 있는 실수
          self.step_by_step_answer = """
  Let's think step by step.
  1. The farmer starts with 15 apples.
  2. He sells 3 apples, so he has 15 - 3 = 12 apples.
  3. He then buys 5 more apples, so he now has 12 + 5 = 17 apples.
  4. He divides the 17 apples among his 4 children.
  5. 17 divided by 4 is 4 with a remainder of 1. So, each child gets 4 apples, and there is 1 apple left over.
  Answer: Each child gets 4 apples.
  """

      def generate(self, prompt):
          if "step by step" in prompt:
              return [{'generated_text': prompt + self.step_by_step_answer}]
          else:
              return [{'generated_text': prompt + self.direct_answer}]

  def main():
      """메인 실행 함수"""
      print("=== 문제 4.3: 사고의 연쇄(Chain-of-Thought)를 통한 추론 유도 ===")
      
      problem = "A farmer has 15 apples. He sells 3 to his neighbor and then buys 5 more. He then divides the apples equally among his 4 children. How many apples does each child get?"

      if TRANSFORMERS_AVAILABLE:
          print("\n=== 실제 모델을 사용한 추론 과정 비교 ===")
          # 추론 작업에는 더 큰 모델이 유리할 수 있음
          # 리소스 부족 시 'distilgpt2'로 변경
          generator = pipeline('text-generation', model='gpt2', device=-1)
      else:
          print("\n=== 모의 모델을 사용한 추론 과정 비교 ===")
          generator = MockCoTModel()

      # --- 시도 1: 최종 답변만 요구 ---
      print("\n--- 시도 1: 최종 답변만 요구 ---")
      prompt1 = f"Question: {problem}\nAnswer:"
      
      if TRANSFORMERS_AVAILABLE:
          response1 = generator(prompt1, max_new_tokens=50, pad_token_id=generator.tokenizer.eos_token_id)
      else:
          response1 = generator.generate(prompt1)
          
      print(response1[0]['generated_text'])
      print("-> 결과 분석: 모델이 중간 계산 과정을 생략하고 성급하게 답을 내놓아 틀릴 수 있습니다.")

      # --- 시도 2: 사고의 연쇄(Chain-of-Thought) 프롬프트 ---
      print("\n--- 시도 2: 사고의 연쇄(Chain-of-Thought) 프롬프트 ---")
      prompt2 = f"Question: {problem}\nLet's think step by step."
      
      if TRANSFORMERS_AVAILABLE:
          response2 = generator(prompt2, max_new_tokens=150, pad_token_id=generator.tokenizer.eos_token_id)
      else:
          response2 = generator.generate(prompt2)
          
      print(response2[0]['generated_text'])
      print("\n-> 결과 분석: 'Let's think step by step'이라는 간단한 문구 하나로 모델이 문제 해결 과정을 단계별로 서술하도록 유도했습니다. 이 과정을 통해 모델은 스스로 추론 과정을 검증하고 더 정확한 답에 도달할 확률이 높아집니다.")

      print("\n\n=== 사고의 연쇄(Chain-of-Thought, CoT) 해설 ===")
      print("1. **정의:** 복잡한 추론 문제를 풀 때, LLM에게 최종 답변뿐만 아니라 중간 생각 과정까지 함께 생성하도록 유도하는 프롬프트 기법입니다.")
      print("2. **원리:**")
      print("   - LLM은 다음 단어를 예측하는 방식으로 작동합니다.")
      print("   - '단계별로 생각하자'는 프롬프트는 모델이 '첫 번째 단계는...', '그 다음은...' 과 같은 형태의 텍스트를 생성하도록 유도합니다.")
      print("   - 이 과정에서 모델은 각 단계를 독립적으로 계산하고, 이전 단계의 결과를 다음 단계의 입력으로 사용하게 됩니다.")
      print("3. **장점:**")
      print("   - **정확도 향상:** 특히 산수, 상식 추론, 논리 문제 등 다단계 추론이 필요한 작업에서 성능을 크게 향상시킵니다.")
      print("   - **해석 가능성:** 모델이 왜 그런 결론에 도달했는지 중간 과정을 보고 이해할 수 있습니다. (디버깅에 용이)")
      print("4. **종류:**")
      print("   - **Zero-shot CoT:** 본 문제처럼 별도의 예시 없이 'Let's think step by step'과 같은 문구만 추가하는 방식.")
      print("   - **Few-shot CoT:** 문제-단계별풀이-답변 형식의 예시 몇 개를 프롬프트에 포함하여 더 명확하게 패턴을 학습시키는 방식.")

  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_4_3_chain_of_thought.py`

---

## Chapter 5: 파인튜닝을 통한 LLM의 효율적 적응

### 문제 5.1: PEFT(LoRA)를 이용한 파라미터 효율적 파인튜닝

- **문제:** LLM 전체를 재학습시키는 것의 막대한 비용 문제를 해결하기 위한 PEFT(파라미터 효율적 파인튜닝) 기법 중 하나인 LoRA를 실습합니다.
- **지시어:** Hugging Face의 `PEFT` 라이브러리를 사용하여 사전 훈련된 소형 모델(`distilgpt2`)에 LoRA(Low-Rank Adaptation)를 적용하세요. 주어진 소규모 맞춤형 데이터셋(법률 질문 -> 답변 형식)으로 모델을 파인튜닝하여, 모델이 법률 용어에 대해 더 적절하게 답변하도록 행동을 변화시키는 과정을 실습하세요.
- **제공데이터:**
  - `legal_data.jsonl` 파일:
    ```json
    {"instruction": "What is a tort?", "output": "A tort is a civil wrong that causes a claimant to suffer loss or harm, resulting in legal liability for the person who commits thetortious act."}
    {"instruction": "Explain the concept of 'habeas corpus'.", "output": "Habeas corpus is a legal recourse through which a person can report an unlawful detention or imprisonment to a court and request that the court order the custodian of the person, usually a prison official, to bring the prisoner to court to determine if the detention is lawful."}
    {"instruction": "What does 'pro bono' mean?", "output": "'Pro bono publico', often shortened to 'pro bono', is a Latin phrase for professional work undertaken voluntarily and without payment. It is a way for professionals to contribute to the community."}
    ```
- **필요한 라이브러리:**
  ```bash
  pip install transformers datasets peft accelerate bitsandbytes trl
  ```
- **소스코드 내용:**
  ```python
  import torch
  import os
  import json

  try:
      from datasets import load_dataset, Dataset
      from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
      from peft import LoraConfig, get_peft_model, PeftModel
      from trl import SFTTrainer
      LIBRARIES_AVAILABLE = True
      print("PEFT 및 관련 라이브러리가 사용 가능합니다.")
  except ImportError:
      LIBRARIES_AVAILABLE = False
      print("필요한 라이브러리(transformers, datasets, peft, trl)가 설치되지 않았습니다.")

  class MockLoRAModel:
      """LoRA 파인튜닝을 시뮬레이션하는 모의 모델"""
      def __init__(self, base_model):
          self.base_model = base_model
          self.lora_knowledge = {}
          print("모의 LoRA 모델을 사용합니다.")

      def train(self, dataset):
          print("\n모의 LoRA 훈련 시작...")
          for item in dataset:
              # instruction에서 키워드 추출
              keyword = item['instruction'].split("'")[1] if "'" in item['instruction'] else item['instruction'].split()[-1].replace('?', '')
              self.lora_knowledge[keyword.lower()] = item['output']
          print("모의 LoRA 훈련 완료. '어댑터'에 지식이 저장되었습니다.")
          print(f"학습된 키워드: {list(self.lora_knowledge.keys())}")

      def generate(self, prompt):
          # 프롬프트에서 키워드 찾기
          for keyword, response in self.lora_knowledge.items():
              if keyword in prompt.lower():
                  return f"{prompt}{response}"
          # LoRA 지식에 없으면 기본 모델 응답
          return self.base_model.generate(prompt)

      def print_trainable_parameters(self):
          print("trainable params: 245,760 || all params: 82,112,000 || trainable%: 0.2993")

  class MockBaseModel:
      def generate(self, prompt):
          return f"{prompt}This is a generic response from the base model."

  def main():
      """메인 실행 함수"""
      print("=== 문제 5.1: PEFT(LoRA)를 이용한 파라미터 효율적 파인튜닝 ===")

      # 데이터셋 파일 생성
      legal_data = [
          {"instruction": "What is a tort?", "output": "A tort is a civil wrong that causes a claimant to suffer loss or harm, resulting in legal liability for the person who commits the tortious act."},
          {"instruction": "Explain the concept of 'habeas corpus'.", "output": "Habeas corpus is a legal recourse through which a person can report an unlawful detention or imprisonment to a court and request that the court order the custodian of the person, usually a prison official, to bring the prisoner to court to determine if the detention is lawful."},
          {"instruction": "What does 'pro bono' mean?", "output": "'Pro bono publico', often shortened to 'pro bono', is a Latin phrase for professional work undertaken voluntarily and without payment. It is a way for professionals to contribute to the community."}
      ]
      dataset_path = "legal_data.jsonl"
      with open(dataset_path, "w") as f:
          for entry in legal_data:
              f.write(json.dumps(entry) + "\n")

      if LIBRARIES_AVAILABLE:
          # --- 실제 라이브러리 사용 ---
          # 1. 데이터셋 로드 및 포맷팅
          def formatting_prompts_func(example):
              output_texts = []
              for i in range(len(example['instruction'])):
                  text = f"### Instruction:\n{example['instruction'][i]}\n\n### Response:\n{example['output'][i]}"
                  output_texts.append(text)
              return {"text": output_texts}

          dataset = load_dataset("json", data_files=dataset_path, split="train")
          dataset = dataset.map(formatting_prompts_func, batched=True)

          # 2. 모델 및 토크나이저 로드
          model_name = "distilgpt2"
          tokenizer = AutoTokenizer.from_pretrained(model_name)
          tokenizer.pad_token = tokenizer.eos_token
          base_model = AutoModelForCausalLM.from_pretrained(model_name)

          # 3. LoRA 설정
          lora_config = LoraConfig(
              r=8, lora_alpha=16,
              target_modules=["c_attn", "c_proj"],
              lora_dropout=0.1, bias="none",
              task_type="CAUSAL_LM"
          )
          peft_model = get_peft_model(base_model, lora_config)
          peft_model.print_trainable_parameters()

          # 4. 훈련 설정 및 실행
          training_args = TrainingArguments(
              output_dir="./lora_results",
              per_device_train_batch_size=1,
              num_train_epochs=10,
              logging_steps=1,
              learning_rate=2e-4,
              remove_unused_columns=False
          )
          trainer = SFTTrainer(
              model=peft_model,
              train_dataset=dataset,
              dataset_text_field="text",
              args=training_args,
              max_seq_length=128,
          )
          print("\nLoRA 파인튜닝 시작...")
          trainer.train()
          print("LoRA 파인튜닝 완료.")

          # 5. 파인튜닝된 모델 테스트
          prompt_text = "### Instruction:\nWhat is a tort?\n\n### Response:\n"
          inputs = tokenizer(prompt_text, return_tensors="pt")
          
          print("\n--- 파인튜닝 전 기본 모델 응답 ---")
          base_outputs = base_model.generate(**inputs, max_new_tokens=60)
          print(tokenizer.decode(base_outputs[0], skip_special_tokens=True))

          print("\n--- 파인튜닝 후 LoRA 모델 응답 ---")
          peft_outputs = peft_model.generate(**inputs, max_new_tokens=60)
          print(tokenizer.decode(peft_outputs[0], skip_special_tokens=True))

      else:
          # --- 모의 객체 사용 ---
          base_model = MockBaseModel()
          lora_model = MockLoRAModel(base_model)
          lora_model.print_trainable_parameters()
          
          lora_model.train(legal_data)
          
          prompt_text = "### Instruction:\nWhat is a tort?\n\n### Response:\n"
          
          print("\n--- 파인튜닝 전 기본 모델 응답 ---")
          print(base_model.generate(prompt_text))
          
          print("\n--- 파인튜닝 후 LoRA 모델 응답 ---")
          print(lora_model.generate(prompt_text))

      # 파일 정리
      if os.path.exists(dataset_path):
          os.remove(dataset_path)

      print("\n\n=== LoRA 해설 ===")
      print("1. **PEFT (Parameter-Efficient Fine-Tuning):** LLM의 모든 파라미터(수십억 개)를 훈련하는 대신, 극소수의 파라미터만 추가하거나 수정하여 모델을 특정 작업에 맞게 조정하는 기법들의 총칭입니다.")
      print("2. **LoRA (Low-Rank Adaptation):** 대표적인 PEFT 기법. 기존의 거대한 가중치 행렬(W)은 그대로 두고, 그 옆에 랭크가 낮은(low-rank) 두 개의 작은 행렬(A, B)을 추가합니다. 훈련 시에는 이 작은 행렬 A와 B만 업데이트합니다.")
      print("3. **장점:**")
      print("   - **메모리 효율성:** 전체 모델을 복사할 필요 없이, 작은 '어댑터'만 저장하면 되므로 디스크 공간을 크게 절약합니다.")
      print("   - **빠른 훈련:** 업데이트할 파라미터 수가 매우 적어 훈련 속도가 빠릅니다.")
      print("   - **재앙적 망각 방지:** 기존 모델의 지식은 그대로 유지되므로, 새로운 작업을 학습하면서 기존 능력을 잊어버리는 현상이 줄어듭니다.")
      print("   - **다중 작업 지원:** 기본 모델 하나에 여러 개의 LoRA 어댑터를 바꿔 끼우며 다양한 작업을 수행할 수 있습니다.")
      print("4. **본 실습의 의미:** 전체 모델(약 8,200만 파라미터) 중 단 0.3%에 해당하는 약 24만 개의 파라미터만 훈련하여, 모델이 새로운 '법률' 도메인에 대한 지식을 학습하도록 만들었습니다.")

  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_5_1_lora_finetuning.py`

---

### 문제 5.2: 개념적 RLHF - 보상 모델의 역할 이해

- **문제:** 인간의 선호를 모델에게 가르치는 RLHF(인간 피드백 기반 강화학습)의 핵심 구성요소인 '보상 모델'의 역할을 개념적으로 이해합니다.
- **지시어:** 이 문제는 코드 작성과 개념 이해를 결합한 문제입니다. LLM이 하나의 프롬프트에 대해 생성한 두 개의 다른 응답이 주어졌을 때, '보상 모델(Reward Model)'의 역할을 하는 간단한 Python 함수를 작성하세요. 이 함수는 정교한 AI 모델이 아니라, 길이, 특정 키워드 포함 여부, 공손함 등 간단한 휴리스틱(heuristic) 규칙을 기반으로 어떤 응답이 '더 나은지' 판단하고 더 높은 점수를 반환해야 합니다.
- **제공데이터:**
  - 프롬프트: `"Explain photosynthesis."`
  - 응답 A: `"It's how plants eat."`
  - 응답 B: `"Photosynthesis is the process used by plants, algae, and certain bacteria to harness energy from sunlight and turn it into chemical energy."`
- **필요한 라이브러리:** 없음
- **소스코드 내용:**
  ```python
  def simple_reward_model(prompt, response_a, response_b):
      """
      간단한 휴리스틱 기반의 보상 모델 함수.
      더 나은 응답에 높은 점수를 부여한다.
      
      :param prompt: 원본 프롬프트
      :param response_a: 첫 번째 응답
      :param response_b: 두 번째 응답
      :return: (응답 A 점수, 응답 B 점수, 더 나은 응답 문자열, 평가 근거)
      """
      score_a, score_b = 0, 0
      reasons = {"A": [], "B": []}
      
      # 휴리스틱 1: 더 길고 상세한 응답에 가산점
      len_a, len_b = len(response_a), len(response_b)
      if len_a > len_b:
          score_a += 1
          reasons["A"].append(f"더 김 (길이 {len_a} > {len_b})")
      elif len_b > len_a:
          score_b += 1
          reasons["B"].append(f"더 김 (길이 {len_b} > {len_a})")
          
      # 휴리스틱 2: 전문 용어 포함 여부에 가산점
      keywords = ["process", "energy", "sunlight", "chemical", "harness"]
      keywords_a = sum(1 for kw in keywords if kw in response_a.lower())
      keywords_b = sum(1 for kw in keywords if kw in response_b.lower())
      
      if keywords_a > 0:
          score_a += keywords_a
          reasons["A"].append(f"전문 용어 {keywords_a}개 포함")
      if keywords_b > 0:
          score_b += keywords_b
          reasons["B"].append(f"전문 용어 {keywords_b}개 포함")
          
      # 휴리스틱 3: 지나치게 짧은 응답에 감점
      if len(response_a.split()) < 5:
          score_a -= 2
          reasons["A"].append("너무 짧음 (5단어 미만)")
      if len(response_b.split()) < 5:
          score_b -= 2
          reasons["B"].append("너무 짧음 (5단어 미만)")

      # 휴리스틱 4: 정중함/격식 표현에 가산점
      polite_phrases = ["is the process", "is a method"]
      if any(p in response_a for p in polite_phrases):
          score_a += 0.5
          reasons["A"].append("격식 있는 표현 사용")
      if any(p in response_b for p in polite_phrases):
          score_b += 0.5
          reasons["B"].append("격식 있는 표현 사용")
          
      # 최종 판단
      if score_a > score_b:
          preferred_response = "Response A"
      elif score_b > score_a:
          preferred_response = "Response B"
      else:
          preferred_response = "Tie"
          
      return score_a, score_b, preferred_response, reasons

  def main():
      """메인 실행 함수"""
      print("=== 문제 5.2: 개념적 RLHF - 보상 모델의 역할 이해 ===")
      
      scenarios = [
          {
              "prompt": "Explain photosynthesis.",
              "response_a": "It's how plants eat.",
              "response_b": "Photosynthesis is the process used by plants, algae, and certain bacteria to harness energy from sunlight and turn it into chemical energy."
          },
          {
              "prompt": "What is the capital of France?",
              "response_a": "The capital is Paris, a beautiful city known for the Eiffel Tower.",
              "response_b": "Paris."
          },
          {
              "prompt": "Can you help me with my homework?",
              "response_a": "No.",
              "response_b": "Of course, I'd be happy to help! What subject are you working on?"
          }
      ]

      for i, scenario in enumerate(scenarios):
          print(f"\n--- 시나리오 {i+1} ---")
          prompt = scenario["prompt"]
          response_a = scenario["response_a"]
          response_b = scenario["response_b"]
          
          score_a, score_b, winner, reasons = simple_reward_model(prompt, response_a, response_b)

          print(f"프롬프트: '{prompt}'")
          print(f"  응답 A: '{response_a}'")
          print(f"    -> 점수: {score_a:.1f}, 근거: {', '.join(reasons['A'])}")
          print(f"  응답 B: '{response_b}'")
          print(f"    -> 점수: {score_b:.1f}, 근거: {', '.join(reasons['B'])}")
          print(f"  ==> 선호되는 응답: {winner}")

      print("\n\n=== RLHF와 보상 모델 해설 ===")
      print("1. **RLHF (Reinforcement Learning from Human Feedback):** 인간의 피드백을 통해 LLM을 미세 조정하는 기법입니다. '더 유용하고, 무해하며, 진실된' 답변을 하도록 모델을 훈련시키는 데 사용됩니다.")
      print("\n2. **RLHF의 3단계 과정:**")
      print("   - **1단계 (SFT 모델 훈련):** 인간이 작성한 고품질의 질문-답변 쌍으로 기본 모델을 파인튜닝합니다 (Supervised Fine-Tuning).")
      print("   - **2단계 (보상 모델 훈련):**")
      print("     a. SFT 모델에게 하나의 질문에 대해 여러 개의 답변을 생성하게 합니다.")
      print("     b. 인간 평가자가 이 답변들 사이의 순위를 매깁니다 (예: A > B > D > C).")
      print("     c. 이 '인간 선호도 데이터'를 사용하여, '좋은' 답변에 높은 점수를, '나쁜' 답변에 낮은 점수를 주는 **보상 모델(Reward Model)**을 훈련시킵니다.")
      print("     d. **본 실습의 `simple_reward_model` 함수가 바로 이 보상 모델의 역할을 개념적으로 흉내 낸 것입니다.**")
      print("   - **3단계 (강화학습 기반 파인튜닝):**")
      print("     a. SFT 모델이 새로운 프롬프트에 대해 답변을 생성합니다.")
      print("     b. 이 답변을 보상 모델에 넣어 '보상 점수'를 받습니다.")
      print("     c. 이 보상 점수를 최대화하는 방향으로 SFT 모델의 정책(policy)을 업데이트합니다. (PPO 알고리즘 사용)")
      print("\n3. **보상 모델의 중요성:** 보상 모델은 '인간의 가치 판단'을 학습한 대리인(proxy)입니다. 강화학습 단계에서 이 보상 모델이 실시간으로 피드백을 주기 때문에, 인간이 모든 생성 결과에 일일이 개입하지 않고도 모델을 올바른 방향으로 훈련시킬 수 있습니다.")
  
  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_5_2_rlhf_reward_model.py`

---

## Chapter 6: 멀티모달 AI의 최전선 탐험

### 문제 6.1: 이미지 캡셔닝

- **문제:** 텍스트뿐만 아니라 이미지도 이해하는 멀티모달(Multi-modal) AI의 능력을 실습합니다.
- **지시어:** Hugging Face `transformers` 라이브러리에서 사전 훈련된 이미지 캡셔닝 모델(예: `ViT-GPT2`)을 사용하세요. 주어진 이미지 URL을 모델에 입력하여, 이미지를 설명하는 자연어 캡션을 생성하고 출력하세요.
- **제공데이터:**
  - 이미지 URL: `http://images.cocodataset.org/val2017/000000039769.jpg` (고양이 두 마리가 컴퓨터 앞에 앉아있는 이미지)
- **필요한 라이브러리:**
  ```bash
  pip install transformers torch Pillow requests
  ```
- **소스코드 내용:**
  ```python
  import torch
  import os
  from PIL import Image
  import requests
  import random

  try:
      from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
      LIBRARIES_AVAILABLE = True
      print("Vision-Language 라이브러리가 사용 가능합니다.")
  except ImportError:
      LIBRARIES_AVAILABLE = False
      print("필요한 라이브러리(transformers, torch, Pillow)가 설치되지 않았습니다.")

  class MockImageCaptioningModel:
      """이미지 캡셔닝을 시뮬레이션하는 모의 모델"""
      def __init__(self, model_name="Mock BLIP Model"):
          self.model_name = model_name
          self.captions = {
              "beach": ["A beautiful beach during sunset with orange sky", "Waves crashing on a sandy shore"],
              "cat": ["A cute cat sitting on a windowsill", "Two cats looking at a computer screen"],
              "people": ["People enjoying a picnic in the park", "A group of friends laughing together"],
              "building": ["A modern skyscraper piercing the clouds", "An old historic building with intricate details"],
              "car": ["A red sports car driving on a winding road", "A vintage car parked on a city street"]
          }
          print(f"모의 캡셔닝 모델 '{self.model_name}'을(를) 사용합니다.")

      def generate(self, image_path):
          # 파일 경로에서 이미지 유형 추론
          image_type = "unknown"
          for key in self.captions.keys():
              if key in image_path:
                  image_type = key
                  break
          
          if image_type != "unknown":
              caption = random.choice(self.captions[image_type])
          else:
              caption = "An image of an object."
              
          confidence = random.uniform(0.6, 0.95)
          return caption, confidence

  def setup_image_dir():
      """실습용 이미지 디렉토리와 샘플 이미지를 준비하는 함수"""
      if not os.path.exists("images"):
          os.makedirs("images")
      
      image_urls = {
          "beach_sunset.jpg": "https://images.unsplash.com/photo-1507525428034-b723a9ce6890",
          "cat_portrait.jpg": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",
          "people_park.jpg": "https://images.unsplash.com/photo-1525498128493-380d1990a112",
          "modern_building.jpg": "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df",
          "vintage_car.jpg": "https://images.unsplash.com/photo-1552519507-da3b142c6e3d"
      }
      
      for filename, url in image_urls.items():
          path = os.path.join("images", filename)
          if not os.path.exists(path):
              try:
                  print(f"'{filename}' 다운로드 중...")
                  response = requests.get(url, stream=True)
                  response.raise_for_status()
                  with open(path, 'wb') as f:
                      for chunk in response.iter_content(chunk_size=8192):
                          f.write(chunk)
              except requests.exceptions.RequestException as e:
                  print(f"'{filename}' 다운로드 실패: {e}")
                  # 실패 시 빈 파일 생성하여 오류 방지
                  Image.new('RGB', (100, 100), color = 'red').save(path)

  def get_caption(image_path, model, tokenizer=None, feature_extractor=None, device='cpu'):
      """단일 이미지에 대한 캡션을 생성하는 함수"""
      try:
          image = Image.open(image_path).convert("RGB")
      except FileNotFoundError:
          return f"오류: '{image_path}' 파일을 찾을 수 없습니다.", 0.0

      if LIBRARIES_AVAILABLE and tokenizer and feature_extractor:
          pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
          generated_ids = model.generate(pixel_values, max_length=32, num_beams=4)
          caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
          # 실제 모델은 신뢰도 점수를 직접 제공하지 않으므로, 시뮬레이션을 위해 랜덤 값 사용
          confidence = random.uniform(0.8, 0.98)
      else: # Mock model
          caption, confidence = model.generate(image_path)
          
      return caption, confidence

  def main():
      """메인 실행 함수"""
      print("=== 문제 6.1: 이미지 캡셔닝 구현 ===")
      
      # 이미지 준비
      setup_image_dir()

      # 모델 로드
      if LIBRARIES_AVAILABLE:
          print("\n실제 BLIP 모델을 로드하는 중...")
          model = VisionEncoderDecoderModel.from_pretrained("Salesforce/blip-image-captioning-base")
          feature_extractor = ViTImageProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
          tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          model.to(device)
          print("실제 모델 로드 완료!")
          models_to_test = {"BLIP": (model, tokenizer, feature_extractor)}
      else:
          device = 'cpu'
          models_to_test = {"Mock BLIP Model": (MockImageCaptioningModel(), None, None)}
          print("\nVision-Language 라이브러리가 없어 모의 모델을 사용합니다.")

      # --- 데모 1: 단일 이미지 캡셔닝 ---
      print("\n=== 데모 1: 단일 이미지 캡셔닝 ===")
      image_path = "images/cat_portrait.jpg"
      print(f"테스트 이미지: {image_path}")
      
      model_obj, tok, fe = list(models_to_test.values())[0]
      caption, confidence = get_caption(image_path, model_obj, tok, fe, device)
      
      print(f"\n생성된 캡션: '{caption}'")
      print(f"신뢰도 (시뮬레이션): {confidence:.3f}")

      # --- 데모 2: 다양한 이미지에 대한 캡션 생성 ---
      print("\n=== 데모 2: 다양한 이미지에 대한 자동 캡션 생성 ===")
      sample_images = {
          "풍경": "images/beach_sunset.jpg",
          "동물": "images/cat_portrait.jpg",
          "사람": "images/people_park.jpg",
          "사물": "images/vintage_car.jpg"
      }
      for category, path in sample_images.items():
          caption, conf = get_caption(path, model_obj, tok, fe, device)
          print(f"\n카테고리: {category} ({path})")
          print(f"  -> 캡션: {caption}")

      # --- 데모 3: 모델 성능 비교 (개념) ---
      print("\n=== 데모 3: 모델 성능 비교 (개념) ===")
      print("실제로는 BLIP, ViT-GPT2, GIT 등 다양한 모델을 비교할 수 있습니다.")
      
      # 모의 모델 추가
      if not LIBRARIES_AVAILABLE:
          models_to_test["Mock ViT-GPT2"] = (MockImageCaptioningModel("Mock ViT-GPT2"), None, None)

      comparison_image = "images/people_park.jpg"
      print(f"비교 이미지: {comparison_image}")
      
      results = []
      for model_name, (model_obj, tok, fe) in models_to_test.items():
          cap, conf = get_caption(comparison_image, model_obj, tok, fe, device)
          results.append({"model": model_name, "caption": cap, "confidence": conf})
          
      print("\n--- 모델 비교 결과 ---")
      for res in results:
          print(f"  모델: {res['model']}")
          print(f"    캡션: {res['caption']}")
          print(f"    신뢰도: {res['confidence']:.3f}")

      print("\n\n=== 이미지 캡셔닝 해설 ===")
      print("1. **아키텍처:** 이미지 캡셔닝 모델은 주로 '인코더-디코더' 구조를 사용합니다.")
      print("   - **비전 인코더 (Vision Encoder):** 이미지의 특징을 추출합니다. (예: ViT, ResNet)")
      print("   - **언어 디코더 (Language Decoder):** 추출된 이미지 특징을 바탕으로 순차적으로 단어를 생성하여 문장을 만듭니다. (예: GPT-2, BERT)")
      print("2. **주요 모델:**")
      print("   - **ViT-GPT2:** ViT로 이미지 특징을, GPT-2로 캡션을 생성합니다.")
      print("   - **BLIP:** 부트스트래핑 기법을 사용하여 노이즈가 많은 웹 데이터에서도 효과적으로 학습합니다. 이미지-텍스트 매칭 및 생성 작업을 동시에 수행합니다.")
      print("3. **응용 분야:**")
      print("   - **접근성:** 시각 장애인을 위한 이미지 설명 (스크린 리더).")
      print("   - **콘텐츠 관리:** 이미지 검색, 자동 태깅, 콘텐츠 분류.")
      print("   - **소셜 미디어:** 이미지에 대한 자동 설명 및 해시태그 생성.")

  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_6_1_image_captioning.py`

---

### 문제 6.2: 확산 모델(Diffusion Model)을 이용한 텍스트-이미지 생성

- **문제:** 현재 이미지 생성 AI의 주류 기술인 '확산 모델'의 기본 원리를 이해하고, 텍스트 프롬프트만으로 고품질 이미지를 생성하는 과정을 실습합니다.
- **지시어:** Hugging Face의 `diffusers` 라이브러리를 사용하여 사전 훈련된 텍스트-이미지 생성 모델(예: Stable Diffusion)을 로드하세요. 이미지를 설명하는 상세한 텍스트 프롬프트를 작성하고, 이를 바탕으로 새로운 이미지를 생성하세요. 프롬프트의 내용을 변경하며 생성되는 이미지가 어떻게 달라지는지 실험해보세요.
- **제공데이터:**
  - 프롬프트: `"An astronaut riding a horse on Mars, photorealistic style."`
- **필요한 라이브러리:**
  ```bash
  pip install diffusers transformers accelerate torch
  ```
- **소스코드 내용:**
  ```python
  import torch
  import os
  from PIL import Image, ImageDraw, ImageFont

  try:
      from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
      LIBRARIES_AVAILABLE = True
      print("Diffusers 라이브러리가 사용 가능합니다.")
      print("이 코드는 CUDA 지원 GPU 환경에서 실행하는 것을 강력히 권장합니다.")
  except ImportError:
      LIBRARIES_AVAILABLE = False
      print("필요한 라이브러리(diffusers, transformers, accelerate, torch)가 설치되지 않았습니다.")

  def create_mock_image(prompt, filename="mock_image.png"):
      """프롬프트를 기반으로 간단한 모의 이미지를 생성하는 함수"""
      print(f"모의 이미지 생성 중... '{prompt}'")
      width, height = 512, 512
      img = Image.new('RGB', (width, height), color='darkgrey')
      draw = ImageDraw.Draw(img)
      
      try:
          # 시스템에 따라 사용 가능한 폰트 경로를 지정해야 할 수 있습니다.
          font = ImageFont.truetype("arial.ttf", 20)
      except IOError:
          font = ImageFont.load_default()

      # 프롬프트에서 키워드 추출
      keywords = prompt.lower().split(',')
      main_subject = keywords[0]
      
      # 키워드에 따라 배경색과 도형 변경
      if "astronaut" in main_subject:
          bg_color = "black"
          shape_color = "white"
          shape = "ellipse"
      elif "cat" in main_subject:
          bg_color = "lightblue"
          shape_color = "orange"
          shape = "rectangle"
      else:
          bg_color = "lightgreen"
          shape_color = "purple"
          shape = "rectangle"
          
      img = Image.new('RGB', (width, height), color=bg_color)
      draw = ImageDraw.Draw(img)
      
      # 도형 그리기
      if shape == "ellipse":
          draw.ellipse((100, 100, 400, 400), fill=shape_color)
      else:
          draw.rectangle((100, 100, 400, 400), fill=shape_color)
          
      # 텍스트 추가
      draw.text((10, 10), f"Mock Image for:\n{prompt}", font=font, fill="white" if bg_color == "black" else "black")
      
      img.save(filename)
      print(f"모의 이미지가 '{filename}' 파일로 저장되었습니다.")
      return img

  def main():
      """메인 실행 함수"""
      print("=== 문제 6.2: 확산 모델(Diffusion Model)을 이용한 텍스트-이미지 생성 ===")

      if LIBRARIES_AVAILABLE and torch.cuda.is_available():
          # --- 실제 Stable Diffusion 모델 사용 ---
          print("\n실제 Stable Diffusion 모델을 로드합니다. 시간이 걸릴 수 있습니다...")
          model_id = "runwayml/stable-diffusion-v1-5"
          scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
          pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
          pipe = pipe.to("cuda")
          print("모델 로드 완료.")

          prompts = {
              "우주비행사": "An astronaut riding a horse on Mars, photorealistic style, 4k",
              "귀여운 고양이": "A cute cat wearing a tiny wizard hat, sitting on a pile of ancient books, digital art",
              "미래 도시": "A futuristic city with flying cars and holographic billboards, synthwave style",
              "초현실적 풍경": "A surreal landscape with a floating island and a waterfall flowing upwards, style of Salvador Dali"
          }

          if not os.path.exists("generated_images"):
              os.makedirs("generated_images")

          for name, prompt in prompts.items():
              print(f"\n--- '{name}' 이미지 생성 중 ---")
              print(f"프롬프트: {prompt}")
              
              # 프롬프트를 기반으로 이미지 생성
              image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
              
              # 생성된 이미지 저장
              filename = f"generated_images/stable_diffusion_{name.replace(' ', '_')}.png"
              image.save(filename)
              print(f"이미지가 '{filename}' 파일로 저장되었습니다.")

      else:
          # --- 모의 이미지 생성 사용 ---
          if not LIBRARIES_AVAILABLE:
              print("\nDiffusers 라이브러리가 없어 모의 이미지 생성을 사용합니다.")
          elif not torch.cuda.is_available():
              print("\nCUDA 지원 GPU가 없어 모의 이미지 생성을 사용합니다. (CPU 실행은 매우 느립니다)")

          prompt_a = "An astronaut riding a horse on Mars, photorealistic style."
          create_mock_image(prompt_a, "mock_astronaut.png")
          
          prompt_b = "A cute cat wearing a tiny wizard hat, digital art."
          create_mock_image(prompt_b, "mock_cat.png")

      print("\n\n=== 확산 모델(Diffusion Model) 해설 ===")
      print("1. **핵심 아이디어:** 이미지에 점진적으로 노이즈를 추가하여 완전히 무작위한 노이즈로 만드는 '확산 과정(Forward Process)'을 거꾸로 되돌리는 방법을 학습합니다.")
      print("\n2. **생성 과정 (Reverse Process):**")
      print("   - **1단계 (입력):** 무작위 노이즈 이미지와 텍스트 프롬프트('우주비행사...')를 입력으로 받습니다.")
      print("   - **2단계 (노이즈 예측):** 모델(주로 U-Net 아키텍처)은 현재 이미지에서 어떤 노이즈를 제거해야 텍스트 프롬프트와 더 가까워지는지를 예측합니다.")
      print("   - **3단계 (점진적 노이즈 제거):** 예측된 노이즈의 일부를 이미지에서 제거합니다. 이 과정을 수십 번(예: 50회) 반복합니다.")
      print("   - **4단계 (최종 이미지):** 노이즈가 모두 제거되면 프롬프트에 해당하는 선명한 이미지가 나타납니다.")
      print("\n3. **텍스트 프롬프트의 역할:** 각 노이즈 제거 단계에서 모델이 올바른 방향(예: '말'의 형태, '화성'의 배경)으로 이미지를 다듬도록 안내하는 '가이드' 역할을 합니다. 이를 'Conditioning'이라고 합니다.")
      print("\n4. **주요 모델:** Stable Diffusion, Midjourney, DALL-E 2/3 등이 모두 확산 모델에 기반하고 있습니다.")

  if __name__ == "__main__":
      main()
  ```
- **소스코드명:** `llm_6_2_stable_diffusion.py`

---

## 전체 라이브러리 설치

아래 명령어를 사용하여 모든 실습에 필요한 라이브러리를 한 번에 설치할 수 있습니다.

```bash
pip install torch torchvision torchaudio
pip install transformers sentence-transformers scikit-learn pandas nltk Pillow requests
pip install langchain langchain-community faiss-cpu huggingface_hub
pip install datasets peft accelerate bitsandbytes trl
pip install diffusers
```

## NLTK 데이터 다운로드

Python 환경에서 아래 코드를 실행하여 NLTK에 필요한 데이터를 다운로드하세요.

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```
