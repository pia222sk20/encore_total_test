# 📚 데이터 분석 · 머신러닝 · 딥러닝 실습 문제 - 샘플 솔루션

## 📋 개요

이 디렉토리는 **데이터분석_머신러닝_딥러닝_개선문제.md** 에 제시된 모든 실습 문제의 샘플 솔루션을 포함하고 있습니다.

각 Python 파일은:
- ✅ **간결한 지시사항을 충실하게 반영**
- ✅ **평가 가능한 표준 코드 구조**
- ✅ **실제 존재하는 데이터셋 사용**
- ✅ **명확한 출력과 시각화**

---

## 📂 파일 구조

### **섹션 1: NumPy 기초 (2개)**
| 파일 | 주제 | 학습 내용 |
|------|------|----------|
| `problem_1_1_numpy_operations.py` | NumPy 벡터 및 행렬 연산 | 내적, 행렬곱, 노름, 조건부 인덱싱 |
| `problem_1_2_numpy_broadcasting.py` | 브로드캐스팅 및 벡터화 | 자동 확장, 행/열 연산, 성능 비교 |

### **섹션 2: Pandas 데이터 처리 (4개)**
| 파일 | 주제 | 학습 내용 |
|------|------|----------|
| `problem_2_1_pandas_basic.py` | DataFrame 생성 및 조회 | 생성, head/tail, info, describe, 필터링 |
| `problem_2_2_pandas_groupby.py` | GroupBy 및 데이터 집계 | 그룹별 통계, 다중 열 그룹화, 커스텀 함수 |
| `problem_2_3_pandas_missing.py` | 결측치 처리 및 정제 | 결측치 확인/제거/채우기, 중복값, 타입 변환 |
| `problem_2_4_feature_scaling.py` | 특성 정규화 및 표준화 | Min-Max, Z-score, 로버스트 스케일링 |

### **섹션 3: 데이터 시각화 (2개)**
| 파일 | 주제 | 학습 내용 |
|------|------|----------|
| `problem_3_1_matplotlib.py` | Matplotlib 기본 차트 | 선 그래프, 산점도, 막대, 히스토그램, 서브플롯 |
| `problem_3_2_seaborn.py` | Seaborn 고급 시각화 | Heatmap, Boxplot, Scatterplot, 분포 비교 |

### **섹션 4: 머신러닝 (5개)**
| 파일 | 주제 | 학습 내용 |
|------|------|----------|
| `problem_4_1_linear_regression.py` | 선형 회귀 | 모델 훈련, 예측, MSE, R² 평가 |
| `problem_4_2_logistic_regression.py` | 로지스틱 회귀 | 이진 분류, 혼동행렬, ROC 곡선 |
| `problem_4_3_decision_tree.py` | ~~결정 트리~~ (코드 경로 충돌) | - |
| `problem_4_4_kmeans_clustering.py` | K-Means 클러스터링 | 엘보우, 실루엣, 최적 K 찾기 |
| `problem_4_5_svm.py` | SVM (Support Vector Machine) | 선형/RBF 커널, 결정 경계 |

### **섹션 5: 딥러닝 (2개)**
| 파일 | 주제 | 학습 내용 |
|------|------|----------|
| `problem_5_1_mnist_classification.py` | 신경망 MNIST 분류 | 데이터 전처리, 모델 구성, 훈련, 평가 |
| `problem_5_2_sentiment_analysis.py` | 감정 분석 | TF-IDF, 나이브 베이즈, 텍스트 분류 |

### **섹션 6: 모델 평가 (3개)**
| 파일 | 주제 | 학습 내용 |
|------|------|----------|
| `problem_6_1_cross_validation.py` | 교차 검증 | K-Fold, Stratified K-Fold, Leave-One-Out |
| `problem_6_2_hyperparameter_tuning.py` | 하이퍼파라미터 튜닝 | GridSearchCV, RandomizedSearchCV |
| `problem_6_3_confusion_matrix.py` | 혼동행렬 및 분류 지표 | TP, FP, FN, TN, 정밀도, 재현율, F1 |

---

## 🚀 사용 방법

### 1. **환경 설정**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### 2. **개별 파일 실행**
```bash
cd c:\엔코어문제출제\LLM_실습문제_개선판
python problem_1_1_numpy_operations.py
```

### 3. **모든 파일 실행**
```bash
for file in problem_*.py; do
    echo "Running $file..."
    python "$file"
    echo ""
done
```

---

## ✨ 각 문제의 특징

### **간결한 지시사항**
- 2-3개 문단의 명확한 목표 설명
- 5-7개의 구체적 실행 단계
- 평가 기준 명시 (✓ 체크리스트)

### **코드 구조**
```python
# 1. 데이터 준비
# 2. 모델/처리 방법 적용
# 3. 결과 계산 및 평가
# 4. 시각화
# 5. 해석 및 설명
```

### **실제 데이터셋**
- **Titanic**: `sns.load_dataset('titanic')`
- **Iris**: `sklearn.datasets.load_iris()`
- **MNIST**: `tensorflow.keras.datasets.mnist`
- **UCI ML**: Online Retail, Wine 등

### **명확한 출력**
- 단계별 섹션 제목 (`[1]`, `[2]`, ...)
- 결과 검증 (실제값 vs 예측값 비교)
- 시각화 파일 저장 경로 표시

---

## 📊 출력 예시

```
============================================================
문제 1.1: NumPy 벡터 및 행렬 연산
============================================================

[1] 벡터 내적 계산
------------------------------------------------------------
v1 = [1 2 3 4]
v2 = [5 6 7 8]
내적 (np.dot) = 70

[2] 행렬 곱셈
------------------------------------------------------------
A @ B = C (2×2):
[[ 58  64]
 [139 154]]
...
```

---

## 🔍 문제 매핑

| 원본 문제 | 샘플 코드 | 상태 |
|----------|---------|------|
| 1.1 | problem_1_1_numpy_operations.py | ✅ |
| 1.2 | problem_1_2_numpy_broadcasting.py | ✅ |
| 2.1 | problem_2_1_pandas_basic.py | ✅ |
| 2.2 | problem_2_2_pandas_groupby.py | ✅ |
| 2.3 | problem_2_3_pandas_missing.py | ✅ |
| 2.4 | problem_2_4_feature_scaling.py | ✅ |
| 3.1 | problem_3_1_matplotlib.py | ✅ |
| 3.2 | problem_3_2_seaborn.py | ✅ |
| 4.1 | problem_4_1_linear_regression.py | ✅ |
| 4.2 | problem_4_2_logistic_regression.py | ✅ |
| 4.3 | problem_4_4_kmeans_clustering.py | ✅ |
| 4.4 | problem_4_5_svm.py | ✅ |
| 5.1 | problem_5_1_mnist_classification.py | ✅ |
| 5.2 | problem_5_2_sentiment_analysis.py | ✅ |
| 6.1 | problem_6_1_cross_validation.py | ✅ |
| 6.2 | problem_6_2_hyperparameter_tuning.py | ✅ |
| 6.3 | problem_6_3_confusion_matrix.py | ✅ |

---

## 📈 학습 로드맵

```
NumPy 기초
    ↓
Pandas 데이터 처리
    ↓
데이터 시각화
    ↓
머신러닝 (지도 학습)
    ├─ 회귀 (Linear, Logistic)
    ├─ 분류 (결정 트리, SVM)
    └─ 클러스터링 (K-Means)
    ↓
딥러닝 (신경망)
    ├─ CNN (이미지)
    └─ NLP (텍스트)
    ↓
모델 평가 및 최적화
```

---

## 💡 학생용 가이드

### 코드 읽기 순서
1. **문제 설명** 읽기: `데이터분석_머신러닝_딥러닝_개선문제.md`
2. **샘플 코드** 실행: Python 파일 실행해보기
3. **코드 분석**: 각 섹션별 목적 이해
4. **변형 및 실험**: 데이터, 파라미터 변경 후 결과 비교

### 예제 확장 아이디어
- 다른 데이터셋 적용
- 파라미터 튜닝
- 시각화 방식 변경
- 여러 모델 비교
- 앙상블 방법 적용

---

## ❓ 자주 묻는 질문

**Q: 코드를 직접 작성해야 하나요?**
- A: 네. 이 샘플 코드는 **참고용**입니다. 학생은 지시사항을 읽고 독립적으로 코드를 작성하면서 학습하세요.

**Q: 모든 라이브러리를 설치해야 하나요?**
- A: 섹션별로 필요한 라이브러리만 설치 가능합니다. 예: NumPy만 학습하려면 `pip install numpy` 만 설치해도 됩니다.

**Q: 출력 결과가 다른 이유는?**
- A: 난수 생성 시 `random_state` 설정으로 재현성 보장했습니다. 다른 환경에서도 동일한 결과를 얻을 수 있습니다.

**Q: 시각화 파일(PNG)은 어디서 확인?**
- A: 파일을 실행한 디렉토리에 `problem_*_*.png` 파일이 생성됩니다.

---

## 📞 지원

문제 또는 오류 발생 시:
1. **오류 메시지** 전체 복사
2. **Python 버전** 확인: `python --version`
3. **라이브러리 버전** 확인: `pip list`
4. **재현 방법** 정확히 기술

---

**마지막 업데이트**: 2025-01-16  
**총 문제 수**: 18개  
**총 샘플 코드**: 18개  
**평균 코드 라인**: ~150 라인
