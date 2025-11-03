"""
문제 5.2: 감정 분석 (Sentiment Analysis) 

요구사항:
1. 텍스트 전처리
2. TF-IDF 벡터화
3. 나이브 베이즈 분류
4. 성능 평가
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

print("="*60)
print("문제 5.2: 감정 분석 (Sentiment Analysis)")
print("="*60)

# 1. 샘플 텍스트 데이터
print("\n[1] 샘플 텍스트 데이터 준비")
print("-" * 60)

texts = [
    "이 영화 정말 좋아요! 최고의 작품입니다.",
    "최악의 영화네요. 시간 낭비했습니다.",
    "훌륭한 배우와 멋진 스토리. 강력 추천합니다.",
    "정말 실망했습니다. 별로 재미없어요.",
    "멋진 영화예요. 다시 보고 싶습니다.",
    "이렇게 나쁜 영화는 처음입니다.",
    "완벽한 영화! 정말 감동했습니다.",
    "지루하고 따분한 영화입니다.",
    "매우 재미있고 감동적이었습니다.",
    "돈 낭비했어요. 다시는 안 봅니다.",
    "멋진 감독의 뛰어난 작품입니다.",
    "너무 길고 복잡합니다. 추천 안 합니다.",
    "긍정적이고 희망적인 이야기입니다.",
    "부정적인 메시지가 많네요.",
    "정말 훌륭합니다! 모두 봐야 해요.",
    "평범하고 흥미로운 점이 없습니다.",
    "최고의 영화 중 하나입니다!",
    "낮은 등급을 받을만합니다.",
    "아름다운 영상과 훌륭한 연기입니다.",
    "정말 형편없는 작품입니다."
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

print(f"총 샘플 수: {len(texts)}")
print(f"긍정: {sum(labels)}, 부정: {len(labels) - sum(labels)}")
print(f"\n샘플:")
for i in range(3):
    sentiment = "긍정" if labels[i] == 1 else "부정"
    print(f"  [{sentiment}] {texts[i]}")


# 2. 데이터 분할
print("\n[2] 데이터 분할")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

print(f"훈련 데이터: {len(X_train)}")
print(f"테스트 데이터: {len(X_test)}")


# 3. TF-IDF 벡터화
print("\n[3] TF-IDF 벡터화")
print("-" * 60)

vectorizer = TfidfVectorizer(max_features=50, lowercase=True, 
                            stop_words=['이', '는', '을', '를', '이', '했', '합니다'])
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"벡터화 후 특성 수: {X_train_tfidf.shape[1]}")
print(f"훈련 데이터 형태: {X_train_tfidf.shape}")
print(f"테스트 데이터 형태: {X_test_tfidf.shape}")

# 상위 특성 확인
feature_names = np.array(vectorizer.get_feature_names_out())
print(f"\n추출된 단어 예시 (처음 10개):")
print(feature_names[:10])


# 4. 나이브 베이즈 모델 훈련
print("\n[4] 나이브 베이즈 분류 모델 훈련")
print("-" * 60)

clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

print("✓ 모델 훈련 완료")


# 5. 예측 및 평가
print("\n[5] 예측 및 성능 평가")
print("-" * 60)

y_pred = clf.predict(X_test_tfidf)
y_pred_proba = clf.predict_proba(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy:.4f}")

print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=['부정', '긍정']))

cm = confusion_matrix(y_test, y_pred)
print("\n혼동행렬:")
print(f"[[TN={cm[0,0]}, FP={cm[0,1]}],")
print(f" [FN={cm[1,0]}, TP={cm[1,1]}]]")


# 6. 샘플 예측
print("\n[6] 새로운 텍스트 감정 분석")
print("-" * 60)

test_phrases = [
    "이 영화 정말 최고예요!",
    "너무 실망했습니다.",
    "흥미로운 작품입니다.",
]

print("새 텍스트에 대한 예측:")
for phrase in test_phrases:
    phrase_tfidf = vectorizer.transform([phrase])
    pred_label = clf.predict(phrase_tfidf)[0]
    pred_proba = clf.predict_proba(phrase_tfidf)[0]
    sentiment = "긍정" if pred_label == 1 else "부정"
    confidence = pred_proba[pred_label] * 100
    print(f"  '{phrase}'")
    print(f"    → {sentiment} (신뢰도: {confidence:.1f}%)\n")


# 7. 시각화
print("\n[7] 결과 시각화")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 혼동행렬
ax = axes[0]
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set_title('혼동행렬', fontweight='bold', fontsize=12)
ax.set_ylabel('실제', fontsize=11)
ax.set_xlabel('예측', fontsize=11)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['부정', '긍정'])
ax.set_yticklabels(['부정', '긍정'])

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black',
                fontsize=14, fontweight='bold')

# 특성 중요도 (클래스별)
ax = axes[1]
feature_importance = np.array(clf.feature_log_prob_[1] - clf.feature_log_prob_[0])
top_indices = np.argsort(np.abs(feature_importance))[-10:]
top_features = feature_names[top_indices]
top_scores = feature_importance[top_indices]

colors = ['green' if score > 0 else 'red' for score in top_scores]
ax.barh(range(len(top_features)), top_scores, color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features)
ax.set_xlabel('특성 중요도', fontsize=11)
ax.set_title('상위 10개 특성 (긍정/부정)', fontweight='bold', fontsize=12)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('problem_5_2_sentiment_analysis.png', dpi=100)
print("✓ 시각화 저장: problem_5_2_sentiment_analysis.png")
plt.close()

print("\n" + "="*60)
print("감정 분석 완료")
print("="*60)
