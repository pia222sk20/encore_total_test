"""
문제 3.2: 생성 파라미터 제어

요구사항:
1. 온도 파라미터 변화 실험
2. Top-P 샘플링 비교
3. 다양성 평가
4. 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

print("="*60)
print("문제 3.2: 생성 파라미터 제어")
print("="*60)

# 시뮬레이션된 생성 결과
# (실제로는 LLM API 호출로 생성)

print("\n[1] 온도(Temperature) 파라미터 효과")
print("-" * 60)

temperature_results = {
    0.1: "The future of AI is bright and promising. Technology will advance rapidly.",
    0.5: "The future of AI is quite promising. We will see new breakthroughs soon.",
    1.0: "The future of AI brings both challenges and opportunities for society.",
    1.5: "The future of AI, with quantum computing, could reshape everything weirdly."
}

print("프롬프트: 'The future of AI is'")
print("\n온도별 생성 결과:")
for temp, text in temperature_results.items():
    print(f"\n  온도 {temp} (일관성: {'높음' if temp < 0.5 else '중간' if temp < 1.0 else '낮음'})")
    print(f"    → {text}")


# Top-P 샘플링
print("\n[2] Top-P (Nucleus Sampling) 효과")
print("-" * 60)

topk_results = {
    "P=0.5": "The future of AI is bright and promising.",
    "P=0.7": "The future of AI is quite optimistic and transformative.",
    "P=0.9": "The future of AI brings both opportunities and challenges.",
    "P=0.95": "The future of AI, with quantum computing, could reshape society."
}

print("프롬프트: 'The future of AI is'")
print("\nTop-P 값별 결과:")
for p, text in topk_results.items():
    print(f"\n  {p}: {text}")


# 다양성 평가
print("\n[3] 생성 다양성 평가")
print("-" * 60)

# 시뮬레이션: 각 설정으로 5번 생성했을 때의 어휘 다양성
diversity_metrics = {
    "T=0.1": {"고유_단어": 12, "평균_길이": 15, "반복율": 0.25},
    "T=0.5": {"고유_단어": 18, "평균_길이": 16, "반복율": 0.20},
    "T=1.0": {"고유_단어": 22, "평균_길이": 18, "반복율": 0.15},
    "T=1.5": {"고유_단어": 28, "평균_길이": 19, "반복율": 0.10},
}

print("\n| 설정 | 고유 단어 | 평균 길이 | 반복율 |")
print("|------|----------|---------|-------|")
for setting, metrics in diversity_metrics.items():
    print(f"| {setting} | {metrics['고유_단어']} | {metrics['평균_길이']} | {metrics['반복율']:.0%} |")


# 시각화
print("\n[4] 파라미터 효과 시각화")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 온도별 다양성
temperatures = [0.1, 0.5, 1.0, 1.5]
unique_words = [12, 18, 22, 28]
repeat_rates = [0.25, 0.20, 0.15, 0.10]

ax = axes[0]
ax2 = ax.twinx()

line1 = ax.plot(temperatures, unique_words, 'o-', color='blue', linewidth=2, markersize=8, label='고유 단어 수')
line2 = ax2.plot(temperatures, repeat_rates, 's-', color='red', linewidth=2, markersize=8, label='반복율')

ax.set_xlabel('온도 (Temperature)', fontsize=11)
ax.set_ylabel('고유 단어 수', color='blue', fontsize=11)
ax2.set_ylabel('반복율', color='red', fontsize=11)
ax.set_title('온도에 따른 생성 다양성', fontweight='bold', fontsize=12)
ax.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='red')
ax.grid(True, alpha=0.3)
ax.set_xticks(temperatures)

# Top-P별 비교
p_values = [0.5, 0.7, 0.9, 0.95]
p_diversity = [20, 24, 26, 28]

ax = axes[1]
ax.bar(range(len(p_values)), p_diversity, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax.set_xlabel('Top-P 값', fontsize=11)
ax.set_ylabel('생성 다양성 점수', fontsize=11)
ax.set_title('Top-P에 따른 다양성', fontweight='bold', fontsize=12)
ax.set_xticks(range(len(p_values)))
ax.set_xticklabels([f'P={p}' for p in p_values])
ax.set_ylim([0, 30])
ax.grid(True, alpha=0.3, axis='y')

# 값 표시
for i, v in enumerate(p_diversity):
    ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('llm_3_2_generation_parameters.png', dpi=100)
print("✓ 시각화 저장: llm_3_2_generation_parameters.png")
plt.close()


# 권장 설정
print("\n[5] 파라미터 권장 설정")
print("-" * 60)

recommendations = {
    "정확한 정보 (사실)": {
        "온도": 0.1,
        "Top-P": 0.5,
        "용도": "기술 문서, 뉴스, 정보 조회",
        "특징": "일관성 있고 반복 가능한 답변"
    },
    "균형잡힌 생성": {
        "온도": 0.7,
        "Top-P": 0.9,
        "용도": "일반적인 대화, 질문 응답",
        "특징": "적절한 다양성과 안정성"
    },
    "창의적 생성": {
        "온도": 1.2,
        "Top-P": 0.95,
        "용도": "글쓰기, 시 작성, 아이디어 브레인스토밍",
        "특징": "다양하고 창의적인 결과"
    }
}

for use_case, settings in recommendations.items():
    print(f"\n{use_case}:")
    print(f"  온도: {settings['온도']}")
    print(f"  Top-P: {settings['Top-P']}")
    print(f"  용도: {settings['용도']}")
    print(f"  특징: {settings['특징']}")

print("\n" + "="*60)
print("생성 파라미터 분석 완료")
print("="*60)
