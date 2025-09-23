"""
문제 1.2: NumPy 브로드캐스팅을 이용한 데이터 표준화
"""
import numpy as np
import matplotlib.pyplot as plt

def problem_1_2():
    print("=== 문제 1.2: NumPy 브로드캐스팅을 이용한 데이터 표준화 ===")
    
    # 1. 2차원 배열 생성 (학생 성적 데이터 가정)
    np.random.seed(42)
    scores = np.random.randint(60, 100, size=(5, 4))  # 5명 학생, 4과목 성적
    subjects = ['수학', '영어', '과학', '사회']
    students = ['학생A', '학생B', '학생C', '학생D', '학생E']
    
    print("원본 데이터 (5명 학생, 4과목 성적):")
    print("과목:", subjects)
    for i, student in enumerate(students):
        print(f"{student}: {scores[i]}")
    print(f"데이터 형태: {scores.shape}")
    print("-" * 50)
    
    # 2. 기본 통계 계산
    print("=== 기본 통계 ===")
    print(f"전체 평균: {np.mean(scores):.2f}")
    print(f"전체 표준편차: {np.std(scores):.2f}")
    print(f"최댓값: {np.max(scores)}")
    print(f"최솟값: {np.min(scores)}")
    print("-" * 50)
    
    # 3. 과목별 통계 (axis=0: 행 방향으로 계산)
    subject_means = np.mean(scores, axis=0)
    subject_stds = np.std(scores, axis=0)
    
    print("=== 과목별 통계 (axis=0) ===")
    for i, subject in enumerate(subjects):
        print(f"{subject}: 평균 {subject_means[i]:.2f}, 표준편차 {subject_stds[i]:.2f}")
    print("-" * 50)
    
    # 4. 학생별 통계 (axis=1: 열 방향으로 계산)
    student_means = np.mean(scores, axis=1)
    student_stds = np.std(scores, axis=1)
    
    print("=== 학생별 통계 (axis=1) ===")
    for i, student in enumerate(students):
        print(f"{student}: 평균 {student_means[i]:.2f}, 표준편차 {student_stds[i]:.2f}")
    print("-" * 50)
    
    # 5. 브로드캐스팅을 이용한 과목별 표준화
    print("=== 과목별 표준화 (Z-score) ===")
    print("공식: (점수 - 과목평균) / 과목표준편차")
    
    # 브로드캐스팅: (5,4) - (4,) = (5,4), (5,4) / (4,) = (5,4)
    standardized_by_subject = (scores - subject_means) / subject_stds
    
    print("표준화된 데이터 (과목별):")
    for i, student in enumerate(students):
        print(f"{student}: {standardized_by_subject[i]}")
    
    # 표준화 후 과목별 평균과 표준편차 확인
    print(f"\n표준화 후 과목별 평균: {np.mean(standardized_by_subject, axis=0)}")
    print(f"표준화 후 과목별 표준편차: {np.std(standardized_by_subject, axis=0)}")
    print("-" * 50)
    
    # 6. 학생별 표준화
    print("=== 학생별 표준화 ===")
    print("공식: (점수 - 학생평균) / 학생표준편차")
    
    # reshape 또는 전치를 이용한 브로드캐스팅
    student_means_reshaped = student_means.reshape(-1, 1)  # (5,1) 형태로 변환
    student_stds_reshaped = student_stds.reshape(-1, 1)    # (5,1) 형태로 변환
    
    standardized_by_student = (scores - student_means_reshaped) / student_stds_reshaped
    
    print("표준화된 데이터 (학생별):")
    for i, student in enumerate(students):
        print(f"{student}: {standardized_by_student[i]}")
    
    # 표준화 후 학생별 평균과 표준편차 확인
    print(f"\n표준화 후 학생별 평균: {np.mean(standardized_by_student, axis=1)}")
    print(f"표준화 후 학생별 표준편차: {np.std(standardized_by_student, axis=1)}")
    print("-" * 50)
    
    # 7. 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 원본 데이터 히트맵
    im1 = axes[0, 0].imshow(scores, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('원본 점수')
    axes[0, 0].set_xlabel('과목')
    axes[0, 0].set_ylabel('학생')
    axes[0, 0].set_xticks(range(4))
    axes[0, 0].set_xticklabels(subjects)
    axes[0, 0].set_yticks(range(5))
    axes[0, 0].set_yticklabels(students)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 과목별 표준화 히트맵
    im2 = axes[0, 1].imshow(standardized_by_subject, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    axes[0, 1].set_title('과목별 표준화')
    axes[0, 1].set_xlabel('과목')
    axes[0, 1].set_ylabel('학생')
    axes[0, 1].set_xticks(range(4))
    axes[0, 1].set_xticklabels(subjects)
    axes[0, 1].set_yticks(range(5))
    axes[0, 1].set_yticklabels(students)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 학생별 표준화 히트맵
    im3 = axes[1, 0].imshow(standardized_by_student, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    axes[1, 0].set_title('학생별 표준화')
    axes[1, 0].set_xlabel('과목')
    axes[1, 0].set_ylabel('학생')
    axes[1, 0].set_xticks(range(4))
    axes[1, 0].set_xticklabels(subjects)
    axes[1, 0].set_yticks(range(5))
    axes[1, 0].set_yticklabels(students)
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 브로드캐스팅 개념 설명 그래프
    axes[1, 1].text(0.1, 0.9, "브로드캐스팅 규칙:", fontsize=12, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.8, "1. 뒤에서부터 차원 비교", fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, "2. 크기가 1이면 확장 가능", fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, "3. 차원이 없으면 1로 추가", fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, "", fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, "예시:", fontsize=11, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.3, "(5,4) - (4,) → (5,4) - (1,4) → OK", fontsize=9, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.2, "(5,4) - (5,1) → OK", fontsize=9, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.1, "(5,4) - (3,) → Error", fontsize=9, transform=axes[1, 1].transAxes)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 8. 브로드캐스팅 실험
    print("=== 브로드캐스팅 실험 ===")
    
    # 실험 1: 스칼라와 배열
    print("실험 1: 스칼라와 배열")
    arr = np.array([1, 2, 3, 4])
    scalar = 10
    result1 = arr + scalar
    print(f"{arr} + {scalar} = {result1}")
    
    # 실험 2: 1차원 배열들
    print("\n실험 2: 1차원 배열들")
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([10, 20, 30])
    result2 = arr1 + arr2
    print(f"{arr1} + {arr2} = {result2}")
    
    # 실험 3: 2차원과 1차원
    print("\n실험 3: 2차원과 1차원")
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    arr1d = np.array([10, 20, 30])
    result3 = arr2d + arr1d
    print(f"{arr2d}\n+\n{arr1d}\n=\n{result3}")
    
    # 실험 4: reshape을 이용한 브로드캐스팅
    print("\n실험 4: reshape을 이용한 브로드캐스팅")
    arr1d_col = arr1d.reshape(-1, 1)  # 열벡터로 변환
    result4 = arr2d + arr1d_col
    print(f"{arr2d}\n+\n{arr1d_col}\n=\n{result4}")
    
    return {
        'original_scores': scores,
        'subject_standardized': standardized_by_subject,
        'student_standardized': standardized_by_student,
        'subject_stats': (subject_means, subject_stds),
        'student_stats': (student_means, student_stds)
    }

if __name__ == "__main__":
    results = problem_1_2()