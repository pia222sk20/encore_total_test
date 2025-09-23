"""
문제 4.4: SVM을 이용한 분류
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

def problem_4_4():
    print("=== 문제 4.4: SVM을 이용한 분류 ===")
    
    # 1. 데이터 로드
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"데이터 형태: {X.shape}")
    print(f"클래스 수: {len(np.unique(y))}")
    print(f"이미지 크기: {digits.images.shape[1]}x{digits.images.shape[2]}")
    print(f"클래스별 분포:")
    print(pd.Series(y).value_counts().sort_index())
    
    # 2. 샘플 이미지 시각화
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        ax = axes[i//5, i%5]
        # 각 숫자의 첫 번째 이미지 찾기
        idx = np.where(y == i)[0][0]
        ax.imshow(digits.images[idx], cmap='gray')
        ax.set_title(f'Digit: {i}')
        ax.axis('off')
    plt.suptitle('Sample Digits (0-9)')
    plt.tight_layout()
    plt.show()
    
    # 3. 데이터 전처리
    print(f"\n=== 데이터 전처리 ===")
    
    # 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"정규화 전 범위: [{X.min():.1f}, {X.max():.1f}]")
    print(f"정규화 후 범위: [{X_scaled.min():.1f}, {X_scaled.max():.1f}]")
    
    # 4. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
    
    # 5. PCA 차원 축소 (시각화 및 성능 비교용)
    print(f"\n=== PCA 차원 축소 ===")
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)
    
    # 누적 설명 분산 비율
    cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum_ratio >= 0.95) + 1
    n_components_99 = np.argmax(cumsum_ratio >= 0.99) + 1
    
    print(f"95% 분산 설명하는 컴포넌트 수: {n_components_95}")
    print(f"99% 분산 설명하는 컴포넌트 수: {n_components_99}")
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 21), pca.explained_variance_ratio_[:20], 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, 51), cumsum_ratio[:50], 'ro-')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95%')
    plt.axhline(y=0.99, color='b', linestyle='--', label='99%')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 6. 기본 SVM 모델
    print(f"\n=== 기본 SVM 모델 (RBF 커널) ===")
    svm_basic = SVC(kernel='rbf', random_state=42)
    svm_basic.fit(X_train, y_train)
    y_pred_basic = svm_basic.predict(X_test)
    
    accuracy_basic = accuracy_score(y_test, y_pred_basic)
    print(f"기본 SVM 정확도: {accuracy_basic:.4f}")
    
    # 7. 하이퍼파라미터 튜닝
    print(f"\n=== 하이퍼파라미터 튜닝 ===")
    
    # 작은 데이터셋으로 그리드 서치
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    # 일부 데이터로 빠른 그리드 서치
    X_small = X_train[:500]
    y_small = y_train[:500]
    
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_small, y_small)
    
    print(f"최적 파라미터: {grid_search.best_params_}")
    print(f"최적 CV 점수: {grid_search.best_score_:.4f}")
    
    # 8. 최적화된 SVM 모델
    print(f"\n=== 최적화된 SVM 모델 ===")
    svm_optimized = SVC(**grid_search.best_params_, random_state=42)
    svm_optimized.fit(X_train, y_train)
    y_pred_optimized = svm_optimized.predict(X_test)
    
    accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
    print(f"최적화된 SVM 정확도: {accuracy_optimized:.4f}")
    print(f"성능 향상: {accuracy_optimized - accuracy_basic:.4f}")
    
    # 9. 커널별 성능 비교
    print(f"\n=== 커널별 성능 비교 ===")
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    kernel_scores = {}
    
    for kernel in kernels:
        svm_kernel = SVC(kernel=kernel, random_state=42)
        svm_kernel.fit(X_train, y_train)
        score = svm_kernel.score(X_test, y_test)
        kernel_scores[kernel] = score
        print(f"{kernel.upper()} 커널: {score:.4f}")
    
    # 커널 성능 시각화
    plt.figure(figsize=(8, 5))
    kernels_list = list(kernel_scores.keys())
    scores_list = list(kernel_scores.values())
    
    bars = plt.bar(kernels_list, scores_list, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('SVM Performance by Kernel Type')
    plt.ylim(0.9, 1.0)
    
    # 막대 위에 점수 표시
    for bar, score in zip(bars, scores_list):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # 10. 혼동 행렬
    cm = confusion_matrix(y_test, y_pred_optimized)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.title('Confusion Matrix - Digit Classification')
    plt.show()
    
    # 11. 분류 보고서
    print(f"\n=== Classification Report ===")
    print(classification_report(y_test, y_pred_optimized))
    
    # 12. 잘못 분류된 이미지 분석
    wrong_indices = np.where(y_pred_optimized != y_test)[0]
    print(f"\n잘못 분류된 이미지 수: {len(wrong_indices)}")
    
    if len(wrong_indices) > 0:
        plt.figure(figsize=(15, 6))
        for i, idx in enumerate(wrong_indices[:10]):  # 처음 10개만
            original_idx = X_test.shape[0] * 0.7 + idx  # 원본 인덱스 추정
            actual_idx = int(len(digits.data) * 0.7) + idx  # 실제 테스트 인덱스
            
            plt.subplot(2, 5, i + 1)
            # 테스트 데이터를 다시 이미지 형태로 변환
            test_image = scaler.inverse_transform(X_test[idx].reshape(1, -1))
            test_image = test_image.reshape(8, 8)
            
            plt.imshow(test_image, cmap='gray')
            plt.title(f'True: {y_test[idx]}, Pred: {y_pred_optimized[idx]}')
            plt.axis('off')
        
        plt.suptitle('Misclassified Digits')
        plt.tight_layout()
        plt.show()
    
    # 13. 클래스별 정확도
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(10), class_accuracy, color='skyblue', alpha=0.7)
    plt.xlabel('Digit')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(range(10))
    
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== 클래스별 정확도 ===")
    for digit, acc in enumerate(class_accuracy):
        print(f"숫자 {digit}: {acc:.4f}")
    
    return svm_optimized, grid_search.best_params_

if __name__ == "__main__":
    model, best_params = problem_4_4()