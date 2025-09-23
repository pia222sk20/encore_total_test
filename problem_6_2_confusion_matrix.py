"""
문제 6.2: 혼동 행렬 기반 평가 지표

요구사항:
1. 혼동 행렬(Confusion Matrix) 계산 및 시각화
2. 정밀도(Precision), 재현율(Recall), F1-score 계산
3. ROC 곡선 및 AUC 계산
4. 다중 클래스 분류에서의 평가 지표
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                           roc_curve, auc, roc_auc_score, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics_manually(y_true, y_pred):
    """혼동 행렬로부터 수동으로 평가 지표 계산"""
    cm = confusion_matrix(y_true, y_pred)
    
    # 이진 분류의 경우
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        return {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'accuracy': accuracy,
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn
        }
    
    # 다중 클래스의 경우
    else:
        n_classes = len(np.unique(y_true))
        class_metrics = {}
        
        for i in range(n_classes):
            # 클래스 i를 positive로, 나머지를 negative로 간주
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[f'class_{i}'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': np.sum(cm[i, :])
            }
        
        # 전체 정확도
        accuracy = np.trace(cm) / np.sum(cm)
        
        return {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'class_metrics': class_metrics
        }

def plot_confusion_matrices(y_true_binary, y_pred_binary, y_true_multi, y_pred_multi, class_names=None):
    """혼동 행렬 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 이진 분류 혼동 행렬
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'],
                ax=axes[0])
    axes[0].set_title('Binary Classification\nConfusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # 다중 클래스 혼동 행렬
    cm_multi = confusion_matrix(y_true_multi, y_pred_multi)
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(y_true_multi)))]
    
    sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title('Multi-class Classification\nConfusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

def plot_roc_curves(y_true_binary, y_prob_binary, y_true_multi, y_prob_multi):
    """ROC 곡선 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 이진 분류 ROC 곡선
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob_binary[:, 1])
    roc_auc = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Binary Classification ROC Curve')
    axes[0].legend(loc="lower right")
    axes[0].grid(True)
    
    # 다중 클래스 ROC 곡선 (One-vs-Rest)
    n_classes = y_prob_multi.shape[1]
    colors = ['red', 'blue', 'green']
    
    for i in range(n_classes):
        y_true_class = (y_true_multi == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_class, y_prob_multi[:, i])
        roc_auc = auc(fpr, tpr)
        
        axes[1].plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'Class {i} (AUC = {roc_auc:.3f})')
    
    axes[1].plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Multi-class ROC Curves (One-vs-Rest)')
    axes[1].legend(loc="lower right")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curves(y_true_binary, y_prob_binary, y_true_multi, y_prob_multi):
    """Precision-Recall 곡선 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 이진 분류 PR 곡선
    precision, recall, _ = precision_recall_curve(y_true_binary, y_prob_binary[:, 1])
    avg_precision = np.trapz(precision, recall)
    
    axes[0].plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Binary Classification\nPrecision-Recall Curve')
    axes[0].legend(loc="lower left")
    axes[0].grid(True)
    
    # 다중 클래스 PR 곡선
    n_classes = y_prob_multi.shape[1]
    colors = ['red', 'blue', 'green']
    
    for i in range(n_classes):
        y_true_class = (y_true_multi == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_class, y_prob_multi[:, i])
        avg_precision = np.trapz(precision, recall)
        
        axes[1].plot(recall, precision, color=colors[i], lw=2,
                    label=f'Class {i} (AP = {avg_precision:.3f})')
    
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Multi-class Precision-Recall Curves')
    axes[1].legend(loc="lower left")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    print("=== 문제 6.2: 혼동 행렬 기반 평가 지표 ===")
    
    # === 1. 이진 분류 데이터 생성 ===
    print("\n=== 1. 이진 분류 평가 ===")
    X_binary, y_binary = make_classification(n_samples=1000, n_features=20, 
                                           n_redundant=10, n_informative=10,
                                           n_clusters_per_class=1, random_state=42)
    
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_binary, y_binary, test_size=0.3, random_state=42)
    
    # 모델 훈련
    rf_binary = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_binary.fit(X_train_bin, y_train_bin)
    
    # 예측
    y_pred_bin = rf_binary.predict(X_test_bin)
    y_prob_bin = rf_binary.predict_proba(X_test_bin)
    
    # 수동 평가 지표 계산
    manual_metrics = calculate_metrics_manually(y_test_bin, y_pred_bin)
    
    print("이진 분류 혼동 행렬:")
    print(manual_metrics['confusion_matrix'])
    print(f"\n수동 계산 결과:")
    print(f"정확도 (Accuracy): {manual_metrics['accuracy']:.4f}")
    print(f"정밀도 (Precision): {manual_metrics['precision']:.4f}")
    print(f"재현율 (Recall): {manual_metrics['recall']:.4f}")
    print(f"특이도 (Specificity): {manual_metrics['specificity']:.4f}")
    print(f"F1-Score: {manual_metrics['f1_score']:.4f}")
    
    print(f"\n혼동 행렬 구성 요소:")
    print(f"True Positive: {manual_metrics['true_positive']}")
    print(f"True Negative: {manual_metrics['true_negative']}")
    print(f"False Positive: {manual_metrics['false_positive']}")
    print(f"False Negative: {manual_metrics['false_negative']}")
    
    # Scikit-learn과 비교
    print(f"\nScikit-learn 결과와 비교:")
    print(f"정확도: {accuracy_score(y_test_bin, y_pred_bin):.4f}")
    print(f"정밀도: {precision_score(y_test_bin, y_pred_bin):.4f}")
    print(f"재현율: {recall_score(y_test_bin, y_pred_bin):.4f}")
    print(f"F1-Score: {f1_score(y_test_bin, y_pred_bin):.4f}")
    
    # ROC-AUC
    auc_score = roc_auc_score(y_test_bin, y_prob_bin[:, 1])
    print(f"ROC-AUC: {auc_score:.4f}")
    
    # === 2. 다중 클래스 분류 ===
    print("\n" + "="*60)
    print("=== 2. 다중 클래스 분류 평가 ===")
    
    # Iris 데이터셋 사용
    iris = load_iris()
    X_multi, y_multi = iris.data, iris.target
    
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.3, random_state=42)
    
    # 모델 훈련
    rf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_multi.fit(X_train_multi, y_train_multi)
    
    # 예측
    y_pred_multi = rf_multi.predict(X_test_multi)
    y_prob_multi = rf_multi.predict_proba(X_test_multi)
    
    # 수동 평가 지표 계산
    multi_metrics = calculate_metrics_manually(y_test_multi, y_pred_multi)
    
    print("다중 클래스 혼동 행렬:")
    print(multi_metrics['confusion_matrix'])
    print(f"\n전체 정확도: {multi_metrics['accuracy']:.4f}")
    
    print(f"\n클래스별 평가 지표:")
    for class_name, metrics in multi_metrics['class_metrics'].items():
        print(f"{class_name}:")
        print(f"  정밀도: {metrics['precision']:.4f}")
        print(f"  재현율: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Support: {metrics['support']}")
    
    # Scikit-learn의 classification_report
    print(f"\nScikit-learn Classification Report:")
    print(classification_report(y_test_multi, y_pred_multi, 
                              target_names=iris.target_names))
    
    # === 3. 시각화 ===
    print("\n" + "="*60)
    print("=== 3. 시각화 ===")
    
    # 혼동 행렬 시각화
    plot_confusion_matrices(y_test_bin, y_pred_bin, y_test_multi, y_pred_multi,
                           class_names=iris.target_names)
    
    # ROC 곡선 시각화
    plot_roc_curves(y_test_bin, y_prob_bin, y_test_multi, y_prob_multi)
    
    # Precision-Recall 곡선 시각화
    plot_precision_recall_curves(y_test_bin, y_prob_bin, y_test_multi, y_prob_multi)
    
    # === 4. 고급 평가 지표 ===
    print("\n" + "="*60)
    print("=== 4. 고급 평가 지표 ===")
    
    # 가중 평균 계산 (다중 클래스)
    precision_weighted = precision_score(y_test_multi, y_pred_multi, average='weighted')
    recall_weighted = recall_score(y_test_multi, y_pred_multi, average='weighted')
    f1_weighted = f1_score(y_test_multi, y_pred_multi, average='weighted')
    
    precision_macro = precision_score(y_test_multi, y_pred_multi, average='macro')
    recall_macro = recall_score(y_test_multi, y_pred_multi, average='macro')
    f1_macro = f1_score(y_test_multi, y_pred_multi, average='macro')
    
    precision_micro = precision_score(y_test_multi, y_pred_multi, average='micro')
    recall_micro = recall_score(y_test_multi, y_pred_multi, average='micro')
    f1_micro = f1_score(y_test_multi, y_pred_multi, average='micro')
    
    print("다중 클래스 평균 지표:")
    print(f"Weighted Average - Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")
    print(f"Macro Average - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
    print(f"Micro Average - Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1: {f1_micro:.4f}")
    
    # 클래스 불균형 시나리오 분석
    print(f"\n=== 클래스 불균형 영향 분석 ===")
    
    # 불균형 데이터 생성
    X_imb, y_imb = make_classification(n_samples=1000, n_features=20,
                                     n_redundant=10, weights=[0.9, 0.1],
                                     flip_y=0.01, random_state=42)
    
    X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
        X_imb, y_imb, test_size=0.3, random_state=42)
    
    rf_imb = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_imb.fit(X_train_imb, y_train_imb)
    y_pred_imb = rf_imb.predict(X_test_imb)
    
    print(f"클래스 분포: {np.bincount(y_test_imb)}")
    print(f"정확도: {accuracy_score(y_test_imb, y_pred_imb):.4f}")
    print(f"균형 정확도: {recall_score(y_test_imb, y_pred_imb, average='macro'):.4f}")
    print(f"F1-Score (macro): {f1_score(y_test_imb, y_pred_imb, average='macro'):.4f}")
    print(f"F1-Score (weighted): {f1_score(y_test_imb, y_pred_imb, average='weighted'):.4f}")
    
    print("------------------------------------------------------------")
    print("=== 평가 지표 해석 가이드 ===")
    print("1. 정밀도 (Precision): 예측한 Positive 중 실제 Positive 비율")
    print("2. 재현율 (Recall): 실제 Positive 중 예측한 Positive 비율")
    print("3. F1-Score: 정밀도와 재현율의 조화 평균")
    print("4. 특이도 (Specificity): 실제 Negative 중 예측한 Negative 비율")
    print("5. ROC-AUC: ROC 곡선 아래 면적 (0.5~1.0, 높을수록 좋음)")
    print("6. Macro 평균: 클래스별 지표의 단순 평균")
    print("7. Weighted 평균: 클래스 크기로 가중된 평균")
    print("8. Micro 평균: 전체 예측의 정확도와 동일")

if __name__ == "__main__":
    main()