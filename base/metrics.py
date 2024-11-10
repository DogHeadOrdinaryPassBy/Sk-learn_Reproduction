import numpy as np

class Metrics:
    def __init__(self, y_true, y_pred=None, y_scores=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_scores = y_scores

    def accuracy_score(self):
        """计算预测的准确率"""
        if self.y_pred is None:
            raise ValueError("y_pred 不能为空")
        return np.mean(self.y_true == self.y_pred)

    def confusion_matrix(self):
        """计算混淆矩阵"""
        if self.y_pred is None:
            raise ValueError("y_pred 不能为空")
        unique_labels = np.unique(np.concatenate((self.y_true, self.y_pred)))
        matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}

        for true_label, pred_label in zip(self.y_true, self.y_pred):
            matrix[label_map[true_label], label_map[pred_label]] += 1

        return matrix

    def roc_curve(self):
        """计算 ROC 曲线"""
        if self.y_scores is None:
            raise ValueError("y_scores 不能为空")
        thresholds = np.sort(np.unique(self.y_scores))[::-1]
        tprs, fprs = [], []
        for threshold in thresholds:
            y_pred = (self.y_scores >= threshold).astype(int)
            tp = np.sum((y_pred == 1) & (self.y_true == 1))
            fp = np.sum((y_pred == 1) & (self.y_true == 0))
            fn = np.sum((y_pred == 0) & (self.y_true == 1))
            tn = np.sum((y_pred == 0) & (self.y_true == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tprs.append(tpr)
            fprs.append(fpr)

        return np.array(fprs), np.array(tprs), thresholds

    def auc_score(self, fpr, tpr):
        """计算 AUC"""
        return np.trapz(tpr, fpr)
