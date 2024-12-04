from sklearn.metrics import roc_auc_score

# Hàm để tính các chỉ số: Accuracy, Sensitivity, Specificity
def calc_metrics(output, label):
    output = output.astype(int)  # Chuyển output về kiểu int
    label = label.astype(int)

    TP = ((output == 1) & (label == 1)).sum()
    TN = ((output == 0) & (label == 0)).sum()
    FP = ((output == 1) & (label == 0)).sum()
    FN = ((output == 0) & (label == 1)).sum()

    # Accuracy
    acc = (TP + TN) / (TP + TN + FP + FN)
    
    # Sensitivity (Recall or True Positive Rate)
    if (TP + FN) > 0:
        se = TP / (TP + FN)
    else:
        se = 0

    # Specificity (True Negative Rate)
    if (TN + FP) > 0:
        sp = TN / (TN + FP)
    else:
        sp = 0

    return acc, se, sp

# Hàm tính AUC (Area Under the ROC Curve)
def calc_auc(output, label):
    return roc_auc_score(label, output)
