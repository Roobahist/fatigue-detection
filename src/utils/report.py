import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
)


def report(y_pred, y_true, unique_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    thresholds = []

    # Sort unique classes and calculate thresholds
    unique_classes = sorted(unique_classes)
    for i in range(len(unique_classes) - 1):
        thresholds.append((unique_classes[i] + unique_classes[i + 1]) / 2)

    y_pred_classified = np.array([unique_classes[i] 
                                  for i in np.digitize(y_pred, thresholds, right=False)])
    
    unique_classes    = [str(float(i)) for i in unique_classes]
    y_true            = np.array([str(float(i)) for i in y_true])
    y_pred            = np.array([str(float(i)) for i in y_pred])
    y_pred_classified = np.array([str(float(i)) for i in y_pred_classified])


    classification_report_results = classification_report(y_true        = y_true,
                                                          y_pred        = y_pred_classified,
                                                          zero_division = False,
                                                          output_dict   = True)

    for cls in unique_classes:
        idx = y_true == cls
        if cls not in list(classification_report_results.keys()) or sum(idx) == 0:
            classification_report_results[cls] = {
                'precision': 0.0,
                'recall'   : 0.0,
                'f1-score' : 0.0,
                'mae'      : 0.0,
                'support'  : 0.0}
            continue
            
        y_true_cls = [float(i) for i in y_true[idx]]
        y_pred_cls = [float(i) for i in y_pred[idx]]
        mae = mean_absolute_error(y_true = y_true_cls, y_pred = y_pred_cls)
        classification_report_results[cls]['mae'] = mae
        classification_report_results[cls] = {i:classification_report_results[cls][i] 
                                              for i in ['precision', 'recall', 'f1-score', 'mae', 'support']}
    
    confusion_matrix_results = confusion_matrix(y_true=y_true, 
                                                y_pred=y_pred_classified)
    
    classification_report_results['confusion matrix'] = confusion_matrix_results

    classification_report_results = {i:classification_report_results[i] 
                                     for i in unique_classes + ['accuracy', 'macro avg', 'weighted avg', 'confusion matrix']}

    return [float(i) for i in y_pred], classification_report_results