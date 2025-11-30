import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error

from Zh.src.utils.common import has_multiple_mappings, order_dict_by_key_list


def report(model, 
           x, 
           y_true, 
           group_identifiers, 
           unique_classes):

    y_true = np.array(y_true)
    y_pred = np.array(model.predict(x, verbose=0)).flatten()

    for i in y_true:
        if i not in unique_classes:
            raise ValueError(f"Not all classes are mentioned in unique_classes. --{i}--")
    if has_multiple_mappings(y_true, group_identifiers):
        raise ValueError("Groups don't match with Targets.")
    if not len(y_true) == len(y_pred) == len(group_identifiers):
        raise ValueError("Inputs y_true, y_pred, and group_identifiers must have the same length.")

    # Sort unique classes and calculate thresholds
    unique_classes = sorted(unique_classes)
    thresholds     = [(unique_classes[i] + unique_classes[i + 1]) / 2 for i in range(len(unique_classes)-1)]
    unique_classes = [str(float(i)) for i in unique_classes]
    
    def assign_class(value, thresholds, unique_classes):
        assert len(thresholds) + 1 == len(unique_classes)
        assert all(isinstance(i, (int, float)) for i in thresholds)
        unique_classes = [float(i) for i in unique_classes]
        return next((unique_classes[i] for i, t in enumerate(thresholds) if value < t), unique_classes[-1])

    def calculate_classification_metrics(y_true, y_pred, unique_classes):
        y_true = y_true.astype(float).astype(str)
        y_pred = y_pred.astype(float).astype(str)

        results = classification_report(
            y_true        = y_true,
            y_pred        = y_pred,
            labels        = unique_classes,
            output_dict   = True,
            zero_division = False
        )

        for cls in unique_classes:
            filter = y_true == cls
            if sum(filter) == 0:
                results[cls] = {
                    'precision': 0.0,
                    'recall'   : 0.0,
                    'f1-score' : 0.0,
                    'mae'      : 0.0,
                    'support'  : 0.0}
                continue

            results[cls]['mae'] = mean_absolute_error(
                y_true = [cls]*sum(filter), 
                y_pred = y_pred[(y_true == cls)].to_list()
                )
            
            results[cls] = order_dict_by_key_list(results[cls], 
                                                  ['precision', 'recall', 'f1-score', 'mae', 'support'])

        results['confusion matrix'] = confusion_matrix(
            y_true = y_true, 
            y_pred = y_pred, 
            labels=unique_classes)
        
        results['mae'] = mean_absolute_error(
            y_true = y_true,
            y_pred = y_pred
        )

        results = order_dict_by_key_list(results, 
                                         unique_classes + ['accuracy', 'mae', 'macro avg', 'weighted avg', 'confusion matrix'])
        return results

    #Calculating classification metrics for slices
    slice_data = pd.DataFrame({'y_true'           : y_true, 
                               'y_pred'           : y_pred,
                               'group_identifiers': group_identifiers})

    slice_data['y_pred_class'] = slice_data['y_pred'].apply(lambda x: assign_class(x, thresholds, unique_classes))
    slice_data['y_true']       = slice_data['y_true'].astype(float).astype(str)
    slice_data['y_pred']       = slice_data['y_pred'].astype(float).astype(str)
    slice_data['y_pred_class'] = slice_data['y_pred_class'].astype(float).astype(str)

    slice_results               = calculate_classification_metrics(slice_data['y_true'], slice_data['y_pred_class'], unique_classes)

    #Calculating classification metrics for videos
    video_data = pd.DataFrame(columns=['group_identifier', 'y_true', 'y_pred_majority', 'y_pred_average', 'y_pred_class_average'])  

    for name, group in slice_data.groupby('group_identifiers'):
        video_data = pd.concat([video_data, pd.DataFrame({
            'group_identifier'    : [name],
            'y_true'              : [group['y_true'].unique()[0]],
            'y_pred_majority'     : [group['y_pred_class'].mode()[0]],
            'y_pred_average'      : [str(float(assign_class(group['y_pred'].astype(float).mean(), thresholds, unique_classes)))],
            'y_pred_class_average': [str(float(assign_class(group['y_pred_class'].astype(float).mean(), thresholds, unique_classes)))],
        })], ignore_index=True)

    video_results_majority      = calculate_classification_metrics(video_data['y_true'], video_data['y_pred_majority'], unique_classes)
    video_results_average       = calculate_classification_metrics(video_data['y_true'], video_data['y_pred_average'], unique_classes)
    video_results_class_average = calculate_classification_metrics(video_data['y_true'], video_data['y_pred_class_average'], unique_classes)

    return (
        [float(i) for i in y_pred],
        {
            "slice": slice_results,
            "video": {
                "majority": video_results_majority,
                "average": video_results_average,
                "class_average": video_results_class_average,
            },
        },
    )