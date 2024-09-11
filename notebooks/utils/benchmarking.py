import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

"""
Contains utility functions to handle the results from sahi or fiftyone to compare the models
"""

# import data_cleaning
def load_coco_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def iou(box1, box2,sizes=None):
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    if sizes is not None:
        w1,w2 = w1*sizes[0],w2*sizes[0]
        h1,h2 = h1*sizes[1],h2*sizes[1]
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    return iou
def results2pd(results,
               verbose : bool = True):
    """
    Convert results to pandas dataframe

    Parameters:
    - results: The results object containing the evaluation metrics

    Returns:
    - dataframe: A pandas dataframe containing the evaluation metrics for each class and overall
    """
    if verbose :
        dataframe=pd.DataFrame()
        dataframe["Class"]=[name for name in results.names.values()]
        #dataframe["Images"]=results.box.n
        #dataframe["Support"]=results.box.support
        dataframe["Precision"]=results.box.p
        dataframe["Recall"]=results.box.r
        dataframe["F1"]=results.box.f1
        dataframe["AP"]=results.box.ap
        dataframe["mAP50"]=results.box.ap50
        dataframe["mAP50-95"]=results.box.maps

        dataframe.loc["All_classes"]=None
        dataframe.at["All_classes", "Precision"]=results.box.mp
        dataframe.at["All_classes", "Recall"]=results.box.mr
        dataframe.at["All_classes", "mAP50"]=results.box.map50
        dataframe.at["All_classes", "mAP50-95"]=results.box.map
        dataframe.at["All_classes", "F1"]=dataframe["F1"].mean()
        dataframe.at["All_classes", "AP"]=dataframe["AP"].mean()
        dataframe.at["All_classes", "Class"]="All_classes"
        dataframe.reset_index(drop=True, inplace=True)
    return dataframe

def fo_result2pd(results,file_path : str = "results.json",
               verbose : bool = True,
               save_json : bool = False,
               save_csv : bool = False,
               csv_file_path : str = "classification_report.csv",
               save_xlsx : bool = False,
               xlsx_file_path : str = "classification_report.xlsx"):
    
    """
    Convert results to pandas dataframe

    Args:
    - results: The results object containing the evaluation metrics
    - file_path (str): Path to save the JSON file containing the results
    - verbose (bool): If True, display the classification report DataFrame
    - save_json (bool): If True, save the JSON file containing the results
    - save_csv (bool): If True, save the classification report as a CSV file
    - csv_file_path (str): Path to save the CSV file containing the classification report

    Returns:
    - dataframe: A pandas dataframe containing the evaluation metrics for each class and overall
    """
    

    results.write_json(file_path)

    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extracting ground truth and predictions from the dataset
    ground_truth = data['ytrue']
    predictions = data['ypred']

    # Generate classification report for the full dataset
    report = classification_report(ground_truth, predictions, output_dict=True)

    # Convert the report to a DataFrame for better visualization
    report_df = pd.DataFrame(report).transpose()

    # Format the numerical columns to 4 decimal places
    report_df = report_df.map(lambda x: round(x, 3) if isinstance(x, (int, float)) else x)


    # rename (none) as False Negatives
    report_df.rename(index={"(none)": 'False Negatives'}, inplace=True)

    if verbose:
        # Display the DataFrame
        print(report_df)
    if not save_json:
        os.remove(file_path)

    if save_csv:
        report_df.to_csv(csv_file_path, index=False)

    if save_xlsx:
        with pd.ExcelWriter(xlsx_file_path) as writer:
            report_df.to_excel(writer,sheet_name='classification_report')

    return report_df


def fo_plot_confusion_matrix(results, file_path: str = "results.json",
                             save_json: bool = False,
                             normalize=False, title='Confusion Matrix'):
    """
    Plot the confusion matrix from the JSON file containing 'ytrue' and 'ypred'.

    Parameters:
    json_path (str): Path to the JSON file containing 'ytrue' and 'ypred'.
    normalize (bool): If True, normalize the confusion matrix.
    title (str): Title for the confusion matrix plot.

    Returns:
    None: The function directly plots the confusion matrix.
    """
    results.write_json(file_path)

    # Load the JSON data
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if not save_json:
        os.remove(file_path)
    
    # Extract ground truth and predictions
    y_true = data['ytrue']
    y_pred = data['ypred']
    
    # Get the list of unique labels
    labels = sorted(list(set(y_true + y_pred)))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_title = 'Normalized ' + title
    else:
        cm_title = title

    # Create custom annotations
    annot = np.where(cm == 0, '', cm.round(2).astype(str))

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(cm_title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

import numpy as np

def build_confusion_matrix(gt_json, pred_json, iou_threshold=0.5):
    gt_data = load_coco_json(gt_json)
    pred_data = load_coco_json(pred_json)
    
    gt_annotations = gt_data['annotations']
    pred_annotations = pred_data['annotations']

    id2id= {cat['id']:id for id,cat in enumerate(gt_data['categories'])}

    categories = {cat['id']: cat['name'] for cat in gt_data['categories']}
    num_categories = len(categories)
    
    # Add the background category with a new ID as num_categories + 1
    background_id = num_categories + 1
    categories[background_id] = "background"
    id2id= {cat_id:id for id,cat_id in enumerate(categories)}
    # Initialize the confusion matrix, adding an extra row/column for the background category
    matrix = np.zeros((num_categories + 1, num_categories + 1))
    
    # Create mappings for quick access to images by ID
    gt_images = {img['id']: img for img in gt_data['images']}
    pred_images = {img['id']: img for img in pred_data['images']}
    
    # First loop: update the matrix based on ground truth annotations
    for gt in gt_annotations:
        matched = False
        gt_image_file = gt_images[gt['image_id']]['file_name']
        
        for pred in pred_annotations:
            pred_image_file = pred_images[pred['image_id']]['file_name']
            if gt_image_file == pred_image_file:
                gt_box = gt['bbox']
                pred_box = pred['bbox']
                gt_category_id = id2id[gt['category_id']]
                pred_category_id = id2id[pred['category_id']]
                
                if iou(gt_box, pred_box) >= iou_threshold:
                    matrix[gt_category_id][pred_category_id] += 1
                    matched = True
                    break
        
        if not matched:
            matrix[id2id[gt_category_id]-1][id2id[background_id]] += 1
    
    # Second loop: update the matrix based on predictions
    for pred in pred_annotations:
        matched = False
        pred_image_file = pred_images[pred['image_id']]['file_name']
        
        for gt in gt_annotations:
            gt_image_file = gt_images[gt['image_id']]['file_name']
            if gt_image_file == pred_image_file:
                pred_box = pred['bbox']
                gt_box = gt['bbox']
                pred_category_id = id2id[pred['category_id']]
                gt_category_id = id2id[gt['category_id']]
                
                if iou(gt_box, pred_box) >= iou_threshold:
                    matched = True
                    break
        
        if not matched:
            matrix[id2id[background_id]][id2id[pred_category_id]] += 1
    
    # No need to reorder the matrix, we simply return it along with the categories as is
    return matrix, categories



def plot_confusion_matrix(matrix, categories, normalize=False):
    """Plot the confusion matrix with an option to normalize."""
    category_names = [categories[i] for i in sorted(categories.keys())]
    
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        matrix = np.nan_to_num(matrix)  # Replace NaNs with 0s for rows with no data
    
    # Custom annotation function to avoid displaying 0.00
    def format_value(value):
        if value == 0:
            return ""
        return f"{value:.2f}" if normalize else f"{int(value)}"
    
    # Convert matrix to string format with custom annotation
    annot = np.array([[format_value(matrix[i, j]) for j in range(matrix.shape[1])] for i in range(matrix.shape[0])])

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=annot, fmt="", cmap="Blues", xticklabels=category_names, yticklabels=category_names)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
    plt.show()