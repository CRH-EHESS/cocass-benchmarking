from pathlib import Path
from matplotlib import pyplot as plt
import json
import os
import numpy as np
import pandas as pd

def inspect_class_repartition(json_file_paths, title="Class Repartition", return_dataframe=False, normalize=False,log_scale=False, legend_names=None):
    """
    Inspect the class repartition of a dataset given a json file or a list of json files.
    
    Args :
    json_file_paths (str or list of str): The path to the json file containing the dataset. Can be a list of json files.
    title (str): The title of the plot.
    return_dataframe (bool): If True, return a DataFrame containing the class repartition.
    normalize (bool): If True, normalize the class repartition by the total number of annotations for each json file given.
    log_scale (bool): If True, plot the class repartition in log scale. (useful for highly imbalanced datasets)
    legend_names (list of str): The names of the datasets to be displayed in the legend.
    """

    if isinstance(json_file_paths, str):
        json_file_paths = [json_file_paths]
    
    all_class_counts = []
    dataset_labels = []
    class_name_map = {}
    total_annotations = []

    for json_file_path in json_file_paths:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        class_counts = {}
        
        # Extract class names
        for category in data['categories']:
            class_name_map[category['id']] = category['name']
        
        for item in data['annotations']:
            class_id = item['category_id']
            
            # Ensure class_id is hashable
            if isinstance(class_id, set):
                class_id = tuple(class_id)
            
            if class_id not in class_counts:
                class_counts[class_id] = 0
            class_counts[class_id] += 1
        
        all_class_counts.append(class_counts)
        dataset_labels.append(os.path.basename(json_file_path))
        total_annotations.append(sum(class_counts.values()))
    
    # Get all unique class ids
    all_class_ids = set()
    for class_counts in all_class_counts:
        all_class_ids.update(class_counts.keys())
    
    all_class_ids = sorted(all_class_ids)
    all_class_names = [class_name_map[class_id] for class_id in all_class_ids]
    
    if return_dataframe:
        # Create a DataFrame
        data = {}
        for label, class_counts, total in zip(dataset_labels, all_class_counts, total_annotations):
            if normalize:
                data[label] = [class_counts.get(cls, 0) / total for cls in all_class_ids]
            else:
                data[label] = [class_counts.get(cls, 0) for cls in all_class_ids]
        df = pd.DataFrame(data, index=all_class_names)
        return df
    else:
        # Plotting the class repartition
        plt.figure(figsize=(10, 5))
        
        bar_width = 0.2
        spacing = 0.5
        
        if log_scale:
            plt.yscale('log')
        index = np.arange(len(all_class_ids))*(1+spacing)
        
        for i, (class_counts, total) in enumerate(zip(all_class_counts, total_annotations)):
            if normalize:
                counts = [class_counts.get(cls, 0) / total for cls in all_class_ids]
            else:
                counts = [class_counts.get(cls, 0) for cls in all_class_ids]
            plt.bar(index + i * bar_width, counts, bar_width, label=dataset_labels[i])
        
        label = 'Count' if not normalize else 'Normalized Count'
        if log_scale:
            label += ' (log scale)'
        if normalize:
            title = 'Normalized ' + title

        
        plt.title(title)
        plt.xlabel('Class')
        plt.ylabel(label)
        plt.xticks(index + bar_width * (len(all_class_counts) - 1) / 2, all_class_names, rotation=90)
        plt.legend()
        if legend_names:
            plt.legend(legend_names)
        plt.tight_layout()
        plt.show()

def inspect_image_origin(json_file_paths, keywords, title="Image Origin Distribution"):
    """
    Inspect the origin of the images in a dataset given a json file or a list of json files, if the dataset contains images of different temporary datsets.
    """
    
    if isinstance(json_file_paths, str):
        json_file_paths = [json_file_paths]
    
    image_origin_counts = {}
    
    for json_file_path in json_file_paths:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        for image in data['images']:
            image_name = image['file_name']
            for keyword in keywords:
                if keyword in image_name:
                    dataset_name = keyword
                    if dataset_name not in image_origin_counts:
                        image_origin_counts[dataset_name] = 0
                    image_origin_counts[dataset_name] += 1
    
    # Generate the pie chart
    labels = list(image_origin_counts.keys())
    sizes = list(image_origin_counts.values())

    title = title + ' in Validation set'
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)  # Fix: Pass a format string as autopct
    plt.title(title)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()       