import json
from pathlib import Path
import os

"""
This module contains utility functions for cleaning and filtering COCO-style JSON files.

You will also find a function to remap image IDs and update annotations based on the categories in another JSON file. 

There is as well the function to add the ambiguity matrix to the annotations.
"""

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

def is_bbox_inside_another(inner_box, outer_box, threshold=0.1):
    """
    Check if the inner_box is mostly inside the outer_box, allowing for a threshold.
    
    Parameters:
    inner_box (list): [x, y, width, height] of the inner bounding box.
    outer_box (list): [x, y, width, height] of the outer bounding box.
    threshold (float): The fraction of the inner box that can be outside the outer box.
    
    Returns:
    bool: True if inner_box is mostly inside outer_box, False otherwise.
    """
    x1_inner, y1_inner, w1_inner, h1_inner = inner_box
    x2_inner, y2_inner = x1_inner + w1_inner, y1_inner + h1_inner
    
    x1_outer, y1_outer, w1_outer, h1_outer = outer_box
    x2_outer, y2_outer = x1_outer + w1_outer, y1_outer + h1_outer
    
    # Calculate the area of the inner box that is outside the outer box
    outside_area = 0
    if x1_inner < x1_outer:
        outside_area += (x1_outer - x1_inner) * h1_inner
    if y1_inner < y1_outer:
        outside_area += (y1_outer - y1_inner) * w1_inner
    if x2_inner > x2_outer:
        outside_area += (x2_inner - x2_outer) * h1_inner
    if y2_inner > y2_outer:
        outside_area += (y2_inner - y2_outer) * w1_inner
    
    inner_area = w1_inner * h1_inner
    return outside_area / inner_area <= threshold

def remap_image_ids(json_file_path2,json_file_path1=None,inplace=True):
    # Load the first JSON file
    with open(json_file_path2, 'r') as f:
        data2 = json.load(f)

    if json_file_path1 is not None:
    # Load the second JSON file
        with open(json_file_path1, 'r') as f:
            data1 = json.load(f)

        # Create a mapping from the image names to the new image ids
        image_name_to_id_mapping = {image['file_name']: image['id'] for image in data1['images']}
        # Create a mapping from the original image ids to the new image ids based on the image names
        image_id_mapping = {image['id']: image_name_to_id_mapping[image['file_name']] for image in data2['images']}
    else:
        # Create a mapping from the original image ids to the new image ids
        image_id_mapping = {image['id']: i for i, image in enumerate(data2['images'])}
    
    # Remap the image ids
    for image in data2['images']:
        image['id'] = image_id_mapping[image['id']]
    
    # Remap the image ids in the annotations
    for annotation in data2['annotations']:
        annotation['image_id'] = image_id_mapping[annotation['image_id']]
    
    # Save the new json file
    if not inplace:
        json_file_path2 = os.path.join(os.path.dirname(json_file_path2), 'remapped_' + os.path.basename(json_file_path2))
    with open(json_file_path2, 'w') as file:
        json.dump(data2, file)

def update_annotations_based_on_first_json(json_file_path1, json_file_path2,inplace=False,verbose=True):
    """
    Update the category IDs in the annotations of the second JSON file based on the categories in the first JSON file.
    
    Parameters:
    - json_file_path1 (str or Path): Path to the first JSON file (source of categories).
    - json_file_path2 (str or Path): Path to the second JSON file (target to update annotations).
    
    Returns:
    - updated_json_file_path2 (Path): Path to the updated second JSON file.
    """
    
    # Load the first JSON file
    with open(json_file_path1, 'r') as f:
        data1 = json.load(f)

    # Load the second JSON file
    with open(json_file_path2, 'r') as f:
        data2 = json.load(f)

    # Extract the categories from the first JSON file and create a mapping
    category_mapping = {cat['name']: cat['id'] for cat in data1['categories']}

    # Initialize a mapping from old category IDs to new category IDs for the second JSON file
    old_to_new_category_mapping = {}
    for cat in data2['categories']:
        if cat['name'] in category_mapping:
            old_to_new_category_mapping[cat['id']] = category_mapping[cat['name']]

    # Update the annotations in the second JSON file with the new category IDs
    updated_annotations = []
    for annotation in data2['annotations']:
        old_category_id = annotation['category_id']
        new_category_id = old_to_new_category_mapping.get(old_category_id)
        if new_category_id is not None:
            annotation['category_id'] = new_category_id
            updated_annotations.append(annotation)

    # Replace the annotations in the second JSON file with the updated ones
    data2['annotations'] = updated_annotations

    # Replace the categories in the second JSON file with those from the first JSON file
    data2['categories'] = data1['categories']

    # Save the updated second JSON file
    json_file_path2 = Path(json_file_path2)
    updated_json_file_path2 = json_file_path2.with_name(json_file_path2.stem + '_updated.json')
    if inplace :
        updated_json_file_path2 = json_file_path2
    with open(updated_json_file_path2, 'w') as f:
        json.dump(data2, f, indent=4)
    if verbose:
        print("Annotations in the second JSON file have been updated successfully.")
    return updated_json_file_path2

def fiftyone_extraction_remapping(json_file_path1,json_file_path2,inplace=True,
                                  rename=False, name=None):

    remap_image_ids(json_file_path1=json_file_path2,
                    json_file_path2=json_file_path1,inplace=inplace)
    print("Image IDs and annotations have been remapped successfully.")

    update_annotations_based_on_first_json(json_file_path1=json_file_path2,
                                           json_file_path2=json_file_path1,inplace=inplace,verbose=False)
    print("Annotations in the second JSON file have been updated successfully.")

    if rename:
        if name is None:
            os.rename(json_file_path2, os.path.join(os.path.dirname(json_file_path2),'.json'))
        os.rename(json_file_path2, name)
        print("File has been renamed successfully.")

def add_ambiguity_metric(json_file_path,inplace=True):
    from utils.benchmarking import iou
    import json

    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    for ann1 in data['annotations']:
        ambiguity=0
        adjacents=0
        for ann2 in data['annotations']:
            if ann1['image_id']==ann2['image_id']:
                iou_value=iou(ann1['bbox'],ann2['bbox'])
                ambiguity+=iou_value*(1-abs(ann1['score']-ann2['score']))
                # add adjacent if the iou is not 0
                if iou_value>0:
                    adjacents+=1
        ambiguity=(ambiguity/adjacents)*(1-ann1['score'])
        ann1['ambiguity']=ambiguity
    if not inplace:
        with open(os.path.join(os.path.dirname(json_file_path),"_ambiguity"), 'w') as file:
            json.dump(data, file)
    else:
        with open(json_file_path, 'w') as file:
            json.dump(data, file)

def confidence_filtering(data=None,json_file_path = None,
                         threshold:float=0.5,
                         inplace: bool = False,
                         keep_annotations: bool = False,
                         save: bool = True,data_garbage=None):
    """
    Filter out annotations with confidence below the threshold.
    """
    if data is None:
        if json_file_path is None:
            raise ValueError("Either data or json_file_path must be provided.")
        else:
            data = load_coco_json(json_file_path)
    if data_garbage is None:       
        data_garbage = data.copy()
        data_garbage['annotations'] = []
    # Filter out annotations with confidence below the threshold
    filtered_annotations = [ann for ann in data['annotations'] if ann['score'] >= threshold]

    # Update the annotations in the data
    data_copy = data.copy()
    data_copy['annotations'] = filtered_annotations

    # Save the updated JSON file
    if save and json_file_path is not None:
        if not inplace:
            json_file_path = os.path.join(os.path.dirname(json_file_path), 'filtered_' + os.path.basename(json_file_path))
        with open(json_file_path, 'w') as f:
            json.dump(data, f)
        print("Annotations have been filtered successfully.")

    
    if keep_annotations:
        garbage = [ann for ann in data['annotations'] if ann['score'] < threshold]
        for ann in garbage:
            if ann not in data_garbage['annotations']:
                data_garbage['annotations'].append(ann)
        if save and json_file_path is not None:
            with open(os.path.join(os.path.dirname(json_file_path), 'conf_garbage_' + os.path.basename(json_file_path)), 'w') as f:
                json.dump(data_garbage, f)
            print("Garbage annotations have been saved successfully.")

    return data_copy, data_garbage

def aspect_ratio_filtering(data=None,json_file_path = None,
                           height_width_ratio : float = 1.0,
                           below : bool = True,
                           inplace: bool = False,
                           keep_annotations: bool = False,
                           save: bool = True,data_garbage=None):
    """
    Filter out annotations based on the aspect ratio of their bounding boxes.
    """
    if data is None:
        if json_file_path is None:
            raise ValueError("Either data or json_file_path must be provided.")
        else:
            data = load_coco_json(json_file_path)

    if data_garbage is None:
        data_garbage = data.copy()
        data_garbage['annotations'] = []

    # Filter out annotations based on the aspect ratio of their bounding boxes
    if below:
        filtered_annotations = [ann for ann in data['annotations'] if (ann['bbox'][3] / ann['bbox'][2]) > height_width_ratio]
        if keep_annotations:
            garbage = [ann for ann in data['annotations'] if (ann['bbox'][3] / ann['bbox'][2]) <= height_width_ratio]
    else:
        filtered_annotations = [ann for ann in data['annotations'] if (ann['bbox'][3] / ann['bbox'][2]) < height_width_ratio]
        if keep_annotations:
            garbage = [ann for ann in data['annotations'] if (ann['bbox'][3] / ann['bbox'][2]) >= height_width_ratio]

    data_copy = data.copy()
    data_copy['annotations'] = filtered_annotations

    if save and json_file_path is not None:
        if not inplace:
            json_file_path = os.path.join(os.path.dirname(json_file_path), 'filtered_' + os.path.basename(json_file_path))
        with open(json_file_path, 'w') as f:
            json.dump(data_copy, f)      

    if keep_annotations:
        for ann in garbage:
            if ann not in data_garbage['annotations']:
                data_garbage['annotations'].append(ann)
        if save :
            with open(os.path.join(os.path.dirname(json_file_path), 'aspect_garbage_' + os.path.basename(json_file_path)), 'w') as f:
                json.dump(data_garbage, f)
            print("Garbage annotations have been saved successfully.")

    return data_copy, data_garbage

def iou_filtering(data=None,json_file_path = None,
                  iou_threshold:float=0.5,
                  inplace: bool = False,
                  keep_annotations: bool = False,
                  save: bool = True,
                  class_agnostic: bool = False,data_garbage=None):
    """
    Filter out annotations based on the IoU between their bounding boxes.
    """
    if data is None:
        if json_file_path is None:
            raise ValueError("Either data or json_file_path must be provided.")
        else:
            data = load_coco_json(json_file_path)
    if data_garbage is None:
            data_garbage = data.copy()
            data_garbage['annotations'] = []

    # Filter out annotations based on the IoU between their bounding boxes
    filtered_annotations = []
    garbage = []

    for ann1 in data['annotations']:
        iou_flag = False
        for ann2 in data['annotations']:
            if ann1 != ann2:
                if ann1['image_id'] != ann2['image_id']:
                    continue
                if not class_agnostic and ann1['category_id'] != ann2['category_id']:
                    continue
                if iou(ann1['bbox'], ann2['bbox']) >= iou_threshold:
                    if ann1['score'] < ann2['score']:
                        # print(f"Annotation {ann1['id']} has IoU with annotation {ann2['id']} above the threshold.")
                        iou_flag = True
                        continue
        if not iou_flag:
            filtered_annotations.append(ann1)
        else:
            garbage.append(ann1)

    data_copy = data.copy()
    data_copy['annotations'] = filtered_annotations

    data_garbage['annotations'] = garbage

    if not inplace and json_file_path is not None:
        json_file_path = os.path.join(os.path.dirname(json_file_path), 'filtered_' + os.path.basename(json_file_path))
    if save and json_file_path is not None:
        if not inplace:
            json_file_path = os.path.join(os.path.dirname(json_file_path), 'filtered_' + os.path.basename(json_file_path))
        with open(json_file_path, 'w') as f:
            json.dump(data_copy, f)

    if keep_annotations and json_file_path is not None:
        for ann in garbage:
            if ann not in data_garbage['annotations']:
                data_garbage['annotations'].append(ann)
        if save:
            with open(os.path.join(os.path.dirname(json_file_path), 'iou_garbage_' + os.path.basename(json_file_path)), 'w') as f:
                json.dump(data_garbage, f)
            print("Garbage annotations have been saved successfully.")

    return data_copy, data_garbage

def inside_iou_filtering(data=None,json_file_path = None,
                         iou_threshold:float=0.1,
                         inplace: bool = False,
                         keep_annotations: bool = False,
                         save: bool = True,data_garbage=None,class_agnostic: bool = True,
                         keep_abbey: bool = False):
    """
    Filter out annotations based on the IoU between their bounding boxes.
    """
    if data is None:
        if json_file_path is None:
            raise ValueError("Either data or json_file_path must be provided.")
        else:
            data = load_coco_json(json_file_path)
    if data_garbage is None:
        data_garbage = data.copy()
        data_garbage['annotations'] = []
    
    id2cat = {cat['id']: cat['name'] for cat in data['categories']}

    # Filter out annotations based on the IoU between their bounding boxes
    filtered_annotations = []
    garbage = []
    for ann1 in data['annotations']:
        iou_flag = False
        for ann2 in data['annotations']:
            if ann1 != ann2:
                if ann1['image_id'] != ann2['image_id']:
                    continue
                if not class_agnostic and ann1['category_id'] != ann2['category_id']:
                    continue
                if is_bbox_inside_another(ann1['bbox'], ann2['bbox'], threshold=iou_threshold):
                    if keep_abbey:
                        if id2cat[ann1['category_id']] in ['abbaye',"prieure"] or id2cat[ann2['category_id']] in ['abbaye','prieure']:
                            continue
                    iou_flag = True
                    break

        if not iou_flag:
            filtered_annotations.append(ann1)
        else:
            garbage.append(ann1)

    data_copy = data.copy()
    data_copy['annotations'] = filtered_annotations

    if save and json_file_path is not None:
        if not inplace :
            json_file_path = os.path.join(os.path.dirname(json_file_path), 'filtered_' + os.path.basename(json_file_path))
        with open(json_file_path, 'w') as f:
            json.dump(data_copy, f)

    if keep_annotations:
        for ann in garbage:
            if ann not in data_garbage['annotations']:
                data_garbage['annotations'].append(ann)
        if save and json_file_path is not None:
            with open(os.path.join(os.path.dirname(json_file_path), 'inside_iou_garbage_' + os.path.basename(json_file_path)), 'w') as f:
                json.dump(data_garbage, f)
            print("Garbage annotations have been saved successfully.")

    return data_copy, data_garbage

def filter_annotations(data=None,json_file_path = None,
                          conf_threshold:float=0.5,
                          height_width_ratio : float = 1.0,
                          second_height_width_ratio : float = 1.0,
                          iou_threshold:float=0.5,
                          inside_threshold:float=0.1,
                          inplace: bool = False,
                          keep_annotations: bool = False,
                          save: bool = True,
                          save_all: bool = False,
                          class_agnostic: bool = False,
                          keep_abbey: bool = False):
    """
    Filter out annotations based on confidence, aspect ratio, and IoU.
    """
    print("Filtering annotations...")
    if data is None:
        if json_file_path is None:
            raise ValueError("Either data or json_file_path must be provided.")
        else:
            data = load_coco_json(json_file_path)

    data_garbage = data.copy()
    data_garbage['annotations'] = []
    
    # Filter out annotations based on confidence
    data, data_garbage = confidence_filtering(data=data, threshold=conf_threshold,
                                               inplace=inplace, keep_annotations=keep_annotations, save=save_all)

    # Filter out annotations based on aspect ratio
    data, data_garbage = aspect_ratio_filtering(data=data, height_width_ratio=height_width_ratio, below=True,
                                                 inplace=inplace, keep_annotations=keep_annotations, save=save_all,data_garbage=data_garbage)
    data, data_garbage = aspect_ratio_filtering(data=data, height_width_ratio=second_height_width_ratio, below=False,
                                                 inplace=inplace, keep_annotations=keep_annotations, save=save_all,data_garbage=data_garbage)
    # # Filter out annotations based on IoU
    data, data_garbage = iou_filtering(data=data, iou_threshold=iou_threshold,
                                        inplace=inplace, keep_annotations=keep_annotations, save=save_all, class_agnostic=class_agnostic,data_garbage=data_garbage)
    
    data, data_garbage = inside_iou_filtering(data=data, iou_threshold=inside_threshold,
                                               inplace=inplace, keep_annotations=keep_annotations, save=save_all,data_garbage=data_garbage,
                                               keep_abbey=keep_abbey)
    if save :
        if not inplace:
            json_file_path = os.path.join(os.path.dirname(json_file_path), 'filtered_' + os.path.basename(json_file_path))
        with open(json_file_path, 'w') as f:
            json.dump(data, f)
        print("Annotations have been filtered successfully.")
        with open(os.path.join(os.path.dirname(json_file_path), 'garbage_' + os.path.basename(json_file_path)), 'w') as f:
            json.dump(data_garbage, f)
        print("Garbage annotations have been saved successfully.")
    return data, data_garbage