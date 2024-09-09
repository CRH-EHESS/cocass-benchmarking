# src/dataset/
from PIL import Image
import torch
import torchvision
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from transformers import DetrImageProcessor
import json
import os
import shutil

class CocoDetection(torchvision.datasets.CocoDetection):
    """
    Custom dataset class for COCO detection.

    Args:
        image_directory_path (str): Path to the directory containing the images.
        annotation_file_path (str): Path to the COCO annotation file.
        image_processor: Image processor object used for preprocessing the images.
        train (bool, optional): Whether the dataset is for training or not. Defaults to True.
    """

    def __init__(
        self, 
        image_directory_path: str, 
        annotation_file_path: str,
        image_processor, 
        train: bool = True
    ):
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        """
        Retrieves the preprocessed image and target label at the given index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: Preprocessed image tensor and target label.
        """
        images, annotations = super(CocoDetection, self).__getitem__(idx)        
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target


# Helper function to create temporary JSON annotation files
def create_subset_annotations(original_annotations, subset_ids, subset_file):
    """
    Create a subset of annotations based on the given subset_ids and save it to the subset_file.

    Args:
        original_annotations (dict): The original annotations dictionary.
        subset_ids (list): The list of image IDs to include in the subset.
        subset_file (str): The file path to save the subset annotations.

    Returns:
        None
    """
    import json

    subset_annotations = {
        'images': [img for img in original_annotations['images'] if img['id'] in subset_ids],
        'annotations': [ann for ann in original_annotations['annotations'] if ann['image_id'] in subset_ids],
        'categories': original_annotations['categories']
    }
    with open(subset_file, 'w') as f:
        json.dump(subset_annotations, f)

def create_coco_pth_datasets(annotations_file : str,
                             images_folder : str,
                             image_processor : str = "facebook/detr-resnet-152",
                             test_size : float = 0.2,
                             splitted : bool = False,
                             save=False,
                             split_only=False,
                             train_ann_name : str = 'train_annotations.json',
                             val_ann_name : str = 'val_annotations.json'):
    """
    Create COCO PyTorch datasets for training and validation.

    Parameters:
    - annotations_file (str): Path to the COCO annotations file.
    - images_folder (str): Path to the folder containing the images.
    - image_processor (str): Name of the image processor model to use. Default is "facebook/detr-resnet-152".
    - test_size (float): Proportion of the dataset to include in the validation set. Default is 0.2.
    - save (bool): Whether to save the temporary JSON files. Default is False.
    - split_only (bool): Whether to only perform the dataset split and return early. Default is False.
    - train_ann_name (str): Name of the train annotations file. Default is 'train_annotations.json'.
    - val_ann_name (str): Name of the validation annotations file. Default is 'val_annotations.json'.

    Returns:
    - train_dataset (CocoDetection): COCO dataset object for training.
    - val_dataset (CocoDetection): COCO dataset object for validation.
    """
    
    from pathlib import Path
    import os
    
    with open(annotations_file) as f:
        annotations = json.load(f)

    # Get image IDs
    image_ids = [img['id'] for img in annotations['images']]
    if not splitted:
        # Step 2: Split the dataset
        train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

        # Create temporary JSON files for train and val subsets
        train_annotations_file = os.path.join(Path(annotations_file).parent,train_ann_name)
        val_annotations_file = os.path.join(Path(annotations_file).parent,val_ann_name)
        create_subset_annotations(annotations, train_ids, train_annotations_file)
        create_subset_annotations(annotations, val_ids, val_annotations_file)
        if split_only:
            return 
    else:
        train_annotations_file = os.path.join(Path(annotations_file).parent,train_ann_name)
        val_annotations_file = os.path.join(Path(annotations_file).parent,val_ann_name)


    # Step 3: Create PyTorch datasets
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")#,size={"shortest_edge": 800, "longest_edge": 1333})


    train_dataset = CocoDetection(image_directory_path=images_folder, annotation_file_path=train_annotations_file, image_processor=processor)
    val_dataset = CocoDetection(image_directory_path=images_folder, annotation_file_path=val_annotations_file, image_processor=processor)

    id2id = {original_key: i for i, original_key in enumerate(train_dataset.coco.cats.keys())}

    train_dataset.coco.cats = {new_key: {**value, 'id': new_key} for new_key, (_, value) in enumerate(train_dataset.coco.cats.items())}
    val_dataset.coco.cats = {new_key: {**value, 'id': new_key} for new_key, (_, value) in enumerate(val_dataset.coco.cats.items())}

    for annot in train_dataset.coco.dataset['annotations']:
        annot['category_id'] = id2id[annot['category_id']]
        annot['area']=annot['bbox'][2]*annot['bbox'][3]

    for annot in val_dataset.coco.dataset['annotations']:
        annot['category_id'] = id2id[annot['category_id']]
        annot['area']=annot['bbox'][2]*annot['bbox'][3]

    if not save and not splitted:
        # Erase temporary JSON files
        os.remove(train_annotations_file)
        os.remove(val_annotations_file)
    elif save and not splitted:
        shutil.move(train_annotations_file, '../'+images_folder)
        shutil.move(val_annotations_file, '../'+images_folder)

    return train_dataset, val_dataset