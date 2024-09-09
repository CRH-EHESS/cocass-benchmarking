import fiftyone as fo
import fiftyone.utils.random as four
import numpy as np
from pathlib import Path
import cv2
import json
import funcy
from sklearn.model_selection import train_test_split
from typing import Optional
import os
import yaml
import shutil
from pycocotools.coco import COCO
import json

def fo_export_yolo_data(
    samples,
    export_dir,
    classes,
    label_field = "annotations",
    split_ratio : float = None,
    #split = ["train", "val"]
    ):

    ## delete existing tags to start fresh
    samples.untag_samples(samples.distinct("tags"))

    # If there is no split_ratio (or split_ratio == 0), then we are exporting the entire dataset
    if not split_ratio:
        # COde quand split_ratio est None ou 0

        samples.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split="val"
        )
        return
    
    # Else, split the dataset into train and val      
    split_ratio = np.clip(split_ratio, 0, 1) 
    #Code quand split_ratio est un float entre 0 et 1

    four.random_split(
        samples,
        {"train": split_ratio, "val": 1-split_ratio}
    )
    splits = ["train", "val"]
    for split in splits:
        split_view = samples.match_tags(split)

        split_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        classes=classes,
        split=split
    )

def detr_create_data_pairs(input_path, detectron_img_path, detectron_annot_path, dir_type = 'train'):
    """
    Create pairs of image and annotation paths for DETR training from a YOLO formatted dataset.

    Args:
        input_path (str): Path to the dataset.
        detectron_img_path (str): Path to the images in YOLO format. (usually the same as input_path)
        detectron_annot_path (str): Path to the annotations in Detectron format. (usually the same as input_path)
        dir_type (str): Type of the dataset. Default is 'train'.

    Returns:
        list: List of pairs of image and annotation paths.

    """
    img_paths = Path(input_path + dir_type + '/images/').glob('*.jpg')
    pairs = []

    for img_path in img_paths:
        file_name_tmp = str(img_path).split("\\")[-1].split('.')
        file_name_tmp.pop(-1)
        file_name = '.'.join((file_name_tmp))

        label_path = Path(input_path + dir_type + '/labels/' + file_name + '.txt')
        if label_path.is_file():

            line_img = detectron_img_path + dir_type+'/images/'+ file_name + '.jpg'
            line_annot = detectron_annot_path+dir_type+'/labels/' + file_name + '.txt'
            pairs.append([line_img, line_annot])
        
    return pairs

def detr_create_coco_format(data_pairs,verbose=True):

    """
    Create COCO formatted list from pairs of image and annotation paths.

    Args:
        data_pairs (list): List of pairs of image and annotation paths.
        verbose (bool): Print information about the process. Default is True.

    Returns:
        list(dict): List of COCO formatted data. 
            The dictionary contains the following keys:
                - file_name (str): Path to the image.
                - image_id (int): Image ID.
                - height (int): Image height.
                - width (int): Image width.
                - annotations (list(dict)): List of annotations.
                    - bbox (list): Bounding box coordinates.
                    - bbox_mode (BoxMode): Bounding box mode.
                    - category_id (int): Category ID.
                    - iscrowd (int): Is crowd.
    """
    from detectron2.structures import BoxMode

    data_list = []

    for i, path in enumerate(data_pairs):
        
        filename = path[0]

        img_h, img_w = cv2.imread(filename).shape[:2]

        img_item = {}
        img_item['file_name'] = filename
        img_item['image_id'] = i
        img_item['height']= img_h
        img_item['width']= img_w
        if verbose:
            print(f"Added to list {os.path.basename(path[0]).split('.')[0]} with its annotations.")


        annotations = []
        with open(path[1]) as annot_file:
            lines = annot_file.readlines()
            for line in lines:
                if line.strip() == "":
                    continue
                if line[-1]=="\n":
                  box = line[:-1].split(' ')
                else:
                  box = line.split(' ')

                class_id = int(box[0])
                x_c = float(box[1])
                y_c = float(box[2])
                width = float(box[3])
                height = float(box[4])

                x1 = (x_c - (width/2)) * img_w
                y1 = (y_c - (height/2)) * img_h
                x2 = (x_c + (width/2)) * img_w
                y2 = (y_c + (height/2)) * img_h

                annotation = {
                    "bbox": list(map(float,[x1, y1, x2, y2])),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": class_id,
                    "iscrowd": 0
                }
                annotations.append(annotation)
            img_item["annotations"] = annotations
        data_list.append(img_item)
    return data_list 


def save_coco(file, images, annotations, categories):
    """
    Save annotations in COCO format
    """
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)


def filter_annotations(annotations, images):
    """
    Filter annotations based on images
    """
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def det2_splitter(annotations_path : str,
                store_train_path : str,
                store_val_path : str,
                train_split_size : Optional[int] = 0.8):
    
    """
    Split COCO dataset into train and val subsets for detectron 2 model training.
    Creates two new annotations files for the train and val subsets.

    Can also be used to normally split a COCO dataset into train and val subsets.

    Args:
        annotations_path (str): Path to the annotations file.
        store_train_path (str): Path to store the train subset.
        store_val_path (str): Path to store the val subset.
        train_split_size (float): Size of the train subset. Default is 0.8.

    """
    with open(annotations_path, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if True:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=train_split_size)

        save_coco(store_train_path, x, filter_annotations(annotations, x), categories)
        save_coco(store_val_path,  y, filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(len(x), store_train_path, len(y), store_val_path))


def coco2yolo(
        coco_annotation_path : str,
        coco_image_folder_path : str,
        output_dir : str,
        copy_images : bool = True,
        yolo_type : str = "yolov8", # 'yolov5' or 'yolov8' TODO: add 'yolov5
        split : bool = False,
        split_ratio : float = 0.8,
        augment : bool = False
    ):
    """
    Convert COCO dataset to YOLO format

    Args:
        coco_annotation_path (str): Path to the COCO annotations file.
        coco_image_folder_path (str): Path to the COCO images folder.
        output_dir (str): Output directory.
        copy_images (bool): Copy images to the output directory. Default is True.
                            If it is False, only the annotations will be created.
        yolo_type (str): YOLO format type. Default is 'yolov8'. To convert the folder organisation either to YoLoV5 or YoLoV8 format.
        split (bool): Split the dataset into train and val subsets. Default is False.
                    By default, the entire dataset will be converted to YOLO format in one folder of annotations.
        split_ratio (float): Size of the train subset. Default is 0.8.
        augment (bool): Augment the dataset using albumentations. Default is False.
    """
    from pylabel import importer

    os.makedirs(output_dir, exist_ok=True)

    if split==True:
        det2_splitter(coco_annotation_path, 'train_annotations.json', 'val_annotations.json', split_ratio)
        
        for split in ['train', 'val']:
            coco2yolo(
                split+'_annotations.json',
                coco_image_folder_path,
                output_dir+"/"+split+"/",
                copy_images=copy_images,
                yolo_type=yolo_type,
                split=False,
            )
            os.remove(split+'_annotations.json')


        with open(os.path.join(output_dir+"/train/dataset.yaml"), 'r') as file:
            data=yaml.safe_load(file)
        data['train'] = "../train/images"
        data['val'] = "../val/images"
        for split in ['train', 'val']:
            os.remove(os.path.join(output_dir,f"{split}/dataset.yaml"))

        new_yaml_dir = output_dir+"/data.yaml"
        #os.makedirs(new_yaml_dir,exist_ok=True)
        with open(new_yaml_dir, 'w') as file:
            yaml.dump(data, file,default_flow_style=False)



    else:
        dataset = importer.ImportCoco(coco_annotation_path, coco_image_folder_path)
        dataset.export.ExportToYoloV5(output_dir+"/labels/", 
                                      copy_images=copy_images,
                                      cat_id_index=0)
        

def merge_yaml(data_folder,train_name:str="train",valid_name:str="valid",augment:bool=False):
    """
    Merge the data.yaml files in the given folder into one data.yaml file.
    
    Args:
        data_folder (str): Path to the folder containing data.yaml files.
        train_name (str): Name of the train data.yaml file. Default is 'train'.
        valid_name (str): Name of the valid data.yaml file. Default is 'valid'.
    """

    for file in os.listdir(os.path.join(data_folder,train_name)):
        if file.endswith('.yaml'):
            with open(os.path.join(data_folder,train_name, file), 'r') as f:
                data = yaml.safe_load(f)
    for file in os.listdir(os.path.join(data_folder,valid_name)):
        if file.endswith('.yaml'):
            with open(os.path.join(data_folder,valid_name, file), 'r') as f:
                data_val = yaml.safe_load(f)

    if not augment:
            data.update({
            'hsv_h': 0.0,
            'hsv_s': 0.0,   
            'hsv_v': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'bgr': 0.0,
            'mosaic': 0.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'erasing': 0.0,
            'crop_fraction': 0.0,
            })


    data['val'] = data_val['val']
    with open(os.path.join(data_folder, 'data.yaml'), 'w') as f:
        yaml.dump(data, f)
    for file in os.listdir(os.path.join(data_folder,train_name)):
        if file.endswith('.yaml'):
            os.remove(os.path.join(data_folder,train_name, file))
    for file in os.listdir(os.path.join(data_folder,valid_name)):
        if file.endswith('.yaml'):
            os.remove(os.path.join(data_folder,valid_name, file))
    print(f"Merged data.yaml files in {os.path.basename(data_folder)} into one data.yaml file.")

def convert_yolov5_to_yolov8(yolov5_path, yolov8_path):
    """
    Convert YOLOv5 dataset format to YOLOv8 dataset format.
    
    Args:
        yolov5_path (str): Path to the YOLOv5 formatted dataset.
        yolov8_path (str): Path to save the YOLOv8 formatted dataset.
    """
    # Create necessary directories for YOLOv8 format
    os.makedirs(os.path.join(yolov8_path, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(yolov8_path, 'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(yolov8_path, 'valid/images'), exist_ok=True)
    os.makedirs(os.path.join(yolov8_path, 'valid/labels'), exist_ok=True)
    
    # Copy images and labels from YOLOv5 to YOLOv8 format
    for split in ['train', 'val']:
        split_new = 'train' if split == 'train' else 'valid'
        for item in ['images', 'labels']:
            src_dir = os.path.join(yolov5_path, item, split)
            dst_dir = os.path.join(yolov8_path, split_new, item)
            if os.path.exists(src_dir):
                for file_name in os.listdir(src_dir):
                    shutil.copy(os.path.join(src_dir, file_name), os.path.join(dst_dir, file_name))

    # Copy the data.yaml file
    shutil.copy(os.path.join(yolov5_path, 'data.yaml'), os.path.join(yolov8_path, 'data.yaml'))
    print(f"Converted YOLOv5 dataset at {yolov5_path} to YOLOv8 format at {yolov8_path}.")

def convert_yolov8_to_yolov5(yolov8_path, yolov5_path):
    """
    Convert YOLOv8 dataset format to YOLOv5 dataset format.
    
    Args:
        yolov8_path (str): Path to the YOLOv8 formatted dataset.
        yolov5_path (str): Path to save the YOLOv5 formatted dataset.
    """
    # Create necessary directories for YOLOv5 format
    os.makedirs(os.path.join(yolov5_path, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(yolov5_path, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(yolov5_path, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(yolov5_path, 'labels/val'), exist_ok=True)
    
    # Copy images and labels from YOLOv8 to YOLOv5 format
    for split in ['train', 'val']:
        split_new = 'train' if split == 'train' else 'val'
        for item in ['images', 'labels']:
            src_dir = os.path.join(yolov8_path, split, item)
            dst_dir = os.path.join(yolov5_path, item, split_new)
            if os.path.exists(src_dir):
                for file_name in os.listdir(src_dir):
                    shutil.copy(os.path.join(src_dir, file_name), os.path.join(dst_dir, file_name))

    # Copy the data.yaml file
    shutil.copy(os.path.join(yolov8_path, 'data.yaml'), os.path.join(yolov5_path, 'data.yaml'))
    print(f"Converted YOLOv8 dataset at {yolov8_path} to YOLOv5 format at {yolov5_path}.")

def albu_coco_augmentation1(annotations_file : str,
        images_folder : str,
        output_folder : str,
        annotations_name : str = "annotations",
        blur : bool = False,
        blur_limit : int = 15,
        grayscale : bool = False,
        equalize : bool= False,
        dropout : bool = False,
        dropout_percentage : float = 0.1,
        hue_saturation : bool = False,
        hue_shift_limit : int = 10,
        saturation_limit : int = 10,
        brightness : bool = False,
        brightness_limit : float = 0.2,
        contrast_limit : float = 0.2,
        gamma : bool = False,
        gamma_range : tuple = (10, 150),
        augmentation_ratio : float = 0.1,
        verbose : bool = False
        ):
    """
    Augment COCO dataset using albumentations
    The augmentation applied is chosen randomly among the selected techniques.
    Only a given percentage of the dataset will be augmented. Default is 10%.

    Args:
        **Files:**
            annotations_file (str): Path to the annotations file.
            images_folder (str): Path to the images folder.
            output_folder (str): Path to the output folder.
            annotations_name (str): Name of the output annotations file. Default is 'annotations'.
              
        **Augmentation techniques:**
            blur (bool): Apply blur augmentation. Default is False.
            blur_limit (int): How much to blur the image (limit for the random value). Default is 15.
            grayscale (bool): Convert images to grayscale. Default is False.
            equalize (bool): Equalize images. Default is False.
            dropout (bool): Apply dropout augmentation (randomly remove pixels). Default is False.
            dropout_percentage (float): Percentage of pixels to remove. Default is 0.1.
            hue_saturation (bool): Apply hue and saturation augmentation. Default is False.
            hue_shift_limit (int): How much to shift the hue (limit for the random value). Default is 10.
            saturation_limit (int): How much to change the saturation (limit for the random value). Default is 10.
            brightness (bool): Apply brightness augmentation. Default is False.
            brightness_limit (float): How much to change the brightness (limit for the random value). Default is 0.2.
            contrast_limit (float): How much to change the contrast (limit for the random value). Default is 0.2.
            gamma (bool): Apply gamma augmentation. Default is False.
            gamma_range (tuple): Gamma range. Default is (10, 150).
       
        **Augmentation ratio:**
            augmentation_ratio (float): Augmentation ratio. Default is 0.1.
         
        verbose (bool): Print information about the process. Default is False.
    """
    import albumentations as A
    import numpy as np
    import shutil

    os.makedirs(output_folder, exist_ok=True)
    shutil.copytree(images_folder, output_folder+"/images")

    transform_dict={
        "blur": A.Blur(blur_limit),
        "grayscale": A.ToGray(),
        "equalize": A.Equalize(mode="cv", by_channels=True),
        "dropout": A.PixelDropout(dropout_prob=dropout_percentage),
        "hue_saturation": A.HueSaturationValue(hue_shift_limit=hue_shift_limit, sat_shift_limit=saturation_limit),
        "brightness": A.RandomBrightnessContrast(brightness_limit=(-brightness_limit,brightness_limit), contrast_limit=(-contrast_limit,contrast_limit)),
        "gamma": A.RandomGamma(gamma_limit=gamma_range)
    }
    transform_list = []
    for key, value in transform_dict.items():
        if eval(key):
            transform_list.append(value)
    if not transform_list:
        raise ValueError("At least one augmentation technique must be selected")
    
    transform = A.Compose([
        A.OneOf(transform_list, p=1)
        ],p=augmentation_ratio)
    
    with open(annotations_file, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        new_images = []
        aug_images=[]
        new_annotations = []
        new_ids =len(images)*2  #*2 so that if it is from the train dataset, the new ids will be unique from also the val dataset (*2 is a security)
        ann_ids = [ann['id'] for ann in annotations]
        augmentations=0
        for image in images:
            new_images.append(image)

            image_path = images_folder +"/"+image['file_name']
            image_data = cv2.imread(image_path)
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            augmented = transform(image=image_data)["image"]
            if not np.array_equal(augmented,image_data):
                augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                new_ids+=1
                augmentations+=1
                if verbose:
                    print(f"Augmented {image['file_name']} ")

                cv2.imwrite(output_folder+"/images/"+os.path.splitext(image['file_name'])[0]+"_augmented.jpg",augmented)
                img_dict = image.copy()
                img_dict["file_name"]=os.path.splitext(image['file_name'])[0]+"_augmented.jpg"
                img_dict["id"]=new_ids
                new_images.append(img_dict)

                for ann in annotations :
                    if ann['image_id']==image['id']:
                        ann_id=max(ann_ids)*2+1
                        ann_ids.append(ann_id)
                        ann2 = ann.copy()
                        ann2['image_id']=new_ids
                        ann2['id']=ann_id
                        annotations.append(ann2)
        if verbose:
            print(f"\n Augmented {augmentations} images in total.")

    aug_coco = {"images":new_images,"annotations":new_annotations,"categories":categories}
    new_coco = {"images":new_images,"annotations":annotations,"categories":categories}
    with open(output_folder+f"/{annotations_name}.json", 'w') as f:
        json.dump(new_coco, f)
            
def albu_coco_augmentation(annotations_file : str,
        images_folder : str,
        output_folder : str,
        annotations_name : str = "annotations",
        blur : bool = False,
        blur_limit : int = 15,
        grayscale : bool = False,
        equalize : bool= False,
        dropout : bool = False,
        dropout_percentage : float = 0.1,
        hue_saturation : bool = False,
        hue_shift_limit : int = 10,
        saturation_limit : int = 10,
        brightness : bool = False,
        brightness_limit : float = 0.2,
        contrast_limit : float = 0.2,
        gamma : bool = False,
        gamma_range : tuple = (10, 150),
        augmentation_ratio : float = 0.1,
        verbose : bool = False
        ):
    """
    Augment COCO dataset using albumentations
    The augmentation applied is chosen randomly among the selected techniques.
    Only a given percentage of the dataset will be augmented. Default is 10%.

    Args:
        **Files:**
            annotations_file (str): Path to the annotations file.
            images_folder (str): Path to the images folder.
            output_folder (str): Path to the output folder.
            annotations_name (str): Name of the output annotations file. Default is 'annotations'.
              
        **Augmentation techniques:**
            blur (bool): Apply blur augmentation. Default is False.
            blur_limit (int): How much to blur the image (limit for the random value). Default is 15.
            grayscale (bool): Convert images to grayscale. Default is False.
            equalize (bool): Equalize images. Default is False.
            dropout (bool): Apply dropout augmentation (randomly remove pixels). Default is False.
            dropout_percentage (float): Percentage of pixels to remove. Default is 0.1.
            hue_saturation (bool): Apply hue and saturation augmentation. Default is False.
            hue_shift_limit (int): How much to shift the hue (limit for the random value). Default is 10.
            saturation_limit (int): How much to change the saturation (limit for the random value). Default is 10.
            brightness (bool): Apply brightness augmentation. Default is False.
            brightness_limit (float): How much to change the brightness (limit for the random value). Default is 0.2.
            contrast_limit (float): How much to change the contrast (limit for the random value). Default is 0.2.
            gamma (bool): Apply gamma augmentation. Default is False.
            gamma_range (tuple): Gamma range. Default is (10, 150).
       
        **Augmentation ratio:**
            augmentation_ratio (float): Augmentation ratio. Default is 0.1.
         
        verbose (bool): Print information about the process. Default is False.
    """
    import albumentations as A
    import numpy as np
    import shutil

    os.makedirs(output_folder, exist_ok=True)
    shutil.copytree(images_folder, output_folder+"/images")

    transform_dict={
        "blur": A.Blur(blur_limit),
        "grayscale": A.ToGray(),
        "equalize": A.Equalize(mode="cv", by_channels=True),
        "dropout": A.PixelDropout(dropout_prob=dropout_percentage),
        "hue_saturation": A.HueSaturationValue(hue_shift_limit=hue_shift_limit, sat_shift_limit=saturation_limit),
        "brightness": A.RandomBrightnessContrast(brightness_limit=(-brightness_limit,brightness_limit), contrast_limit=(-contrast_limit,contrast_limit)),
        "gamma": A.RandomGamma(gamma_limit=gamma_range)
    }
    transform_list = []
    for key, value in transform_dict.items():
        if eval(key):
            transform_list.append(value)
    if not transform_list:
        raise ValueError("At least one augmentation technique must be selected")
    
    transform = A.Compose([
        A.OneOf(transform_list, p=1)
        ],p=augmentation_ratio)
    
    with open(annotations_file, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    aug_images=[]
    aug_annotations = []
    augmentations=0
    for image in images:
        image_path = images_folder +"/"+image['file_name']
        image_data = cv2.imread(image_path)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        augmented = transform(image=image_data)["image"]
        if not np.array_equal(augmented,image_data):
            augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            augmentations+=1
            if verbose:
                print(f"Augmented {image['file_name']} ")

            cv2.imwrite(output_folder+"/images/"+os.path.splitext(image['file_name'])[0]+"_augmented.jpg",augmented)
            img_dict = image.copy()
            img_dict["file_name"]=os.path.splitext(image['file_name'])[0]+"_augmented.jpg"
            aug_images.append(img_dict)

            for ann in annotations :
                if ann['image_id']==image['id']:
                    ann2 = ann.copy()
                    aug_annotations.append(ann2)
    if verbose:
        print(f"\n Augmented {augmentations} images in total.")
            
    aug_coco = {"images":aug_images,"annotations":aug_annotations,"categories":categories}
    # Save the augmented dataset
    # create a new json file for the augmented dataset

    with open(output_folder+f"/aug_coco.json", 'w') as f:
        json.dump(aug_coco, f)
    merge_coco_json([annotations_file,output_folder+f"/aug_coco.json"], output_folder+f"/{annotations_name}.json")
    os.remove(output_folder+f"/aug_coco.json")

def merge_coco_json(json_files: list[str], output_file: str):
    """
    Merge multiple COCO JSON files into a single file.
    This function updates the image IDs, annotation IDs, and category IDs to avoid conflicts.
    It is useful when you want to merge multiple COCO datasets into a single dataset.

    Args:
        json_files (list): List of paths to the COCO JSON files to merge.
        output_file (str): Path to save the merged COCO JSON file.

    Inspired by Miladfa7 on StackOverflow :
    https://stackoverflow.com/a/78246069
    """

    merged_annotations = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    max_image_id = 0
    max_annotation_id = 0
    category_id_offset = 0
    existing_category_ids = set()
    used_annotation_ids = set()
    image_id_mapping = {}

    for idx, file in enumerate(json_files):
        coco = COCO(file)

        # Update image IDs to avoid conflicts
        for image in coco.dataset['images']:
            max_image_id += 1
            image_id_mapping[image['id']] = max_image_id
            image['id'] = max_image_id
            merged_annotations['images'].append(image)

        # Update annotation IDs to avoid conflicts
        for annotation in coco.dataset['annotations']:
            while max_annotation_id in used_annotation_ids:
                max_annotation_id += 1
            annotation['id'] = max_annotation_id
            annotation['image_id'] = image_id_mapping[annotation['image_id']]
            merged_annotations['annotations'].append(annotation)
            used_annotation_ids.add(max_annotation_id)
            max_annotation_id += 1

        # Update categories and their IDs to avoid conflicts
        for category in coco.dataset['categories']:
            if category['id'] not in existing_category_ids and category['name'] not in funcy.pluck('name', merged_annotations['categories']):
                category['id'] += category_id_offset
                merged_annotations['categories'].append(category)
                existing_category_ids.add(category['id'])

    # Order categories by id
    merged_annotations['categories'] = sorted(merged_annotations['categories'], key=lambda x: x['id'])
    # Order annotations by image id
    merged_annotations['annotations'] = sorted(merged_annotations['annotations'], key=lambda x: x['image_id'])

    # Save merged annotations to output file
    with open(output_file, 'w') as f:
        json.dump(merged_annotations, f)

def make_ids_linear(coco_file: str, output_file: str):
    """
    Make category IDs linear in a COCO annotations file and remap the corresponding IDs in the annotations.

    Args:
        coco_file (str): Path to the input COCO JSON file.
        output_file (str): Path to save the output COCO JSON file with linear IDs.
    """
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Extract all unique category IDs
    category_ids = sorted({category['id'] for category in coco_data['categories']})

    # Create a mapping from old IDs to new linear IDs
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(category_ids, start=1)}

    # Update category IDs in the categories section
    for category in coco_data['categories']:
        category['id'] = id_mapping[category['id']]

    # Update category IDs in the annotations section
    for annotation in coco_data['annotations']:
        if annotation['category_id'] in id_mapping:
            annotation['category_id'] = id_mapping[annotation['category_id']]
    # order categories by id
    coco_data['categories'] = sorted(coco_data['categories'], key=lambda x: x['id'])

    # Save the updated COCO data to the output file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)

def update_categories_and_annotations(json_file_path1, json_file_path2):
    """
    Update categories and annotations in two JSON files based on the categories in the first JSON file.
    """


    # Load the first JSON file
    with open(json_file_path1, 'r') as f:
        data1 = json.load(f)

    # Load the second JSON file
    with open(json_file_path2, 'r') as f:
        data2 = json.load(f)

    # Extract the new categories from the first JSON file
    new_categories = data1['categories']

    # Create a mapping from new category names to new category IDs
    new_category_mapping = {cat['name']: cat['id'] for cat in new_categories}

    # Create a mapping from old category IDs to new category IDs based on the category names
    old_categories = data1['categories']
    old_to_new_category_mapping = {cat['id']: new_category_mapping[cat['name']] for cat in old_categories if cat['name'] in new_category_mapping}

    # Update the categories in the first JSON file
    data1['categories'] = new_categories

    # Update the annotations with the new category IDs in the first JSON file, skipping those not found in the mapping
    updated_annotations = []
    for annotation in data1['annotations']:
        old_category_id = annotation['category_id']
        new_category_id = old_to_new_category_mapping.get(old_category_id)
        if new_category_id is not None:
            annotation['category_id'] = new_category_id
            updated_annotations.append(annotation)

    # Replace the annotations in the first JSON file
    data1['annotations'] = updated_annotations

    # Save the updated first JSON file
    updated_json_file_path1 = json_file_path1.with_name(json_file_path1.stem + '_updated.json')
    with open(updated_json_file_path1, 'w') as f:
        json.dump(data1, f, indent=4)

    # Update the categories in the second JSON file
    data2['categories'] = new_categories

    # Update the annotations with the new category IDs in the second JSON file, skipping those not found in the mapping
    updated_annotations = []
    for annotation in data2['annotations']:
        old_category_id = annotation['category_id']
        new_category_id = old_to_new_category_mapping.get(old_category_id)
        if new_category_id is not None:
            annotation['category_id'] = new_category_id
            updated_annotations.append(annotation)

    # Replace the annotations in the second JSON file
    data2['annotations'] = updated_annotations

    # Save the updated second JSON file
    updated_json_file_path2 = json_file_path2.with_name(json_file_path2.stem + '_updated.json')
    with open(updated_json_file_path2, 'w') as f:
        json.dump(data2, f, indent=4)

    print("Categories and annotations updated successfully.")
    return updated_json_file_path1, updated_json_file_path2

def iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

def nms_coco_on_area(json_file, iou_threshold=0.5,output_path="filtered_json", inplace=False):
    """Perform NMS on a JSON file in COCO format. But based on area rather than score because the score is not always present."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    count = len(annotations)
    
    # Sort annotations by area in descending order
    annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
    
    keep = []
    while annotations:
        best = annotations.pop(0)
        keep.append(best)
        
        annotations = [
            ann for ann in annotations
            if ann['category_id'] != best['category_id'] or iou(best['bbox'], ann['bbox']) < iou_threshold
        ]
    
    data['annotations'] = keep
    count = count - len(keep)
    if inplace:
        output_path = json_file
    # Save the filtered annotations back to a JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
