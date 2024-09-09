import random
import cv2
import numpy as np
import supervision as sv
import torchvision
import os
import torch

def pth_dataset_visualize(pth_dataset : torchvision.datasets.CocoDetection,
                          num_images : int = 1,
                          img_size : int = 10,
                          val : bool = False,
                          plot_over : bool = True,
                          model = None,
                          processor = None,
                          CONFIDENCE_TRESHOLD = 0.5,
                          ):
    """
    Function to visualize images from a PyTorch dataset.

    Parameters:
    - pth_dataset (torchvision.datasets.CocoDetection): The PyTorch dataset object.
    - num_images (int): The number of images to visualize. Default is 1.
    - img_size (int): The size of the displayed image. Default is 10.
    - val (bool): Whether to perform validation and show predictions. Default is False.
    - plot_over (bool): Whether to plot predictions over the ground truth annotations. Default is True.
    - model: The object detection model. Required if val is True.
    - processor: The object detection processor. Required if val is True.
    - CONFIDENCE_TRESHOLD (float): The confidence threshold for object detection predictions. Default is 0.5.
    """
    for i in range(num_images):
        # select random image
        image_ids = pth_dataset.coco.getImgIds()
        image_id = random.choice(image_ids)
        print('Image #{}'.format(image_id))

        # load image and annotations 
        image = pth_dataset.coco.loadImgs(image_id)[0]
        annotations = pth_dataset.coco.imgToAnns[image_id]
        image_path = os.path.join(pth_dataset.root, image['file_name'])
        image = cv2.imread(image_path)

        # annotate
        detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)

        # we will use id2label function for training
        categories = pth_dataset.coco.cats
        id2label = {v['id']: v['name'] for k,v in categories.items()}

        labels = [
            f"{id2label[class_id]}" 
            for _, _, class_id, _ 
            in detections
        ]

        box_annotator = sv.BoxAnnotator(thickness=2, text_padding=5)
        frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        
        if val:
            DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                # Load image and predict
                inputs = processor(images=[image], return_tensors='pt').to(DEVICE)
                outputs = model(**inputs)

                # Post-process
                target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
                results = processor.post_process_object_detection(
                    outputs=outputs, 
                    threshold=CONFIDENCE_TRESHOLD, 
                    target_sizes=target_sizes
                )[0]

            predictions = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.5)
            labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in predictions]

            if plot_over:
                frame = box_annotator.annotate(scene=frame, detections=predictions, labels=labels)
                
            else:
                frame_copy = frame.copy()
                pred_frame = box_annotator.annotate(scene=frame_copy, detections=predictions, labels=labels)
                # Combine gt_frame and pred_frame side by side
                # create a blank white with padding
                padding = np.ones((frame.shape[0], 10, 3), dtype=np.uint8) * 255
                combined_frame = np.hstack((frame, padding, pred_frame))
                print('Ground Truth and Predictions')
                # Display combined_frame
                sv.show_frame_in_notebook(combined_frame, (img_size * 2 + 1, img_size))  # Adjust display size as needed
                continue
            print('Ground Truth and Predictions')
        else:
            print('Annotations')

        sv.show_frame_in_notebook(frame, (img_size, img_size))

def det2_dataset_visualize(
        dataset_name : str,
        num_images : int = 1,
        img_size : int = 10,):
    """
    Function to visualize images from a Detectron2 dataset.

    Args:
        dataset_name (str): The name of the dataset.
        num_images (int, optional): The number of images to visualize. Defaults to 1.
        img_size (int, optional): The size of the images. Defaults to 10.

    Returns:
        None
    """
    
    import matplotlib.pyplot as plt
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog, DatasetCatalog

    # Get metadata and dataset
    dataset_metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    # Visualize some dataset samples
    for d in random.sample(dataset_dicts, num_images):
        img = cv2.imread(d["file_name"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualizer = Visualizer(img_rgb[:, :, ::-1], metadata=dataset_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        
        # Convert image from BGR to RGB for matplotlib
        result_image = vis.get_image()[:, :, ::-1]
        
        # Display the image using matplotlib
        plt.figure(figsize=(img_size, img_size))
        plt.imshow(result_image)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()

def visualize_image(image_name : str, annotations_json_path:str):
    """
    Function to visualize an image with annotations.

    Args:
        image_name (str): The name of the image file.
        annotations_json_path (str): The path to the annotations JSON file.
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import json

    # Load image
    image = Image.open(image_name)
    
    # Load annotations
    with open(annotations_json_path) as f:
        data = json.load(f)
    # Visualize image with annotations
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')  # Turn off axis numbers and ticks
    categories=data['categories']
    for image in data['images']:
        if image['file_name'] == os.path.basename(image_name):
            for annotation in data['annotations']:
                if annotation['image_id'] == image['id']:
                    print("oui")
                    bbox = annotation['bbox']
                    for category in categories:
                        if category['id'] == annotation['category_id']:
                            label = category['name']
                    plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='red', linewidth=2))
                    plt.text(bbox[0], bbox[1], label, fontsize=12, color='red', weight='bold', backgroundcolor='white')

    plt.show()