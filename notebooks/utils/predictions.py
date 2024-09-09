from sahi.predict import get_prediction, get_sliced_prediction
import fiftyone as fo
import os
import cv2
from PIL import Image
import numpy as np


def fo_predict_simple(detection_model, sample: fo.Sample, label_field: str, **kwargs):
    """
    Perform object detection using a detection model from SAHI and add the detections to the sample.

    Args:
        detection_model (sahi.AutoDetectionModel): The object detection model to use for prediction.
        sample (fo.Sample): The sample containing the image to predict on.
        label_field (str): The name of the field to store the detections in the sample.
        **kwargs: Additional keyword arguments to pass to the get_prediction function.

    Returns:
        result (sahi.PredictionResult): The result of the object detection.
    """
    result = get_prediction(
        sample.filepath,    # Path to the image to predict on
        detection_model,    # Detection model
    )

    # Transform the detections into fiftyone format and add them to the sample
    sample[label_field] = fo.Detections(detections=result.to_fiftyone_detections())
    return result

def fo_predict_with_slicing(detection_model, sample: fo.Sample, label_field: str, **kwargs):
    """
    Perform object detection using a detection model from SAHI with slicing and add the detections to the sample.

    Args:
        detection_model (sahi.AutoDetectionModel): The object detection model to use for prediction.
        sample (fo.Sample): The sample containing the image to predict on.
        label_field (str): The name of the field to store the detections in the sample.
        **kwargs: Additional keyword arguments to pass to the get_sliced_prediction function.

    Returns:
        result (sahi.SlicedPredictionResult): The result of the object detection with slicing.
    """
    result = get_sliced_prediction(
        sample.filepath,    # Path to the image to predict on
        detection_model,    # Detection model
        verbose=0,
        **kwargs
    )

    # Transform the detections into fiftyone format and add them to the sample
    sample[label_field] = fo.Detections(detections=result.to_fiftyone_detections())
    return result

def fo_extract_bboxes(dataset: fo.Sample,output_directory: str, field : str = "detections", buffer : int = 0):  # Obsolet : Dans la volonté d'extraire les bbox et images par images 
    #Nous avons abandonné cette piste

    """
    Extract images from the sample auroud the bounding boxes and save them to disk.
    We advise around 10~50 pixels buffer for a good extraction.

    Args:
        sample (fo.Sample): The sample containing the image and detections.
        output_directory (str): The directory to save the extracted images to.
        field (str): The name of the field containing the detections in the sample.
        buffer (float): The buffer to add around the bounding boxes.
    """
    os.makedirs(output_directory, exist_ok=True)

    for sample in dataset:
        img_name=os.path.basename(sample.filepath).split(".")[0]

        if sample[field].detections :
            det= sample[field].detections
        else:
            continue
        

        bboxes=[]

        for d in det:
            bboxes.append(d.bounding_box)

        # Load the image
        image = sample['filepath']
        image =Image.open(image)
        image_array = np.array(image)

        # Extract the shape of the image
        height,width = image_array.shape[0],image_array.shape[1]

        for i, bbox in enumerate(bboxes):
            #print(bbox)
            # Extract the bounding box from the image
            
            x, y, w, h = bbox
            if x<0:
                x=0
                w=w+x
            if y<0:
                y=0

            # Get the coordinates of the bounding box in the image
            x1, y1, x2, y2 = int(min(x*width,max(0,x*width-buffer))), int(min(y*height,max(0,y*height-buffer))), int(max((x + w)*width,min(width,(x*width)+buffer))), int(max((y + h)*height,min(height,(y+h)*height+buffer)))
            sub_image = image_array[y1:y2, x1:x2]
            img=Image.fromarray(sub_image)

            # Save the sub-image to disk
            img.save(f"{output_directory}/{img_name}_bbox_{i}.jpg")