import os
import json
import sahi
from sahi.utils.coco import Coco,CocoImage,CocoAnnotation,CocoPrediction
from PIL import Image
from sahi.utils.file import save_json

def json_from_result_as_annotation(image_path,result:sahi.prediction.PredictionResult,pred=True):

    
    with Image.open(image_path) as img:
        width, height = img.size

    coco=Coco()
    coco_image=CocoImage(os.path.basename(image_path),height,width)

    for object_pred in result.object_prediction_list:
        if pred:
            coco_prediction=CocoPrediction(bbox=object_pred.bbox.to_xywh(),
                                           category_id=object_pred.category.id,
                                           category_name=object_pred.category.name,
                                           score=object_pred.score.value)
        else:
            coco_prediction=CocoAnnotation(bbox=object_pred.bbox.to_xywh(),
                                           category_id=object_pred.category.id,
                                           category_name=object_pred.category.name)
        coco_image.add_annotation(coco_prediction)

    coco.add_image(coco_image)

    coco_json=coco.json
    return coco_json