"""
Perceptual Semantics - Object Detection

This script provides functionalities to automatically detect objects in a collection of cultural images. It utilizes a pre-trained object detection model based on the 'DetrForObjectDetection' variant of the DETR (Detection Transformer) architecture to detect various objects in the images. The detected objects are then mapped to their corresponding ConceptNet concepts using the 'get_conceptnet_concept' function.

The primary objective of this script is to enrich cultural image datasets by adding information about the presence of various objects and their associated ConceptNet concepts. This information can be valuable for exploring relationships between abstract concepts and object occurrences in cultural imagery, leading to insights into the depiction of humans and other objects in different cultural contexts.

The script performs the following steps:

1. Object Detection: The 'detect_objects' function takes an input image and utilizes the pre-trained object detection model to detect various objects with corresponding confidence scores. The function returns the input image with bounding boxes around the detected objects and a list of dictionaries containing information about each detected object, including its name, bounding box coordinates, and confidence score.

2. Concept Mapping: For each detected object, the script predicts its corresponding ConceptNet concept using the 'get_conceptnet_concept' function. The ConceptNet concepts provide a semantically enriched representation of the detected objects, enabling researchers to explore connections between objects and abstract concepts in cultural images.

3. Folder Processing: The 'get_folder_objects_json' function processes a folder containing cultural images. It iterates over each image, calls the 'detect_objects' function to detect objects and predict their ConceptNet concepts, and stores the results in two JSON files: one with detailed object information and another with short object information.

Author: Delfina Sol Martinez Pandiani
Date: 20 Jul 2023
"""

import os
import sys
import json
import torch
from PIL import Image
from collections import defaultdict
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import YolosImageProcessor, YolosForObjectDetection, PretrainedConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .conceptnet_matching.conceptnet_mapping import get_conceptnet_concept


def process_folder(annotation_situation, folder_path, output_file):
    """
    Process a folder containing cultural images, detect objects, and predict ConceptNet concepts.

    This function iterates over each image in the specified folder, calls the 'detect_objects' function to detect
    objects and predict their ConceptNet concepts, and stores the results in two JSON files: one with detailed
    object information and another with short object information.

    Parameters:
        folder_path (str): The path to the folder containing cultural images.
        output_file (str): The name of the output JSON file with detailed object information.
        output_short_file (str): The name of the output JSON file with short object information.
        processor (DetrImageProcessor): The object detection processor for the model.
        model (DetrForObjectDetection): The pre-trained object detection model.
        threshold (float): The confidence threshold for object detection.

    Returns:
        str: The path to the output_file with detailed object information.
    """

    annotator = annotation_situation["annotator"]
    detection_threshold = float(annotation_situation["detection_threshold"])

    def detect_objects(image, annotator):
        """
        Detect objects in the input image and predict their ConceptNet concepts.

        This function takes an input image and utilizes a pre-trained object detection model based on the 'DetrForObjectDetection'
        variant of the DETR (Detection Transformer) architecture. The model is fine-tuned to detect various objects in the images.
        For each detected object, the function predicts its ConceptNet concept using the 'get_conceptnet_concept' function.
        The results are returned as a list of dictionaries, each containing information about the detected object, including its
        ConceptNet concept, probability score, and bounding box coordinates.

        Parameters:
            image (PIL.Image.Image): The input image to detect objects.

        Returns:
            PIL.Image.Image: The input image with detected objects.
            list: A list of dictionaries containing information about each detected object. Each dictionary contains the
                  following keys:
                  - 'object_name': The name of the detected object class.
                  - 'coordinates': The bounding box coordinates (x1, y1, x2, y2) of the detected object.
                  - 'score': The probability score of the detected object's class.
        """

        if "detr" in annotator:
            processor = DetrImageProcessor.from_pretrained(annotator)
            model = DetrForObjectDetection.from_pretrained(annotator)

            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # convert outputs (bounding boxes and class logits) to COCO API
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=detection_threshold)[0]

            object_details = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                object_name = model.config.id2label[label.item()]
                coordinates = {
                    "x1": round(box[0].item(), 2),
                    "y1": round(box[1].item(), 2),
                    "x2": round(box[2].item(), 2),
                    "y2": round(box[3].item(), 2)
                }
                object_info = {
                    "object_name": object_name,
                    "coordinates": coordinates,
                    "score": round(score.item(), 3)
                }
                object_details.append(object_info)
            print(object_details)
            for object_info in object_details:
                print(
                    f"Detected {object_info['object_name']} with confidence {object_info['score']} at location {object_info['coordinates']}")
        elif "yolo" in annotator:
            if "household" in annotator:
                # The URL to the model's configuration file
                config_file_url = "https://huggingface.co/degirum/yolo_v5s_household_objects_512x512_quant_n2x_cpu_1/raw/main/yolo_v5s_household_objects_512x512_quant_n2x_cpu_1.json"
                # Load the model configuration from the URL
                config = PretrainedConfig.from_pretrained(config_file_url)
                # Initialize the model using the loaded configuration
                model = YolosForObjectDetection(config)
                # Load the image processor's configuration from the URL
                image_processor_config = PretrainedConfig.from_pretrained(config_file_url)
                # Initialize the image processor using the loaded configuration
                image_processor = YolosImageProcessor(image_processor_config)

            else:
                model = YolosForObjectDetection.from_pretrained(annotator)
                image_processor = YolosImageProcessor.from_pretrained(annotator)

            inputs = image_processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # model predicts bounding boxes and corresponding COCO classes
            logits = outputs.logits
            bboxes = outputs.pred_boxes

            # print results
            target_sizes = torch.tensor([image.size[::-1]])
            results = image_processor.post_process_object_detection(outputs, threshold=detection_threshold, target_sizes=target_sizes)[
                0]

            object_details = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                object_name = model.config.id2label[label.item()]
                coordinates = {
                    "x1": round(box[0].item(), 2),
                    "y1": round(box[1].item(), 2),
                    "x2": round(box[2].item(), 2),
                    "y2": round(box[3].item(), 2)
                }
                object_info = {
                    "object_name": object_name,
                    "coordinates": coordinates,
                    "score": round(score.item(), 3)
                }
                object_details.append(object_info)
            for object_info in object_details:
                print(
                    f"Detected {object_info['object_name']} with confidence {object_info['score']} at location {object_info['coordinates']}")

        return image, object_details

    def get_od_short(object_details):
        ## EXTRA: To get the short output
        detected_objects_short = []
        # Create dictionaries for storing unique conceptnet concepts and their corresponding maximum probabilities
        conceptnet_scores = defaultdict(float)
        conceptnet_scores_short = defaultdict(float)
        for obj_info in object_details:
            conceptnet_concept = get_conceptnet_concept(object_name)
            probability_score = obj_info["score"]
            # Update the maximum probability for the conceptnet concept if it is higher than the existing value
            if probability_score > conceptnet_scores[conceptnet_concept]:
                conceptnet_scores[conceptnet_concept] = probability_score

        # Create the object dictionary for each unique conceptnet concept
        for conceptnet_concept, probability_score in conceptnet_scores.items():
            # Add the object dictionary to the short detected_objects list
            object_dict = {
                "conceptnet_concept": conceptnet_concept,
                "annotation_strength": probability_score
            }
            detected_objects_short.append(object_dict)

        # Add the detailed detected_objects list to the folder_objects dictionary
        folder_objects_short[image_id] = {}
        folder_objects_short[image_id][annotation_type] = {}
        folder_objects_short[image_id][annotation_type][annotation_situation_name] = {
            "detected_objects": detected_objects_short
        }

        # Write the folder_objects_short dictionary to the output_short_file
        with open(output_short_file, "w") as file:
            json.dump(folder_objects_short, file, indent=2)

        return

    folder_objects = {}
    annotation_situation_name = str(
        annotation_situation["annotated_dataset"] + "_" + annotation_situation["annotation_type"] + "_" +
        annotation_situation["annotation_time"])
    annotation_type = annotation_situation["annotation_type"]

    # EXTRA: for short output
    folder_objects_short = {}
    output_short_file = f'../{annotation_situation["annotation_type"]}_output_short.json'

    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image_id = os.path.splitext(filename)[0]
            image = Image.open(image_path)
            image = image.convert("RGB")
            image, object_details = detect_objects(image, annotator)


            # Create the detected_objects dictionary
            detected_objects = []

            # Process each object detail
            for obj_info in object_details:
                object_name = obj_info["object_name"]
                # conceptnet_concept = get_conceptnet_concept(object_name)
                probability_score = obj_info["score"]
                coordinates = obj_info["coordinates"]

                # Add the object dictionary to the detailed detected_objects list
                object_dict = {
                    "detected_object": object_name,
                    # "conceptnet_concept": conceptnet_concept,
                    "coordinates": coordinates,
                    "annotation_strength": probability_score
                }
                detected_objects.append(object_dict)

            # Add the detailed detected_objects list to the folder_objects dictionary
            folder_objects[image_id] = {}
            folder_objects[image_id][annotation_type] = {}
            folder_objects[image_id][annotation_type][annotation_situation_name] = {
                "detected_objects": detected_objects
            }

            # EXTRA: get short od output
            # get_od_short(object_details)

    # Write the folder_objects dictionary to the output_file
    with open(output_file, "w") as file:
        json.dump(folder_objects, file, indent=2)




# annotation_situation = {
#     "annotation_type" : "od",
#     # "annotator" : "Shmoel/yolo-retail-product-recognion",
#     # "annotator" : "degirum/yolo_v5s_household_objects_512x512_quant_n2x_cpu_1",
#     # "annotator" : "hustvl/yolos-tiny",
#     # "annotator" : "valentinafeve/yolos-fashionpedia",
#     "annotator" : "facebook/detr-resnet-101",
#     # "annotator" : "jcm-art/hf_object_detection_DETR_CPPE_5_pipeline",
#     "annotation_place" : "Italy",
#     "annotation_time" : "2023_06_28",
#     "detection_threshold": "0.7",
#     "annotated_dataset": "ARTstract"
# }
#
# folder_path = '../../__prova/test'
#
# output_file = f'../{annotation_situation["annotation_type"]}_output.json'
# output_short_file = f'../{annotation_situation["annotation_type"]}_output_short.json'
#
# process_folder(annotation_situation, folder_path, output_file)
#


