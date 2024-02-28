"""
Perceptual Semantics - Art Style Detection

This script provides functionalities to automatically detect the art style of images using the Vision Transformer (ViT)
model. It leverages a pre-trained ViT model for image classification and maps the detected art styles to their
corresponding ConceptNet concepts using the ConceptNet Mapping module.

The script contains the following functions:

1. detect_style(image):
    Detects the art style of the input image using a pre-trained Vision Transformer model. The function returns the
    index of the detected art style.

2. get_image_art_style(image_path):
    Takes the path to an image file as input and returns the detected art style as a human-readable string.

3. process_folder(folder_path, output_file):
    Processes a folder containing multiple images. For each image, it uses `get_image_art_style` to detect the art
    style and `get_conceptnet_concept` to map it to the corresponding ConceptNet concept. The results are stored in
    a dictionary and saved in a JSON file with the specified `output_file` name.

Note: The script automatically detects the art style of images and enriches their representation with semantic
information from ConceptNet.

Author: Delfina Sol Martinez Pandiani
Date: 20 Jul 2023
"""

import os
import sys
import json
import torch
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .conceptnet_matching.conceptnet_mapping import get_conceptnet_concept
from transformers import ViTFeatureExtractor, ViTForImageClassification

def process_folder(annotation_situation, folder_path, output_file):
    """
    Process a folder containing multiple images, detect the art style of each image, and save the results in a JSON file.

    Parameters:
        folder_path (str): The path to the folder containing image files.
        output_file (str): The name of the output JSON file.

    Returns:
        None
    """
    annotator = annotation_situation["annotator"]

    # art_style_mapping: A dictionary that maps the detected style index to human-readable art style names.
    art_style_mapping = {
        0: "Art Nouveau",
        1: "Baroque",
        2: "Expressionism",
        3: "Impressionism",
        4: "Post-Impressionism",
        5: "Realism",
        6: "Renaissance",
        7: "Romanticism",
        8: "Surrealism",
        9: "Ukiyo-e"
    }

    def detect_style(image):
        """
        Detect the art style of the input image using a pre-trained Vision Transformer (ViT) model.

        Parameters:
            image (PIL.Image): The input image to be classified.

        Returns:
            int: The index of the detected art style in the `art_style_mapping` dictionary.

        Note:
            The function uses the 'oschamp/vit-artworkclassifier' pre-trained ViT model for image classification. The
            'art_style_mapping' dictionary maps the detected style index to human-readable art style names.
        """

        vit = ViTForImageClassification.from_pretrained(annotator)
        vit.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vit.to(device)

        model_name_or_path = 'google/vit-base-patch16-224-in21k'
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

        # Load Image
        encoding = feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoding['pixel_values'].to(device)

        with torch.no_grad():
            outputs = vit(pixel_values)
            logits = outputs.logits

        proba = logits.softmax(-1)
        prediction = logits.argmax(-1)
        art_style_name = art_style_mapping.get(prediction.item(), "Unknown")
        pred_prob = proba[0, prediction.item()].item()  # Get the probability of the top prediction
        return art_style_name, pred_prob

    folder_art_styles = {}
    annotation_situation_name = str(
        annotation_situation["annotated_dataset"] + "_" + annotation_situation["annotation_type"] + "_" +
        annotation_situation["annotation_time"])
    annotation_type = annotation_situation["annotation_type"]

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image_id = os.path.splitext(filename)[0]
            image = Image.open(image_path)
            image = image.convert("RGB")
            art_style_name, class_probs = detect_style(image)
            conceptnet_concept = get_conceptnet_concept(art_style_name)

            folder_art_styles[image_id] = {}
            folder_art_styles[image_id][annotation_type] = {}
            folder_art_styles[image_id][annotation_type][annotation_situation_name] = {
                "art_style": art_style_name,
                "conceptnet_concept": conceptnet_concept,
                "annotation_strength": class_probs
            }

    with open(output_file, "w") as file:
        json.dump(folder_art_styles, file, indent=2)

    print("Processing completed. JSON file generated:", output_file)

#
#
# annotation_situation = {
#     "annotation_type" : "as",
#     "annotator" : "oschamp/vit-artworkclassifier",
#     "annotation_place" : "Italy",
#     "annotation_time" : "2023_06_28",
#     "detection_threshold": "top one",
#     "annotated_dataset": "ARTstract"
# }
#
# folder_path = '../../__prova/test'
#
# output_file = f'../{annotation_situation["annotation_type"]}_output.json'
#
# process_folder(annotation_situation, folder_path, output_file)
#
#
