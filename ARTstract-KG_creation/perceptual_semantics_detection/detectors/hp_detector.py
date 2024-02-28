"""
Perceptual Semantics - Human Presence Detection

This script provides functionalities to automatically detect the presence of humans in a collection of images. It uses a fine-tuned logistic regression classifier, built on top of the 'ViT-B/32' variant
of the CLIP (Contrastive Language-Image Pretraining) model. The classifier is reused from the work of Adham Elarabawy,
who specifically fine-tuned it for human presence detection in fashion-domain images. The script processes images in a
specified folder, and for each image, it predicts whether humans are present or not. The results are stored in a JSON
file with the specified output name.

Author: Delfina Sol Martinez Pandiani
Date: 20 Jul 2023
"""

import os
import json
import clip
import torch
import pickle
from PIL import Image
from huggingface_hub import hf_hub_download


def process_folder(annotation_situation, folder_path, output_file):
    """
    Process a folder containing cultural images and detect human presence.

    This function iterates over each image in the specified folder and calls the 'detect_human_presence' function to
    predict whether humans are present in the image. The results are then stored in a dictionary with the image ID as
    the key, and the human presence prediction as the value. The final results are saved in a JSON file with the
    specified output name.

    Parameters:
        folder_path (str): The path to the folder containing cultural images.
        output_file (str): The name of the output JSON file.

    Returns:
        None
    """
    annotator = annotation_situation["annotator"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device)

    def detect_human_presence(img):
        """
        Detect the presence of humans in an input image.

        This function utilizes a pre-trained logistic regression classifier, fine-tuned for human presence detection in
        fashion-domain images. The classifier is built on top of the 'ViT-B/32' variant of the CLIP model.

        Parameters:
            img (PIL.Image.Image): The input image to analyze for human presence.

        Returns:
            bool: A boolean value indicating whether humans are present ('True') or not ('False') in the image.
        """
        model_path = hf_hub_download(repo_id=annotator, filename="model.pkl")

        with open(model_path, 'rb') as file:
            human_classifier = pickle.load(file)

        features = clip_model.encode_image(clip_preprocess(img).unsqueeze(0).to(device)).detach().cpu().numpy()
        confidence_humans_present = human_classifier.predict_proba(features)[0][1]
        pred = human_classifier.predict(features)  # True = has human, False = no human

        # pred = confidence_humans_present >= 0.5  # True if confidence is greater or equal to 0.5, False otherwise

        print(f"Prediction (has_human): {pred}")
        return pred, confidence_humans_present

    folder_hp = {}
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
            human_presence_pred, confidence_humans_present = detect_human_presence(image)
            human_presence_pred = human_presence_pred.tolist()
            human_presence = human_presence_pred[0]
            if human_presence == True:
                lexical_unit = "human depicted"
                conceptnet_concept = "conceptnet:human"
            else:
                lexical_unit = "human not depicted"
                conceptnet_concept = "conceptnet:nonhuman"

            folder_hp[image_id] = {}
            folder_hp[image_id][annotation_type] = {}
            folder_hp[image_id][annotation_type][annotation_situation_name] = {
                "human_presence": str(human_presence).title(),  # Capitalize the human presence string
                "lexical_presence": lexical_unit.capitalize(),
                "conceptnet_concept": conceptnet_concept,
                "annotation_strength": confidence_humans_present
            }


    with open(output_file, "w") as file:
        json.dump(folder_hp, file, indent=2)

    print("Processing completed. JSON file generated:", output_file)
    return


# annotation_situation = {
#     "annotation_type" : "hp",
#     "annotator" : "adhamelarabawy/fashion_human_classifier",
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

