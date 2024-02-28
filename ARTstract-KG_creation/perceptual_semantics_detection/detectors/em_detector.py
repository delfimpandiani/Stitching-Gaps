"""
Perceptual Semantics - Art Emotion Detection

This script utilizes the Artemis Art Emotion Detector model to analyze images and automatically detect the emotions depicted in them. The model is based on a pre-trained checkpoint, and emotions are classified into distinct categories using the provided art_emotion_mapping dictionary.

Artemis is a specialized art image dataset, the authors of the dataset have also trained a detection model tailored for analyzing art and visual content. The script employs the best pre-trained model available from Artemis to perform the emotion detection.
"""

import os
import sys
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .conceptnet_matching.conceptnet_mapping import get_conceptnet_concept


def process_folder(annotation_situation, folder_path, output_file):
    """
    Processes a folder containing multiple images, detects the art emotion for each image, and generates a JSON file
    with the detected emotions and their corresponding ConceptNet concepts.

    Parameters:
        folder_path (str): Path to the folder containing the images.
        output_file (str): Path to the JSON file where the results will be saved.
    """
    annotator = annotation_situation["annotator"]

    art_emotion_mapping = {
        0: "amusement",
        1: "awe",
        2: "contentment",
        3: "excitement",
        4: "anger",
        5: "disgust",
        6: "fear",
        7: "sadness",
        8: "something else"
    }


    def detect_art_emotion(image_path):
        """
        Detects the emotion associated with an input image.

        Parameters:
            image_path (str): Path to the input image.

        Returns:
            str: Emotion label from the `art_emotion_mapping` or "Unknown" if the emotion is not found.
        """

        if annotator == "artemis_image-emotion-classifier":
            # Load the ImageEmotionClassifier object directly from the checkpoint
            checkpoint_file = "artemis_best_model.pt"
            model = torch.load(checkpoint_file, map_location=torch.device('cpu'))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

        image_net_mean = [0.485, 0.456, 0.406]
        image_net_std = [0.229, 0.224, 0.225]
        img_dim = 256
        img_transform = transforms.Compose([
            transforms.Resize((img_dim, img_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_net_mean, std=image_net_std),
        ])

        image = Image.open(image_path)
        image = image.convert("RGB")

        # Move the preprocessed image to the appropriate device (GPU or CPU)
        # Pass the image through the model and get the prediction
        with torch.no_grad():
            image = img_transform(image).unsqueeze(0)  # Add batch dimension
            image = image.to(device)
            logits = model(image)
            proba = logits.softmax(dim=-1)
            prediction = logits.argmax(dim=-1).item()
            top_emotion_prob, top_emotion_idx = proba.max(dim=-1)
            emotion_name = art_emotion_mapping.get(top_emotion_idx.item(), "Unknown")
        return emotion_name, top_emotion_prob.item()

    folder_art_emotions = {}
    annotation_situation_name = str(
        annotation_situation["annotated_dataset"] + "_" + annotation_situation["annotation_type"] + "_" +
        annotation_situation["annotation_time"])
    annotation_type = annotation_situation["annotation_type"]

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image_id = os.path.splitext(filename)[0]
            emotion, proba = detect_art_emotion(image_path)
            conceptnet_concept = get_conceptnet_concept(emotion)

            folder_art_emotions[image_id] = {}
            folder_art_emotions[image_id][annotation_type] = {}
            folder_art_emotions[image_id][annotation_type][annotation_situation_name] = {
                "emotion": emotion,
                "conceptnet_concept": conceptnet_concept,
                "annotation_strength": proba
            }

    with open(output_file, "w") as file:
        json.dump(folder_art_emotions, file, indent=2)

    print("Processing completed. JSON file generated:", output_file)

#
# annotation_situation = {
#     "annotation_type" : "em",
#     "annotator" : "artemis_image-emotion-classifier",
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
