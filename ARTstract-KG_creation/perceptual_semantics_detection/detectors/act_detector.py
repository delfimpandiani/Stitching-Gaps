from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .conceptnet_matching.conceptnet_mapping import get_conceptnet_concept


def process_folder(annotation_situation, folder_path, output_file):
    class_labels = {
        0: "calling",
        1: "clapping",
        2: "cycling",
        3: "dancing",
        4: "drinking",
        5: "eating",
        6: "fighting",
        7: "hugging",
        8: "laughing",
        9: "listening_to_music",
        10: "running",
        11: "sitting",
        12: "sleeping",
        13: "texting",
        14: "using_laptop"
    }

    def action_recognition(image, annotator):
        extractor = AutoFeatureExtractor.from_pretrained(annotator)
        model = AutoModelForImageClassification.from_pretrained(annotator)

        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        inputs = extractor(images=image, return_tensors="pt")
        inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=1)
        action_index = logits.argmax(-1)
        action_label = class_labels.get(action_index.item(), "Unknown")
        action_probability = probabilities[0][action_index].item()

        return action_label, action_probability

    def get_image_action(image_path, annotation_situation):
        annotator = annotation_situation["annotator"]
        image = Image.open(image_path)
        image = image.convert("RGB")
        action_label, action_probability = action_recognition(image, annotator)
        return action_label, action_probability

    annotation_situation_name = str(
        annotation_situation["annotated_dataset"] + "_" + annotation_situation["annotation_type"] + "_" +
        annotation_situation["annotation_time"])
    annotation_type = annotation_situation["annotation_type"]
    folder_actions = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image_id = os.path.splitext(filename)[0]
            action_label, action_probability = get_image_action(image_path, annotation_situation)
            conceptnet_concept = get_conceptnet_concept(action_label)
            folder_actions[image_id] = {}
            folder_actions[image_id][annotation_type] = {}
            folder_actions[image_id][annotation_type][annotation_situation_name] = {
                "action_label": action_label,
                "conceptnet_concept": conceptnet_concept,
                "annotation_strength": action_probability
            }

    with open(output_file, "w") as file:
        json.dump(folder_actions, file, indent=2)

    print("Processing completed. JSON file generated:", output_file)

# annotation_situation = {
#     "annotation_type" : "action",
#     "annotator" : "DunnBC22/vit-base-patch16-224-in21k_Human_Activity_Recognition",
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
