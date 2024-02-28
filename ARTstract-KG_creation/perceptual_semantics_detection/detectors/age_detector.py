"""
Perceptual Semantics - Human Presence Detection

This script uses a pre-trained logistic regression classifier, based on the 'ViT-B/32' variant of the CLIP (Contrastive Language-Image Pretraining) model, to automatically detect the presence of humans in a collection of images. The classifier was fine-tuned specifically for human presence detection in fashion-domain images by Adham Elarabawy.

The script processes images in a specified folder, and for each image, it predicts whether humans are present or not. The results are stored in a JSON file with the specified output name.

Author: Delfina Sol Martinez Pandiani
Date: 20 Jul 2023
"""
import os
import json
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
from .conceptnet_matching.conceptnet_mapping import get_conceptnet_concept


def process_folder(annotation_situation, folder_path, output_file):
    """
    Process a folder containing multiple images, estimate the age tiers for each image,
    and save the results in a JSON file.

    Parameters:
        folder_path (str): The path to the folder containing image files.
        output_file (str): The name of the output JSON file.

    Returns:
        None
    """
    annotator = annotation_situation["annotator"]
    model = ViTForImageClassification.from_pretrained(annotator)
    transforms = ViTFeatureExtractor.from_pretrained(annotator)

    age_tier_mapping = {
        0: '0-2',
        1: '3-9',
        2: '10-19',
        3: '20-29',
        4: '30-39',
        5: '40-49',
        6: '50-59',
        7: '60-69',
        8: '70+'
    }

    age_concept_mapping = {
        0: 'conceptnet:toddlerhood',
        1: 'conceptnet:childhood',
        2: 'conceptnet:teenagehood',
        3: 'conceptnet:twenties',
        4: 'conceptnet:thirties',
        5: 'conceptnet:fourties',
        6: 'conceptnet:fifties',
        7: 'conceptnet:sixties',
        8: 'conceptnet:old_age'
    }

    def get_image_age(image_path, age_tier_mapping):
        """
          Get the estimated age tier of a subject in the image.

          Parameters:
              image_path (str): The path to the image file.

          Returns:
              Tuple[int, float]: The estimated age tier index and the predicted class probability.
        """
        image = Image.open(image_path)
        image = image.convert("RGB")
        inputs = transforms(image, return_tensors='pt')
        output = model(**inputs)

        # Predicted Class probabilities
        proba = output.logits.softmax(1)

        # Get the top predicted class and its probability
        pred_class_index = proba.argmax(1).item()
        pred_class_prob = proba[0, pred_class_index].item()

        pred_age_tier = age_tier_mapping.get(pred_class_index, 'Unknown')
        pred_concept = age_concept_mapping.get(pred_class_index, 'Unknown')

        return pred_age_tier, pred_concept, pred_class_prob


    folder_ages = {}
    annotation_situation_name = str(
        annotation_situation["annotated_dataset"] + "_" + annotation_situation["annotation_type"] + "_" +
        annotation_situation["annotation_time"])
    annotation_type = annotation_situation["annotation_type"]

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image_id = os.path.splitext(filename)[0]
            pred_age_tier, pred_concept, pred_class_prob = get_image_age(image_path, age_tier_mapping)
            folder_ages[image_id] = {}
            folder_ages[image_id][annotation_type] = {}
            folder_ages[image_id][annotation_type][annotation_situation_name] = {
                "age_tier": pred_age_tier,
                "conceptnet_concept": pred_concept,
                "annotation_strength": pred_class_prob
            }

    with open(output_file, "w") as file:
        json.dump(folder_ages, file, indent=2)

    print("Processing completed. JSON file generated:", output_file)
    return


#
# # Execution over all Abstract Concept image folders.
# # concept_names = ["comfort", "danger", "death", "fitness", "freedom", "power", "safety"]
# concept_names = ["freedom", "power", "safety"]
# for concept_name in concept_names:
#     folder_path = f"/home/delfino/GitHub/ARTstract_Seeing_abstract_concepts/ARTstract_Dataset_v0.1/Local_ARTstract_Images_v0.1/Local_structured_dataset/{concept_name}"
#     output_file = f'age_output_{concept_name}.json'
#     process_folder(folder_path, output_file)
#     print(
#         f"//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// DONE WITH CONCEPT {concept_name}///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
#
# # folder_path = '../__prova/test'
# # output_file = 'output.json'
# # process_folder(folder_path, output_file)





# annotation_situation = {
#     "annotation_type" : "age",
#     "annotator" : "nateraw/vit-age-classifier",
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
