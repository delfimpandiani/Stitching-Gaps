import json
from detectors import act_detector, hp_detector, as_detector, od_detector, age_detector, em_detector, ic_detector, color_detector

def call_process_folder(detection_type, annotation_situation, folder_path, output_file):
    if detection_type == "act":
        act_detector.process_folder(annotation_situation, folder_path, output_file)
    elif detection_type == "age":
        age_detector.process_folder(annotation_situation, folder_path, output_file)
    elif detection_type == "as":
        as_detector.process_folder(annotation_situation, folder_path, output_file)
    elif detection_type == "color":
        color_detector.process_folder(annotation_situation, folder_path, output_file)
    elif detection_type == "em":
        em_detector.process_folder(annotation_situation, folder_path, output_file)
    elif detection_type == "hp":
        hp_detector.process_folder(annotation_situation, folder_path, output_file)
    elif detection_type == "ic":
        ic_detector.process_folder(annotation_situation, folder_path, output_file)
    elif detection_type == "od":
        od_detector.process_folder(annotation_situation, folder_path, output_file)
    else:
        raise ValueError("Invalid module name.")

def detect_for_all_classes(annotation_situation):
    detection_type = annotation_situation["annotation_type"]
    # concept_names = ["comfort", "danger", "death", "fitness", "freedom", "power", "safety"]
    concept_names = ["comfort", "danger", "death", "fitness", "freedom", "power"]
    for concept_name in concept_names:
        folder_path = f"/home/delfino/GitHub/ARTstract_Seeing_abstract_concepts/ARTstract_Dataset_v0.1/Local_ARTstract_Images_v0.1/Local_structured_dataset/{concept_name}"
        output_file = f'output/{detection_type}_{concept_name}.json'
        call_process_folder(detection_type, annotation_situation, folder_path, output_file)
        print(
            f"////////////////////////// DONE WITH CONCEPT {detection_type} for {concept_name}///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")

with open('input/annotation_situations.json', 'r') as file:
    annotation_situations = json.load(file)
    annotation_situations = annotation_situations["annotation_situations"]

for annotation_key, annotation_situation in annotation_situations.items():
    detection_type = annotation_situation["annotation_type"]
    folder_path = '__prova/explain_test'
    # output_file = f'test/{annotation_situation["annotation_type"]}_output.json'
    # output_short_file = f'../{annotation_situation["annotation_type"]}_output_short.json'
    # call_process_folder(detection_type, annotation_situation, folder_path, output_file)
    output_file = f'__prova/explain_test/{annotation_situation["annotation_type"]}_output.json'
    # output_short_file = f'../{annotation_situation["annotation_type"]}_output_short.json'
    call_process_folder(detection_type, annotation_situation, folder_path, output_file)
    print('doing annotation of type ', detection_type)
    # detect_for_all_classes(annotation_situation)

