"""
Perceptual Semantics - ConceptNet Matcher

This script provides functions to match labels from computer vision pipelines with ConceptNet concepts. It cleans the
input labels, checks their existence in ConceptNet, and finds the closest ConceptNet match for labels that do not
directly exist.

The script contains the following functions:

1. get_clean_label(label):
    Cleans the input label by converting it to lowercase, replacing spaces and dashes with underscores. If the label
    has a "conceptnet:" prefix, it is removed. Returns the cleaned label.

2. check_conceptnet_existence(clean_label):
    Checks if a concept with the given clean label exists in ConceptNet. It sends a request to ConceptNet's API and
    checks for a 200 status code. If the concept exists, it returns True along with the clean label, otherwise False.

3. find_closest_conceptnet_match(clean_label):
    Given a clean label, it checks if the concept exists in ConceptNet. If it does, the concept label is returned.
    Otherwise, it finds the closest matching concept by removing the first part of the label and recursively calling
    itself with the modified label. The closest match is returned as "conceptnet:" + modified label.

4. get_conceptnet_mapping_dict():
    Generates or loads the mapping of labels used in computer vision pipelines to their corresponding ConceptNet
    concepts. The mapping is saved in 'conceptnet_mapping.json' and returned as a dictionary.

5. get_conceptnet_concept(label):
    Takes a label from a computer vision pipeline and returns the corresponding ConceptNet concept by looking it up in
    the 'conceptnet_mapping.json' mapping. If the label corresponds to a color name, it maps it to the corresponding
    color name with spaces. Returns the ConceptNet concept associated with the input label.

Note: The mapping of labels to ConceptNet concepts is done based on predefined lists, including art styles, COCO
classes, emotions, and color names.

Author: Delfina Sol Martinez Pandiani
Date: 20 Jul 2023
"""

import os
import json
import requests
from bs4 import BeautifulSoup

def get_clean_label(label):
    """
        Cleans the input label by converting it to lowercase, replacing spaces and dashes with underscores. If the label
        has a "conceptnet:" prefix, it is removed. Returns the cleaned label.

        Parameters:
            label (str): The input label to be cleaned.

        Returns:
            str: The cleaned label with lowercase characters and underscores instead of spaces and dashes.
        """
    clean_label = label.lower().replace(" ", "_").replace("-", "_")
    # Check if the label has "conceptnet:" prefix, and remove it if present
    if clean_label.startswith("conceptnet:"):
        clean_label = clean_label[len("conceptnet:"):]
    return clean_label

def check_conceptnet_existence(clean_label):
    """
    Checks if a concept with the given clean label exists in ConceptNet. It sends a request to ConceptNet's API and
    checks for a 200 status code. If the concept exists, it returns True along with the clean label, otherwise False.

    Parameters:
        clean_label (str): The clean label for which existence in ConceptNet needs to be checked.

    Returns:
        tuple: A tuple containing a boolean value (True if the concept exists, False otherwise) and the clean label.
    """
    conceptnet_base_url = "https://conceptnet.io/c/en/"
    url = conceptnet_base_url + clean_label
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find('title')
        if title_tag and "not found" in title_tag.get_text().lower():
            print(f"{clean_label}: Does not exist in ConceptNet")
            return False, clean_label
        else:
            return True, clean_label
    else:
        print(f"{clean_label}: Does not exist in ConceptNet")
        return False, clean_label

def find_closest_conceptnet_match(clean_label):
    """
    Given a clean label, it checks if the concept exists in ConceptNet. If it does, the concept label is returned.
    Otherwise, it finds the closest matching concept by removing the first part of the label and recursively calling
    itself with the modified label. The closest match is returned as "conceptnet:" + modified label.

    Parameters:
        clean_label (str): The clean label for which the closest ConceptNet match needs to be found.

    Returns:
        str: The closest matching ConceptNet concept in the format "conceptnet:<concept_label>".
    """
    modified_label = '_'.join(clean_label.split('_')[1:])
    exists = check_conceptnet_existence(modified_label)
    if exists:
        concept_label = str("conceptnet:" + modified_label)
        print("Close match that exists is", concept_label)
    else:
        conceptnet_concept = find_closest_conceptnet_match(modified_label)
    return concept_label

def get_conceptnet_mapping_dict():
    """
    Generates or loads the mapping of labels used in computer vision pipelines to their corresponding ConceptNet
    concepts. The mapping is saved in 'conceptnet_mapping.json' and returned as a dictionary.

    Returns:
        dict: A dictionary containing the mapping of labels to their corresponding ConceptNet concepts.
    """
    # for object detection
    coco_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
                    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
                    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    # for art style detection
    art_styles = ["Art Nouveau", "Baroque", "Expressionism", "Impressionism", "Post-Impressionism", "Realism",
                  "Renaissance", "Romanticism", "Surrealism", "Ukiyo-e"]

    # for action detection
    actions = [
        "calling",
        "clapping",
        "cycling",
        "dancing",
        "drinking",
        "eating",
        "fighting",
        "hugging",
        "laughing",
        "listening_to_music",
        "running",
        "sitting",
        "sleeping",
        "texting",
        "using_laptop"
    ]
    # for emotion detection
    artemis_emotions = ['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness',
                        'something else']
    color_names = ["aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige", "bisque", "black",
                   "blanchedalmond", "blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse", "chocolate",
                   "coral", "cornflowerblue", "cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod",
                   "darkgray", "darkgreen", "darkgrey", "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange",
                   "darkorchid", "darkred", "darksalmon", "darkseagreen", "darkslateblue", "darkslategray",
                   "darkslategrey", "darkturquoise", "darkviolet", "deeppink", "deepskyblue", "dimgray", "dimgrey",
                   "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro", "ghostwhite",
                   "gold", "goldenrod", "gray", "green", "greenyellow", "grey", "honeydew", "hotpink", "indianred",
                   "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon", "lightblue",
                   "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgreen", "lightgrey",
                   "lightpink", "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray", "lightslategrey",
                   "lightsteelblue", "lightyellow", "lime", "limegreen", "linen", "magenta", "maroon",
                   "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen",
                   "mediumslateblue", "mediumspringgreen", "mediumturquoise", "mediumvioletred", "midnightblue",
                   "mintcream", "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive", "olivedrab",
                   "orange", "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise", "palevioletred",
                   "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple", "red", "rosybrown",
                   "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna", "silver",
                   "skyblue", "slateblue", "slategray", "slategrey", "snow", "springgreen", "steelblue", "tan", "teal",
                   "thistle", "tomato", "turquoise", "violet", "wheat", "white", "whitesmoke", "yellow", "yellowgreen"]
    # for color detection
    color_names_with_spaces = ["alice blue", "antique white", "aqua", "aquamarine", "azure", "beige", "bisque", "black",
                               "blanched almond", "blue", "blue violet", "brown", "burly wood", "cadet blue",
                               "chartreuse", "chocolate", "coral", "cornflower blue", "cornsilk", "crimson", "cyan",
                               "dark blue", "dark cyan", "dark goldenrod", "dark gray", "dark green", "dark grey",
                               "dark khaki", "dark magenta", "dark olive green", "dark orange", "dark orchid",
                               "dark red", "dark salmon", "dark sea green", "dark slate blue", "dark slate gray",
                               "dark slate grey", "dark turquoise", "dark violet", "deep pink", "deep sky blue",
                               "dim gray", "dim grey", "dodger blue", "firebrick", "floral white", "forest green",
                               "fuchsia", "gainsboro", "ghost white", "gold", "goldenrod", "gray", "green",
                               "green yellow", "grey", "honeydew", "hot pink", "indian red", "indigo", "ivory", "khaki",
                               "lavender", "lavender blush", "lawn green", "lemon chiffon", "light blue", "light coral",
                               "light cyan", "light goldenrod yellow", "light gray", "light green", "light grey",
                               "light pink", "light salmon", "light sea green", "light sky blue", "light slate gray",
                               "light slate grey", "light steel blue", "light yellow", "lime", "lime green", "linen",
                               "magenta", "maroon", "medium aquamarine", "medium blue", "medium orchid",
                               "medium purple", "medium sea green", "medium slate blue", "medium spring green",
                               "medium turquoise", "medium violet red", "midnight blue", "mint cream", "misty rose",
                               "moccasin", "navajo white", "navy", "old lace", "olive", "olive drab", "orange",
                               "orange red", "orchid", "pale goldenrod", "pale green", "pale turquoise",
                               "pale violet red", "papaya whip", "peach puff", "peru", "pink", "plum", "powder blue",
                               "purple", "red", "rosy brown", "royal blue", "saddle brown", "salmon", "sandy brown",
                               "sea green", "seashell", "sienna", "silver", "sky blue", "slate blue", "slate gray",
                               "slate grey", "snow", "spring green", "steel blue", "tan", "teal", "thistle", "tomato",
                               "turquoise", "violet", "wheat", "white", "white smoke", "yellow", "yellow green"]
    label_lists = [art_styles, actions, coco_classes, artemis_emotions, color_names_with_spaces]

    if os.path.exists("conceptnet_mapping.json"):
        with open("conceptnet_mapping.json", "r") as json_file:
            conceptnet_mapping_dict = json.load(json_file)
    else:
        conceptnet_mapping_dict = {}

        # Iterate over label_lists and check if the labels are already in the dictionary
    for label_list in label_lists:
        for label in label_list:
            if label not in conceptnet_mapping_dict:
                print(label, "not yet in dict")
                if label_list == color_names:
                    for index, label in enumerate(label_list):
                        clean_label = color_names_with_spaces[index]
                        existence, clean_label = check_conceptnet_existence(clean_label)
                else:
                    clean_label = get_clean_label(label)
                    existence, clean_label = check_conceptnet_existence(clean_label)

                if existence:
                    conceptnet_mapping_dict[label] = str("conceptnet:" + clean_label)
                else:
                    concept_label = find_closest_conceptnet_match(clean_label)
                    conceptnet_mapping_dict[label] = concept_label

        # Save the updated or newly created conceptnet_mapping_dict as a JSON file
    with open("conceptnet_mapping.json", "w") as json_file:
        json.dump(conceptnet_mapping_dict, json_file)

    return conceptnet_mapping_dict

def get_conceptnet_concept(label):
    """
    Takes a label from a computer vision pipeline and returns the corresponding ConceptNet concept by looking it up in
    the 'conceptnet_mapping.json' mapping. If the label corresponds to a color name, it maps it to the corresponding
    color name with spaces. Returns the ConceptNet concept associated with the input label.

    Parameters:
        label (str): The label from the computer vision pipeline.

    Returns:
        str: The corresponding ConceptNet concept for the input label in the format "conceptnet:<concept_label>".
    """
    color_names = ["aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige", "bisque", "black",
                   "blanchedalmond", "blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse", "chocolate",
                   "coral", "cornflowerblue", "cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod",
                   "darkgray", "darkgreen", "darkgrey", "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange",
                   "darkorchid", "darkred", "darksalmon", "darkseagreen", "darkslateblue", "darkslategray",
                   "darkslategrey", "darkturquoise", "darkviolet", "deeppink", "deepskyblue", "dimgray", "dimgrey",
                   "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro", "ghostwhite",
                   "gold", "goldenrod", "gray", "green", "greenyellow", "grey", "honeydew", "hotpink", "indianred",
                   "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon", "lightblue",
                   "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgreen", "lightgrey",
                   "lightpink", "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray", "lightslategrey",
                   "lightsteelblue", "lightyellow", "lime", "limegreen", "linen", "magenta", "maroon",
                   "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen",
                   "mediumslateblue", "mediumspringgreen", "mediumturquoise", "mediumvioletred", "midnightblue",
                   "mintcream", "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive", "olivedrab",
                   "orange", "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise", "palevioletred",
                   "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple", "red", "rosybrown",
                   "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna", "silver",
                   "skyblue", "slateblue", "slategray", "slategrey", "snow", "springgreen", "steelblue", "tan", "teal",
                   "thistle", "tomato", "turquoise", "violet", "wheat", "white", "whitesmoke", "yellow", "yellowgreen"]
    # for color detection
    color_names_with_spaces = ["alice blue", "antique white", "aqua", "aquamarine", "azure", "beige", "bisque", "black",
                               "blanched almond", "blue", "blue violet", "brown", "burly wood", "cadet blue",
                               "chartreuse", "chocolate", "coral", "cornflower blue", "cornsilk", "crimson", "cyan",
                               "dark blue", "dark cyan", "dark goldenrod", "dark gray", "dark green", "dark grey",
                               "dark khaki", "dark magenta", "dark olive green", "dark orange", "dark orchid",
                               "dark red", "dark salmon", "dark sea green", "dark slate blue", "dark slate gray",
                               "dark slate grey", "dark turquoise", "dark violet", "deep pink", "deep sky blue",
                               "dim gray", "dim grey", "dodger blue", "firebrick", "floral white", "forest green",
                               "fuchsia", "gainsboro", "ghost white", "gold", "goldenrod", "gray", "green",
                               "green yellow", "grey", "honeydew", "hot pink", "indian red", "indigo", "ivory", "khaki",
                               "lavender", "lavender blush", "lawn green", "lemon chiffon", "light blue", "light coral",
                               "light cyan", "light goldenrod yellow", "light gray", "light green", "light grey",
                               "light pink", "light salmon", "light sea green", "light sky blue", "light slate gray",
                               "light slate grey", "light steel blue", "light yellow", "lime", "lime green", "linen",
                               "magenta", "maroon", "medium aquamarine", "medium blue", "medium orchid",
                               "medium purple", "medium sea green", "medium slate blue", "medium spring green",
                               "medium turquoise", "medium violet red", "midnight blue", "mint cream", "misty rose",
                               "moccasin", "navajo white", "navy", "old lace", "olive", "olive drab", "orange",
                               "orange red", "orchid", "pale goldenrod", "pale green", "pale turquoise",
                               "pale violet red", "papaya whip", "peach puff", "peru", "pink", "plum", "powder blue",
                               "purple", "red", "rosy brown", "royal blue", "saddle brown", "salmon", "sandy brown",
                               "sea green", "seashell", "sienna", "silver", "sky blue", "slate blue", "slate gray",
                               "slate grey", "snow", "spring green", "steel blue", "tan", "teal", "thistle", "tomato",
                               "turquoise", "violet", "wheat", "white", "white smoke", "yellow", "yellow green"]
    mapping_file_path = os.path.join(os.path.dirname(__file__), "conceptnet_mapping.json")
    with open(mapping_file_path, "r") as json_file:
        conceptnet_mapping_dict = json.load(json_file)
    color_names_mapping = {label: color_names_with_spaces[index] for index, label in enumerate(color_names)}
    if label in color_names:
        label = color_names_mapping.get(label)
    conceptnet_concept = conceptnet_mapping_dict[label]
    return conceptnet_concept