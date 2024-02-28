"""
Perceptual Semantics - Color Detector

This script provides functionalities to automatically detect and match color labels (lexical units) in images with
ConceptNet concepts. It utilizes the ColorThief library to extract prominent colors from images and maps them to their
corresponding ConceptNet concepts using the ConceptNet Mapping module.

The script contains the following functions:

1. get_color_deets(image_path):
    Extracts the dominant colors from the given image using the ColorThief library. Each color is then matched to its
    corresponding ConceptNet concept, if available. The function returns a list of dictionaries, each containing the RGB
    values, web color name, and its ConceptNet concept.

2. process_folder(folder_path, output_file):
    Processes a folder containing multiple images. For each image, it uses `get_color_deets` to extract color details
    and stores the results in a dictionary. The color information for each image is then saved in a JSON file with the
    specified `output_file` name.

3. get_color_name(rgb):
    Utility function that takes an RGB color tuple as input and returns the closest matching CSS3 web color name. It
    calculates the Euclidean distance between the RGB values and the known CSS3 colors, and if the distance is below a
    threshold (50 in this script), it considers the color as a known CSS3 color and returns the formatted name. If the
    distance is higher, it returns "Unknown."

Note: The script provides a systematic and semantically enriched representation of colors in computer vision
applications by mapping color labels to ConceptNet concepts and using standardized CSS3 web color names.

Author: Delfina Sol Martinez Pandiani
Date: 20 Jul 2023
"""


import os
import sys
import json
from colorthief import ColorThief
from webcolors import CSS3_NAMES_TO_HEX, hex_to_rgb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .conceptnet_matching.conceptnet_mapping import get_conceptnet_concept


def process_folder(annotation_situation, folder_path, output_file):
    """
    Process a folder containing multiple images, extract color details, and save the results in a JSON file.

    Parameters:
        folder_path (str): The path to the folder containing image files.
        output_file (str): The name of the output JSON file.

    Returns:
        None
    """

    annotator = annotation_situation["annotator"]

    def get_color_name(rgb):
        """
        Get the closest matching CSS3 web color name for the given RGB color.

        This function takes an RGB color represented as a tuple containing three integers (r, g, b), where each value is in
        the range of 0 to 255. It calculates the distance between the input RGB color and the predefined CSS3 colors using
        the Euclidean distance formula. The distance is computed in a three-dimensional color space, where each axis
        corresponds to the intensity of the red, green, and blue channels, respectively.

        The function iterates over the predefined CSS3 color names and their corresponding RGB values to find the color that
        has the smallest distance to the input RGB color. The distance between two colors is defined as the Euclidean
        distance in the RGB space.

        Parameters:
            rgb (tuple): A tuple containing the RGB values of the color.

        Returns:
            str: The closest matching CSS3 web color name, or 'Unknown' if no close match is found.

        Note:
            The predefined CSS3 colors and their RGB values are obtained from the 'webcolors' library, which provides a
            dictionary (CSS3_NAMES_TO_HEX) mapping color names to their corresponding hexadecimal RGB values.

            The distance threshold of 50 used in this function is an arbitrary value, which can be adjusted based on
            specific requirements. Colors with distances greater than this threshold are considered as 'Unknown', meaning
            they do not closely match any predefined CSS3 color.
        """
        closest_color = None
        min_distance = float('inf')

        for name, hex_value in CSS3_NAMES_TO_HEX.items():
            r, g, b = hex_to_rgb(hex_value)
            distance = ((r - rgb[0]) ** 2 + (g - rgb[1]) ** 2 + (b - rgb[2]) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_color = name

        if min_distance > 50:  # Threshold for unknown colors (adjust as needed)
            return "Unknown"
        else:
            words = closest_color.split()
            formatted_name = "_".join(words[-2:])
            return formatted_name

    def get_color_deets(image_path):
        """
        Extract the dominant colors from the given image using the ColorThief library and map them to ConceptNet concepts.

        Parameters:
            image_path (str): The path to the image file.

        Returns:
            list: A list of dictionaries containing color details. Each dictionary contains the following keys:
                  - 'rgb': The RGB values of the color as a tuple.
                  - 'webcolor_name': The name of the color in CSS3 format.
                  - 'conceptnet_concept': The corresponding ConceptNet concept if available, or 'Unknown' otherwise.
        """

        color_thief = ColorThief(image_path)
        palette = color_thief.get_palette(color_count=5)
        color_details = []
        for color in palette:
            color_name = get_color_name(color)
            if color_name != "Unknown":
                conceptnet_concept = get_conceptnet_concept(color_name)
                color_detail = {
                    "rgb": color,
                    "webcolor_name": color_name,
                    "conceptnet_concept": conceptnet_concept
                }
                color_details.append(color_detail)
        return color_details

    folder_colors = {}
    annotation_situation_name = str(
        annotation_situation["annotated_dataset"] + "_" + annotation_situation["annotation_type"] + "_" +
        annotation_situation["annotation_time"])
    annotation_type = annotation_situation["annotation_type"]

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image_id = os.path.splitext(filename)[0]
            colors = get_color_deets(image_path)
            folder_colors[image_id] = {}
            folder_colors[image_id][annotation_type] = {}
            folder_colors[image_id][annotation_type][annotation_situation_name] = colors
    with open(output_file, "w") as file:
        json.dump(folder_colors, file, indent=2)
    print("Processing completed. JSON file generated:", output_file)



# annotation_situation = {
#     "annotation_type" : "color",
#     "annotator" : "ColorThief",
#     "annotation_place" : "Italy",
#     "annotation_time" : "2023_06_28",
#     "detection_threshold": "top five",
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
