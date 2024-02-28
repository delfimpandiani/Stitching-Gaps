import json
import nltk
import itertools
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import webcolors
import wordcloud

from collections import Counter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, get_single_color_func
from collections import defaultdict
from itertools import combinations

matplotlib.use('Agg')
plt.interactive(False)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import random

def load_inputs(ACs_list_name):
    with open(f"../input/{ACs_list_name}.json", "r") as file:
        concept_images = json.load(file)
    with open("../input/merged_ARTstract.json", "r") as file:
        merged_ARTstract = json.load(file)
        return concept_images, merged_ARTstract

## Abstract Concept Evocation frequencies
def stats_concept_frequencies(ACs_list_names, dataset_colors, concept_colors):
    def concept_frequency_in_source_datasets(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        # Information on the source dataset for each image
        source_datasets = {}
        for image_id, image_info in merged_ARTstract.items():
            if "source_dataset" in image_info:
                source_datasets[image_id] = image_info["source_dataset"]

        # Initialize a dictionary to store the frequency of each concept in each source dataset
        concept_frequency = {concept: {dataset: 0 for dataset in set(source_datasets.values())} for concept in concept_images}

        # Count the frequency of each concept in each source dataset
        for concept, images in concept_images.items():
            for image in images:
                dataset = source_datasets.get(image)
                if dataset:
                    concept_frequency[concept][dataset] += 1

        # Print the results
        for concept, frequencies in concept_frequency.items():
            print(f"Concept: {concept}")
            for dataset, frequency in frequencies.items():
                print(f"  Source Dataset: {dataset}, Frequency: {frequency}")

        return concept_frequency

    def plot_concept_frequencies(ACs_list_name, dataset_colors, concept_colors):
        concept_frequencies = concept_frequency_in_source_datasets(ACs_list_name)
        # Extract the concepts and source datasets
        concepts = list(concept_frequencies.keys())
        source_datasets = set(dataset for frequencies in concept_frequencies.values() for dataset in frequencies.keys())

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.2
        index = list(range(len(concepts)))
        index = np.array(index)


        # Use numpy.arange to calculate the x-coordinates for the bars
        x_coordinates = np.arange(len(concepts))
        concepts = sorted(list(concept_frequencies.keys()))  # Sort concepts alphabetically

        # Sort the source datasets alphabetically to ensure consistent color assignment
        sorted_datasets = sorted(source_datasets)

        # Create a bar for each source dataset
        for i, dataset in enumerate(sorted_datasets):
            frequencies = [concept_frequencies[concept].get(dataset, 0) for concept in concepts]
            color = concept_colors[i % len(concept_colors)] if concept_colors else None  # Use custom colors in a cycle
            ax.bar(x_coordinates + i * bar_width, frequencies, bar_width, label=dataset, color=color)

        # Add labels, title, and legend
        ax.set_xlabel('Concepts')
        ax.set_ylabel('Frequency')
        ax.set_title('Concept Frequencies in Each Source Dataset')
        ax.set_xticks(index + bar_width * (len(source_datasets) - 1) / 2)
        ax.set_xticklabels(concepts)
        ax.legend()

        save_filename = f'output_imgs/evocation_data/concept_frequencies/concept_frequencies_{ACs_list_name}.png'
        # Show the plot
        plt.tight_layout()
        plt.show()
        # Save the plot as an image
        plt.savefig(save_filename)
        plt.show()

    for ACs_list_name in ACs_list_names:
        plot_concept_frequencies(ACs_list_name, dataset_colors, concept_colors)
    return

## Abstract Concept Evocation strengths
def stats_evocation_strengths(ACs_list_names, dataset_colors, concept_colors, plot_type):
    def get_evocation_strength_by_image(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        evocation_strength_by_concept = {concept: [] for concept in concept_images}

        for concept, images in concept_images.items():
            for img in images:
                for image_id, image_info in merged_ARTstract.items():
                    if img == image_id:
                        evoked_clusters = merged_ARTstract[img].get("evoked_clusters", {})
                        for cluster_id, cluster_info in evoked_clusters.items():
                            img_evocation_strength = cluster_info.get("evocation_strength")
                        evocation_strength_by_concept[concept].append(img_evocation_strength)

        return evocation_strength_by_concept

    def calculate_average_evocation_strength_by_concept(ACs_list_name):
        evocation_strength_by_concept = get_evocation_strength_by_image(ACs_list_name)
        average_strength_by_concept = {}
        for concept, strengths in evocation_strength_by_concept.items():
            if len(strengths) > 0:
                average_strength = sum(strengths) / len(strengths)
                average_strength_by_concept[concept] = average_strength

        return average_strength_by_concept

    def plot_evocation_strengths(ACs_list_name, dataset_colors=None,
                                 concept_colors=None, plot_type="plot"):
        evocation_strength_by_concept = get_evocation_strength_by_image(ACs_list_name)
        # Extract the concepts and average evocation strengths
        concepts = list(evocation_strength_by_concept.keys())
        average_strengths = [sum(strengths) / len(strengths) if len(strengths) > 0 else 0.0 for strengths in
                             evocation_strength_by_concept.values()]

        # Sort the concepts alphabetically
        sorted_indices = np.argsort(concepts)
        concepts_sorted = [concepts[i] for i in sorted_indices]
        average_strengths_sorted = [average_strengths[i] for i in sorted_indices]

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use numpy.arange to calculate the x-coordinates for the concepts
        x_coordinates = np.arange(len(concepts))

        # Get the concept colors in the same order as the sorted concepts
        concept_colors_ordered = [concept_colors[i % len(concept_colors)] for i in range(len(concepts_sorted))]

        if plot_type == "plot":
            # Plot the average evocation strengths using a line plot
            ax.plot(x_coordinates, average_strengths_sorted, marker='o', color='b')


        elif plot_type == "bar":
            # Create a bar chart with each concept having its corresponding color
            ax.bar(x_coordinates, average_strengths_sorted, color=concept_colors_ordered)

        # Add labels, title, and legend
        ax.set_xlabel('Concepts')
        ax.set_ylabel('Average Evocation Strength')
        ax.set_title('Average Evocation Strength by Concept')
        ax.set_xticks(x_coordinates)
        ax.set_xticklabels(concepts, rotation=45, ha='right')

        ax.set_ylim(1, 1.5)

        # Show the plot
        plt.tight_layout()
        plt.show()

        save_filename = f'output_imgs/evocation_data/evocation_strength/{plot_type}_evocation_strengths_{ACs_list_name}.png'
        # Save the plot as an image
        plt.savefig(save_filename)
        plt.show()

        return

    for ACs_list_name in ACs_list_names:
        plot_evocation_strengths(ACs_list_name, dataset_colors=dataset_colors, concept_colors=concept_colors, plot_type=plot_type)
    return

## Object detection
def stats_num_detected_objects(ACs_list_names, dataset_colors, concept_colors, plot_type):
    def get_detected_objects_by_image(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        detected_objects_by_concept = {concept: [] for concept in concept_images}
        num_detected_objects_by_concept = {concept: [] for concept in concept_images}

        for concept, images in concept_images.items():
            for img in images:
                for image_id, image_info in merged_ARTstract.items():
                    if img == image_id:
                        detected_objects_list = merged_ARTstract[img].get("od", {}).get("ARTstract_od_2023_06_28",
                                                                                        {}).get("detected_objects", [])
                        detected_objects = [detected_object["detected_object"] for detected_object in
                                            detected_objects_list]
                        number_of_detected_objects = len(detected_objects)
                        detected_objects_by_concept[concept].extend(detected_objects)
                        num_detected_objects_by_concept[concept].append(number_of_detected_objects)
        # print(detected_objects_by_concept)
        return num_detected_objects_by_concept, detected_objects_by_concept

    def calculate_average_num_detected_objects_by_concept(ACs_list_name):
        num_detected_objects_by_concept, detected_objects_by_concept = get_detected_objects_by_image(ACs_list_name)
        print(num_detected_objects_by_concept)
        average_num_detected_objects_by_concept = {}
        for concept, num_detected_objects in num_detected_objects_by_concept.items():
            if len(num_detected_objects) > 0:
                average_num_detected_objects = sum(num_detected_objects) / len(num_detected_objects)
                average_num_detected_objects_by_concept[concept] = average_num_detected_objects

        return average_num_detected_objects_by_concept

    def plot_avg_num_detected_objects(ACs_list_name, dataset_colors, concept_colors, plot_type):
        num_detected_objects_by_concept, detected_objects_by_concept = get_detected_objects_by_image(ACs_list_name)
        concepts = list(num_detected_objects_by_concept.keys())
        average_num_detected_objects = [sum(num_detected_objects) / len(num_detected_objects) if len(num_detected_objects) > 0 else 0.0 for num_detected_objects in
                              num_detected_objects_by_concept.values()]

        # Sort the concepts alphabetically
        sorted_indices = np.argsort(concepts)
        concepts_sorted = [concepts[i] for i in sorted_indices]
        average_num_detected_objects_sorted = [average_num_detected_objects[i] for i in sorted_indices]

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use numpy.arange to calculate the x-coordinates for the concepts
        x_coordinates = np.arange(len(concepts))

        # Get the concept colors in the same order as the sorted concepts
        concept_colors_ordered = [concept_colors[i % len(concept_colors)] for i in range(len(concepts_sorted))]

        if plot_type == "plot":
            # Plot the average evocation strengths using a line plot
            ax.plot(x_coordinates, average_num_detected_objects_sorted, marker='o', color='b')

        elif plot_type == "bar":
            # Create a bar chart with each concept having its corresponding color
            ax.bar(x_coordinates, average_num_detected_objects_sorted, color=concept_colors_ordered)

        # Add labels, title, and legend
        ax.set_xlabel('Concepts')
        ax.set_ylabel('Average Number of Detected Objects')
        ax.set_title('Average Number of Detected Objects by Concept')
        ax.set_xticks(x_coordinates)
        ax.set_xticklabels(concepts, rotation=45, ha='right')
        ax.set_ylim(1, 5)
        save_filename = f'output_imgs/perceptual_data/detected_objects/{plot_type}_num_detected_objects_{ACs_list_name}.png'
        # Show the plot
        plt.tight_layout()
        plt.show()
        # Save the plot as an image
        plt.savefig(save_filename)
        plt.show()
        return

    for ACs_list_name in ACs_list_names:
        plot_avg_num_detected_objects(ACs_list_name, dataset_colors, concept_colors, plot_type)
        return

def stats_detected_objects(ACs_list_names):
    def get_detected_objects_by_concept(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        # for concept, list in concept_images.items():
            # print(concept, "has these many images ", len(list))
        detected_objects_by_concept = {concept: [] for concept in concept_images}
        for concept, images in concept_images.items():
            for img in images:
                for image_id, image_info in merged_ARTstract.items():
                    if img == image_id:
                        detected_objects_list = merged_ARTstract[img].get("od", {}).get("ARTstract_od_2023_06_28",
                                                                                        {}).get("detected_objects",
                                                                                                [])
                        detected_objects = [detected_object["detected_object"] for detected_object in
                                            detected_objects_list]
                        detected_objects_by_concept[concept].extend(detected_objects)
        # print(detected_objects_by_concept)
        return detected_objects_by_concept

    def calculate_object_frequencies(ACs_list_name):
        detected_objects_by_concept = get_detected_objects_by_concept(ACs_list_name)
        object_frequencies_by_concept = {}
        all_detected_objects = []

        for concept, detected_objects in detected_objects_by_concept.items():
            object_frequencies = Counter(detected_objects)
            object_frequencies_by_concept[concept] = object_frequencies
            all_detected_objects.extend(detected_objects)

        all_object_frequencies = Counter(all_detected_objects)
        return object_frequencies_by_concept, all_object_frequencies

    def plot_object_frequencies(ACs_list_name):
        object_frequencies_by_concept, all_object_frequencies = calculate_object_frequencies(ACs_list_name)
        num_concepts = len(object_frequencies_by_concept)
        fig, axs = plt.subplots(num_concepts + 1, 1, figsize=(12, 5 * (num_concepts + 1)), sharex=True)

        # Calculate top 30 objects across all concepts, excluding the top object "person"
        all_object_frequencies = {}
        for concept, object_frequencies in object_frequencies_by_concept.items():
            for obj, freq in object_frequencies.items():
                if obj != "person":
                    all_object_frequencies[obj] = all_object_frequencies.get(obj, 0) + freq

        sorted_all_frequencies = sorted(all_object_frequencies.items(), key=lambda x: x[1], reverse=True)
        top_all_objects, top_all_frequencies = zip(*sorted_all_frequencies[:30])

        # Plot top 30 objects across all concepts, excluding the top object "person"
        axs[0].bar(top_all_objects, top_all_frequencies)
        axs[0].set_title("Top 30 Detected Objects Across All Concepts (excluding 'person')")
        axs[0].tick_params(axis='x', rotation=45, labelsize=8)

        for i, (concept, object_frequencies) in enumerate(object_frequencies_by_concept.items(), start=1):
            sorted_frequencies = sorted(object_frequencies.items(), key=lambda x: x[1], reverse=True)
            objects, frequencies = zip(
                *sorted_frequencies[1:31])  # Taking top 30 objects, excluding the first object "person"
            axs[i].bar(objects, frequencies)
            axs[i].set_title(f"Detected Objects for Concept '{concept}'")
            axs[i].tick_params(axis='x', rotation=45, labelsize=8)

        plt.tight_layout()
        plt.show()

        # Save the plot as an image
        save_filename = f"output_imgs/perceptual_data/detected_objects/top_30_{ACs_list_name}.jpg"
        plt.savefig(save_filename)
        plt.show()

    def find_common_objects(ACs_list_name):
        object_frequencies_by_concept, all_object_frequencies = calculate_object_frequencies(ACs_list_name)
        # Calculate overall frequency of each object across all concepts
        all_object_frequencies = {}
        for object_frequencies in object_frequencies_by_concept.values():
            for obj, freq in object_frequencies.items():
                all_object_frequencies[obj] = all_object_frequencies.get(obj, 0) + freq

        # Sort objects based on their overall frequency in descending order
        sorted_overall_frequencies = sorted(all_object_frequencies.items(), key=lambda x: x[1], reverse=True)
        ordered_objects = [obj for obj, freq in sorted_overall_frequencies]

        # Get the set of common objects that are present in every concept
        common_objects = set.intersection(
            *[set(frequencies.keys()) for frequencies in object_frequencies_by_concept.values()])

        # Return the ordered set of common objects
        ordered_common_objects = [obj for obj in ordered_objects if obj in common_objects]
        # common_objects = find_common_objects(object_frequencies_by_concept)
        print("for all concepts, the most common objects are (from most common to least: ", common_objects)
        return ordered_common_objects

    def find_unique_objects(ACs_list_name):
        object_frequencies_by_concept, all_object_frequencies = calculate_object_frequencies(ACs_list_name)

        unique_objects_by_concept = {concept: set(frequencies.keys()) for concept, frequencies in
                                     object_frequencies_by_concept.items()}
        for concept, frequencies in object_frequencies_by_concept.items():
            for other_concept, other_frequencies in object_frequencies_by_concept.items():
                if concept != other_concept:
                    unique_objects_by_concept[concept] -= set(other_frequencies.keys())

        for concept, od_set in unique_objects_by_concept.items():
            if len(od_set) == 0:
                print(concept, "does not have any unique caption words")
            else:
                print(concept, "is the only one with detected objects: ", od_set)

        return unique_objects_by_concept

    def find_relevant_objects(ACs_list_name):
        object_frequencies_by_concept, all_object_frequencies = calculate_object_frequencies(ACs_list_name)

        # Calculate overall frequency of each object across all concepts
        all_object_frequencies = Counter()
        for object_frequencies in object_frequencies_by_concept.values():
            all_object_frequencies.update(object_frequencies)

        relevant_objects_by_concept = {}

        for concept, object_frequencies in object_frequencies_by_concept.items():
            # Calculate relative frequency (TF) for each object within the concept
            relative_frequencies = {
                obj: freq / all_object_frequencies[obj]
                for obj, freq in object_frequencies.items()
            }

            # Calculate inverse concept frequency (IDF) for each object
            num_concepts = len(object_frequencies_by_concept)
            inverse_concept_frequency = {obj: num_concepts / sum(
                1 for concept_freqs in object_frequencies_by_concept.values() if obj in concept_freqs)
                                         for obj in object_frequencies}

            # Calculate relevance score for each object in the concept (TF * IDF)
            relevance_scores = {obj: round(relative_frequencies[obj] * inverse_concept_frequency[obj], 3)
                                for obj in object_frequencies}

            # Sort objects based on their relevance scores in descending order
            sorted_relevance_scores = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

            # Keep only the objects unique to the current concept's top 10 objects
            top_10_objects = [obj for obj, score in sorted_relevance_scores[:15]]

            # Check if each object appears in only the current concept's top 10 list
            unique_top_10_objects = []
            for obj in top_10_objects:
                unique_to_current_concept = True
                for other_concept, other_top_10 in relevant_objects_by_concept.items():
                    if other_concept != concept and obj in other_top_10:
                        unique_to_current_concept = False
                        break
                if unique_to_current_concept:
                    unique_top_10_objects.append(obj)

            relevant_objects_by_concept[concept] = unique_top_10_objects
        # print(relevant_objects_by_concept)
        # for concept, od_set in relevant_objects_by_concept.items():
        #     print(concept, "has relevant concepts: ", od_set)
        # print(relevant_objects_by_concept)
        return relevant_objects_by_concept

    def find_top_objects(ACs_list_name, num_top_objects=10):
        object_frequencies_by_concept, all_object_frequencies = calculate_object_frequencies(ACs_list_name)

        top_objects_by_concept = {}
        for concept, frequencies in object_frequencies_by_concept.items():
            total_detected_objects = sum(frequencies.values())
            objects_percentages = {
                obj: (count / total_detected_objects) * 100 for obj, count in frequencies.items()
            }
            sorted_objects = sorted(objects_percentages.items(), key=lambda x: x[1], reverse=True)
            top_objects = dict(sorted_objects[:num_top_objects])
            top_objects_by_concept[concept] = top_objects
        # print("Top objects by concept: ", top_objects_by_concept)
        ordered_lists_by_concept = {}
        for concept, object_scores in top_objects_by_concept.items():
            # Sort objects based on their scores in descending order
            sorted_objects = sorted(object_scores.items(), key=lambda x: x[1], reverse=True)
            # Extract the objects from the sorted list
            ordered_objects = [obj for obj, score in sorted_objects]
            ordered_lists_by_concept[concept] = ordered_objects
        # for concept, od_set in ordered_lists_by_concept.items():
        #     print(concept, "has top concepts: ", od_set)
        return top_objects_by_concept

    def find_top_relevant_objects(ACs_list_name, k=15):
        top_objects_by_concept = find_top_objects(ACs_list_name)
        relevant_objects_by_concept = find_relevant_objects(ACs_list_name)

        top_relevant_objects_by_concept = {}

        for concept, top_concepts in top_objects_by_concept.items():
            # Get the relevant concepts for the current concept
            relevant_concepts = relevant_objects_by_concept.get(concept, [])
            # Calculate the relevance scores for the relevant concepts
            relevance_scores = {obj: 1 for obj in relevant_concepts}
            # Convert top_concepts dictionary to a list of tuples (object, score)
            top_concepts_list = list(top_concepts.items())
            # Sort the top_concepts_list based on scores in descending order
            top_concepts_list.sort(key=lambda x: x[1], reverse=True)
            # Extract the objects from the sorted list
            top_concepts_sorted = [obj for obj, _ in top_concepts_list]
            # Ensure all objects in top_concepts_sorted have scores in relevance_scores dictionary
            relevance_scores.update({obj: 0 for obj in top_concepts_sorted if obj not in relevance_scores})
            # Calculate the Jaccard similarity between the top concepts and relevant concepts
            jaccard_scores = {obj: len(set([obj]).intersection(set(top_concepts_sorted))) / len(
                set([obj]).union(set(top_concepts_sorted))) for obj in relevant_concepts}
            # Sort objects based on Jaccard similarity scores in descending order
            sorted_objects = sorted(jaccard_scores.items(), key=lambda x: x[1], reverse=True)
            # Take only the top k relevant concepts
            top_relevant_objects_by_concept[concept] = [obj for obj, _ in sorted_objects[:k]]
            # Print the top relevant concepts
        # for concept, concepts in top_relevant_objects_by_concept.items():
        #     print(concept, "has top relevant objects:", concepts)
        # print(top_relevant_objects_by_concept)

        file_path = "top_relevant_jsons/top_relevant_objects_by_concept.json"
        # Using json.dump() to write the dictionary to a JSON file
        with open(file_path, "w") as json_file:
            json.dump(top_relevant_objects_by_concept, json_file)

        return top_relevant_objects_by_concept

    def find_top_relevant_objects_by_concept_w_freqs(ACs_list_name):
        top_relevant_objects_by_concept = find_top_relevant_objects(ACs_list_name)
        object_frequencies_by_concept, all_object_frequencies = calculate_object_frequencies(ACs_list_name)
        top_relevant_objects_by_concept_w_freqs = {}

        for concept, top_relevant_objects in top_relevant_objects_by_concept.items():
            # Get the object frequencies for the current concept
            object_frequencies = object_frequencies_by_concept.get(concept, {})

            # Fetch frequencies for top relevant objects and create a dictionary {object: frequency}
            objects_with_freqs = {obj: object_frequencies.get(obj, 0) for obj in top_relevant_objects}

            # Store the dictionary {object: frequency} for the current concept
            top_relevant_objects_by_concept_w_freqs[concept] = objects_with_freqs

        # Print the top relevant concepts and their frequencies
        # for concept, concepts in top_relevant_objects_by_concept_w_freqs.items():
        #     print(concept, "has top relevant concepts and frequencies:", concepts)

        return top_relevant_objects_by_concept_w_freqs

    def create_concepts_wordclouds(ACs_list_name):
        top_objects_by_concept = find_top_objects(ACs_list_name)
        top_relevant_objects_by_concept_w_freqs = find_top_relevant_objects_by_concept_w_freqs(ACs_list_name)
        class GroupedColorFunc(object):
            def __init__(self, color_to_words, default_color):
                self.color_func_to_words = [
                    (get_single_color_func(color), set(words))
                    for (color, words) in color_to_words.items()]

                self.default_color_func = get_single_color_func(default_color)

            def get_color_func(self, word):
                """Returns a single_color_func associated with the word"""
                try:
                    color_func = next(
                        color_func for (color_func, words) in self.color_func_to_words
                        if word in words)
                except StopIteration:
                    color_func = self.default_color_func

                return color_func

            def __call__(self, word, **kwargs):
                return self.get_color_func(word)(word, **kwargs)

        font_color = '#0074D9'  # Use any shade of blue you prefer
        helvetica_font = 'Helvetica.ttf'  # Replace with the path to your Helvetica font file

        # Remove "person" from top_objects_by_concept and top_relevant_objects_by_concept
        for top_objects in top_objects_by_concept.values():
            top_objects.pop('person', None)

        for top_relevant_objects in top_relevant_objects_by_concept_w_freqs.values():
            if 'person' in top_relevant_objects:
                top_relevant_objects.remove('person')

        # Generate word clouds for each concept
        for concept, top_objects in top_objects_by_concept.items():
            # Create word cloud objects
            wc_top_objects = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,  # Set the font to Helvetica
                color_func=lambda *args, **kwargs: font_color  # Set all words to blue color
            ).generate_from_frequencies(top_objects)

            # Get the top relevant objects for the current concept
            top_relevant_objects = top_relevant_objects_by_concept_w_freqs.get(concept, {})
            top_relevant_objects = top_relevant_objects_by_concept_w_freqs.get(concept, {})
            wc_top_relevant_objects = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,  # Set the font to Helvetica
                color_func=lambda *args, **kwargs: font_color  # Set all words to blue color
            ).generate_from_frequencies(top_relevant_objects)

            # Plot the word clouds side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(wc_top_objects, interpolation="bilinear")
            axes[0].axis("off")
            axes[0].set_title(f"Top Objects for Concept: {concept}")

            axes[1].imshow(wc_top_relevant_objects, interpolation="bilinear")
            axes[1].axis("off")
            axes[1].set_title(f"Top Relevant Objects for Concept: {concept}")

            plt.show()
            # Save the plot as an image
            save_filename = f"output_imgs/perceptual_data/detected_objects/wordclouds/{concept}_{ACs_list_name}_object_wordcloud.jpg"
            plt.savefig(save_filename)
            plt.show()

        return

    for ACs_list_name in ACs_list_names:
        ## EXECUTION
        # get_detected_objects_by_concept(ACs_list_name)
        # calculate_object_frequencies(ACs_list_name)
        # plot_object_frequencies(ACs_list_name)
        # find_common_objects(ACs_list_name)
        # find_unique_objects(ACs_list_name)
        # find_relevant_objects(ACs_list_name)
        # find_top_objects(ACs_list_name)
        find_top_relevant_objects(ACs_list_name)
        # find_top_relevant_objects_by_concept_w_freqs(ACs_list_name)
        # create_concepts_wordclouds(ACs_list_name)
    return

def object_co_occurence_heatmaps(ACs_list_names, consider_person):
    def create_co_occurrence_matrix(concept_detected_objects, consider_person):
        # Flatten the list of lists to get all detected object names
        all_object_names = [obj_name for sublist in concept_detected_objects for obj_name in sublist]

        # If consider_person is False, remove 'person' from the object names
        if not consider_person:
            all_object_names = [obj_name for obj_name in all_object_names if obj_name != 'person']

        # Extract unique object names from the flattened list and sort alphabetically
        object_names = sorted(list(set(all_object_names)))
        print('object names is a list ', object_names)

        # Initialize an empty co-occurrence matrix
        num_objects = len(object_names)
        co_occurrence_matrix = np.zeros((num_objects, num_objects), dtype=int)
        print('initial cooccr matrix', co_occurrence_matrix)

        # Create a dictionary to map object names to matrix indices
        object_to_index = {obj_name: index for index, obj_name in enumerate(object_names)}

        # Populate the co-occurrence matrix based on the flattened list
        for detected_objects in concept_detected_objects:
            if not consider_person:
                detected_objects = [obj_name for obj_name in detected_objects if obj_name != 'person']
            for obj_name in detected_objects:
                for other_obj_name in detected_objects:
                    if obj_name != other_obj_name:
                        # Increase the count for co-occurrence of obj_name and other_obj_name
                        i, j = object_to_index[obj_name], object_to_index[other_obj_name]
                        co_occurrence_matrix[i, j] += 1

        print('updated cooccur matrix', co_occurrence_matrix)
        return co_occurrence_matrix, object_names

    def get_co_occurrence_pairs(concept_of_interest, concept_detected_objects, consider_person):
        # Flatten the list of lists to get all detected object names
        all_object_names = [obj_name for sublist in concept_detected_objects for obj_name in sublist]

        # If consider_person is False, remove 'person' from the object names
        if not consider_person:
            all_object_names = [obj_name for obj_name in all_object_names if obj_name != 'person']

        # Count the total number of images for the concept
        total_images = len(concept_detected_objects)

        # Initialize an empty dictionary to store co-occurrence percentages
        co_occurrence_percentages = {}

        # Populate the co-occurrence percentages based on the flattened list
        for detected_objects in concept_detected_objects:
            if not consider_person:
                detected_objects = [obj_name for obj_name in detected_objects if obj_name != 'person']
            for obj_name in detected_objects:
                for other_obj_name in detected_objects:
                    if obj_name != other_obj_name:
                        # Sort the objects alphabetically to ensure consistent ordering
                        co_occurrence_pair = tuple(sorted([obj_name, other_obj_name]))

                        # Increase the count for the co-occurrence pair
                        co_occurrence_percentages[co_occurrence_pair] = co_occurrence_percentages.get(
                            co_occurrence_pair, 0) + 1

        # Convert counts to percentages by dividing by the total number of images
        for pair, count in co_occurrence_percentages.items():
            co_occurrence_percentages[pair] = round(count / total_images * 100, 2)

        # Sort the co-occurrence pairs by percentage in descending order
        sorted_co_occurrence_pairs = sorted(co_occurrence_percentages.items(), key=lambda x: x[1], reverse=True)

        # Create a dictionary with the concept of interest as the key
        concept_co_occurrences_normalized = {concept_of_interest: sorted_co_occurrence_pairs}

        # Load existing data from the JSON file if it exists
        try:
            with open('object_co_occurrences_normalized.json', 'r') as json_file:
                data = json.load(json_file)
        except FileNotFoundError:
            data = {}

        # Update the data with the new co-occurrence information
        data.update(concept_co_occurrences_normalized)

        # Save the updated data to the JSON file
        with open('object_co_occurrences_normalized.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

        return concept_co_occurrences_normalized

    def get_co_occurrence_triples(concept_of_interest, concept_detected_objects, consider_person):
        # Flatten the list of lists to get all detected object names
        all_object_names = [obj_name for sublist in concept_detected_objects for obj_name in sublist]

        # If consider_person is False, remove 'person' from the object names
        if not consider_person:
            all_object_names = [obj_name for obj_name in all_object_names if obj_name != 'person']

        # Count the total number of images for the concept
        total_images = len(concept_detected_objects)

        # Initialize an empty dictionary to store co-occurrence percentages
        co_occurrence_percentages = {}

        # Populate the co-occurrence percentages based on the flattened list
        for detected_objects in concept_detected_objects:
            if not consider_person:
                detected_objects = [obj_name for obj_name in detected_objects if obj_name != 'person']

            # Generate combinations of three object names
            triples = combinations(detected_objects, 3)

            for triple in triples:
                # Sort the objects in the triple alphabetically to ensure consistent ordering
                sorted_triple = tuple(sorted(triple))

                # Increase the count for the co-occurrence triple
                co_occurrence_percentages[sorted_triple] = co_occurrence_percentages.get(sorted_triple, 0) + 1

        # Convert counts to percentages by dividing by the total number of images
        for triple, count in co_occurrence_percentages.items():
            co_occurrence_percentages[triple] = round(count / total_images * 100, 2)

        # Sort the co-occurrence triples by percentage in descending order
        sorted_co_occurrence_triples = sorted(co_occurrence_percentages.items(), key=lambda x: x[1], reverse=True)

        # Create a dictionary with the concept of interest as the key
        concept_co_occurrence_triples_normalized = {concept_of_interest: sorted_co_occurrence_triples}

        # Load existing data from the JSON file if it exists
        try:
            with open('object_co_occurrence_triples_normalized.json', 'r') as json_file:
                data = json.load(json_file)
        except FileNotFoundError:
            data = {}

        # Update the data with the new co-occurrence triple information
        data.update(concept_co_occurrence_triples_normalized)

        # Save the updated data to the JSON file
        with open('object_co_occurrence_triples_normalized.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

        return concept_co_occurrence_triples_normalized

    def get_co_occurrence_pairs(concept_of_interest, concept_detected_objects, consider_person):
        # Flatten the list of lists to get all detected object names
        all_object_names = [obj_name for sublist in concept_detected_objects for obj_name in sublist]

        # If consider_person is False, remove 'person' from the object names
        if not consider_person:
            all_object_names = [obj_name for obj_name in all_object_names if obj_name != 'person']

        # Count the total number of images related to the concept
        total_images = len(concept_detected_objects)

        # Initialize an empty dictionary to count co-occurrences
        co_occurrence_counts = {}

        # Populate the co-occurrence counts based on the flattened list
        for detected_objects in concept_detected_objects:
            if not consider_person:
                detected_objects = [obj_name for obj_name in detected_objects if obj_name != 'person']
            for obj_name in detected_objects:
                for other_obj_name in detected_objects:
                    if obj_name != other_obj_name:
                        # Create a tuple to represent the co-occurrence quadruple
                        co_occurrence_pair = tuple(sorted([obj_name, other_obj_name]))

                        # Increase the count for the co-occurrence quadruple
                        co_occurrence_counts[co_occurrence_pair] = co_occurrence_counts.get(
                            co_occurrence_pair, 0) + 1

        # Calculate the percentage of each co-occurrence quadruple
        co_occurrence_percentages = {
            pair: (count / total_images) * 100
            for pair, count in co_occurrence_counts.items()
        }

        # Sort the co-occurrence quadruples by percentage in descending order
        sorted_co_occurrence_pairs = sorted(co_occurrence_percentages.items(), key=lambda x: x[1], reverse=True)

        # # Print the sorted co-occurrence quadruples
        # print(f'Co-occurrence pairs for {concept_of_interest}')
        # for quadruple, percentage in sorted_co_occurrence_pairs:
        #     print(f"Pairs: {quadruple}, Percentage: {percentage:.2f}%")


        co_occurrence_dict = {}
        for quadruple, percentage in sorted_co_occurrence_pairs:
            co_occurrence_dict["_".join(quadruple)] = percentage

        # Load existing data from the JSON file if it exists
        try:
            with open('object_co_occurrences_pairs.json', 'r') as json_file:
                data = json.load(json_file)
        except FileNotFoundError:
            data = {}

        # Update the data with the new co-occurrence information
        data.update(co_occurrence_dict)

        # Save the updated data to the JSON file
        with open('object_co_occurrences_pairs.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

        return sorted_co_occurrence_pairs

    def get_co_occurrence_quadruples(concept_of_interest, concept_detected_objects, consider_person):
        # Flatten the list of lists to get all detected object names
        all_object_names = [obj_name for sublist in concept_detected_objects for obj_name in sublist]

        # If consider_person is False, remove 'person' from the object names
        if not consider_person:
            all_object_names = [obj_name for obj_name in all_object_names if obj_name != 'person']

        # Count the total number of images related to the concept
        total_images = len(concept_detected_objects)

        # Initialize an empty dictionary to count co-occurrences
        co_occurrence_counts = {}

        # Populate the co-occurrence counts based on the flattened list
        for detected_objects in concept_detected_objects:
            if not consider_person:
                detected_objects = [obj_name for obj_name in detected_objects if obj_name != 'person']

            # Generate all unique quadruples from detected objects
            unique_quadruples = set(itertools.combinations(sorted(detected_objects), 4))

            for quadruple in unique_quadruples:
                # Increase the count for the co-occurrence quadruple
                co_occurrence_counts[quadruple] = co_occurrence_counts.get(quadruple, 0) + 1

        # Calculate the percentage of each co-occurrence quadruple
        co_occurrence_percentages = {
            quadruple: (count / total_images) * 100
            for quadruple, count in co_occurrence_counts.items()
        }

        # Sort the co-occurrence quadruples by percentage in descending order
        sorted_co_occurrence_quadruples = sorted(co_occurrence_percentages.items(), key=lambda x: x[1], reverse=True)
        # Print the sorted co-occurrence quadruples
        print(f'Co-occurrence quadruples for {concept_of_interest}')
        for quadruple, percentage in sorted_co_occurrence_quadruples:
            print(f"Quadruple: {quadruple}, Percentage: {percentage:.2f}%")




        co_occurrence_dict = {}
        for quadruple, percentage in sorted_co_occurrence_quadruples:
            co_occurrence_dict["_".join(quadruple)] = percentage

        # Load existing data from the JSON file if it exists
        try:
            with open('object_co_occurrences_quadruples.json', 'r') as json_file:
                data = json.load(json_file)
        except FileNotFoundError:
            data = {}

        # Update the data with the new co-occurrence information
        data.update(co_occurrence_dict)

        # Save the updated data to the JSON file
        with open('object_co_occurrences_quadruples.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

        return sorted_co_occurrence_quadruples

    def create_heatmap(concept_of_interest, co_occurrence_matrix, object_names):
        # Create a heatmap using seaborn
        plt.figure(figsize=(12, 10))
        sns.heatmap(co_occurrence_matrix, annot=False, fmt='d', cmap="YlGnBu", xticklabels=object_names,
                    yticklabels=object_names, cbar=True, cbar_kws={"label": "Co-occurrence count"})
        plt.title(f"Co-occurrence Heatmap for Concept: {concept_of_interest}")
        plt.xlabel("Detected Objects")
        plt.ylabel("Detected Objects")
        plt.show()
        # Save the plot as an image
        save_filename = f"output_imgs/perceptual_data/detected_objects/co_occurrence_heatmaps/{concept_of_interest}_{ACs_list_name}_object_cooccurr_heatmap.jpg"
        plt.savefig(save_filename)
        plt.show()

    def set_occurrence_heatmaps(ACs_list_name, concept_of_interest, consider_person):
        concept_detected_objects = []
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        for img_id, image_info in merged_ARTstract.items():
            concept_name = None
            detected_objects = None

            # Find the concept_name and detected_objects dynamically
            for key, value in image_info['evoked_clusters'].items():
                if value.get('cluster_name') == concept_of_interest:
                    concept_name = value['cluster_name']
                    detected_objects_list = image_info['od'].get("ARTstract_od_2023_06_28", {}).get("detected_objects",
                                                                                                    [])
                    detected_objects = [detected_object["detected_object"] for detected_object in detected_objects_list]

            if concept_name and detected_objects:
                concept_detected_objects.append(detected_objects)

        # co_occurrence_matrix, object_names = create_co_occurrence_matrix(concept_detected_objects, consider_person)
        # create_heatmap(concept_of_interest, co_occurrence_matrix, object_names)

        print("co occurrence for ", concept_of_interest)
        get_co_occurrence_pairs(concept_of_interest, concept_detected_objects, consider_person)
        # get_co_occurrence_triples(concept_of_interest, concept_detected_objects, consider_person)
        # get_co_occurrence_quadruples(concept_of_interest, concept_detected_objects, consider_person)

    concepts_of_interest = ['comfort', 'danger', 'death', 'fitness', 'freedom', 'power', 'safety']
    for concept_of_interest in concepts_of_interest:
        for ACs_list_name in ACs_list_names:
            set_occurrence_heatmaps(ACs_list_name, concept_of_interest, consider_person)
    return

## Image captions
def stats_image_captions(ACs_list_names):
    def get_image_captions_by_concept(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        # for concept, list in concept_images.items():
        #     print(concept, "has these many images ", len(list))
        image_captions_by_concept = {concept: [] for concept in concept_images}
        for concept, images in concept_images.items():
            for img in images:
                for image_id, image_info in merged_ARTstract.items():
                    if img == image_id:
                        caption_dict = merged_ARTstract[img].get("ic", {}).get("ARTstract_ic_2023_06_28", {})
                        # print(caption_dict)
                        caption = caption_dict['image_description']
                        # print(caption)
                        image_captions_by_concept[concept].append(caption)
        # print(image_captions_by_concept)
        return image_captions_by_concept
    def get_captions_text_by_concept(ACs_list_name):
        image_captions_by_concept = get_image_captions_by_concept(ACs_list_name)
        captions_text_by_concept = {concept: [] for concept in image_captions_by_concept}
        for concept, captions_list in image_captions_by_concept.items():
            delimiter = " "
            caption_text = delimiter.join(captions_list)
            # print(caption_text)
            captions_text_by_concept[concept] = caption_text
        #print(captions_text_by_concept)
        return captions_text_by_concept
    def calculate_caption_words_frequencies(ACs_list_name):
        # Get the set of English stopwords from NLTK
        stop_words = set(stopwords.words('english'))

        # Initialize the Porter Stemmer from NLTK
        lemmatizer = WordNetLemmatizer()

        captions_text_by_concept = get_captions_text_by_concept(ACs_list_name)
        word_frequencies_by_concept = {}

        for concept, caption_text in captions_text_by_concept.items():
            # Tokenize the caption_text into words using nltk word_tokenize
            words = nltk.word_tokenize(caption_text)

            # Convert all words to lowercase to make the comparison case-insensitive
            words = [word.lower() for word in words]

            # Filter out the stop words from the list of words
            words = [word for word in words if word not in stop_words]

            # Stem each word to get its root form
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

            # Calculate word frequencies using Counter on the stemmed words
            word_frequencies = Counter(lemmatized_words)
            word_frequencies_by_concept[concept] = word_frequencies

        # print(word_frequencies_by_concept)
        return word_frequencies_by_concept
    def plot_caption_words_frequencies(ACs_list_name):
        word_frequencies_by_concept = calculate_caption_words_frequencies(ACs_list_name)
        num_concepts = len(word_frequencies_by_concept)
        fig, axs = plt.subplots(num_concepts + 1, 1, figsize=(12, 5 * (num_concepts + 1)), sharex=True)

        # Calculate top 30 objects across all concepts, excluding the top object "person"
        all_word_frequencies = {}
        for concept, word_frequencies in word_frequencies_by_concept.items():
            for word, freq in word_frequencies.items():
                 all_word_frequencies[word] = all_word_frequencies.get(word, 0) + freq

        sorted_all_frequencies = sorted(all_word_frequencies.items(), key=lambda x: x[1], reverse=True)
        top_all_words, top_all_frequencies = zip(*sorted_all_frequencies[:30])

        # Plot top 30 objects across all concepts, excluding the top object "person"
        axs[0].bar(top_all_words, top_all_frequencies)
        axs[0].set_title("Top 30 Words Across All Concept Image Captions")
        axs[0].tick_params(axis='x', rotation=45, labelsize=8)

        for i, (concept, word_frequencies) in enumerate(word_frequencies_by_concept.items(), start=1):
            sorted_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
            words, frequencies = zip(
                *sorted_frequencies[1:31])  # Taking top 30 objects, excluding the first object "person"
            axs[i].bar(words, frequencies)
            axs[i].set_title(f"Top Caption Words for Concept '{concept}'")
            axs[i].tick_params(axis='x', rotation=45, labelsize=8)

        plt.tight_layout()
        plt.show()

        # Save the plot as an image
        save_filename = f"output_imgs/perceptual_data/image_captions/top_30_caption_words_{ACs_list_name}.jpg"
        plt.savefig(save_filename)

        plt.show()
    def find_common_words(ACs_list_name):
        word_frequencies_by_concept = calculate_caption_words_frequencies(ACs_list_name)

        # Calculate overall frequency of each object across all concepts
        all_word_frequencies = {}
        for word_frequencies in word_frequencies_by_concept.values():
            for word, freq in word_frequencies.items():
                all_word_frequencies[word] = all_word_frequencies.get(word, 0) + freq

        # Sort objects based on their overall frequency in descending order
        sorted_overall_frequencies = sorted(all_word_frequencies.items(), key=lambda x: x[1], reverse=True)
        ordered_words = [word for word, freq in sorted_overall_frequencies]

        # Get the set of common objects that are present in every concept
        common_words = set.intersection(
            *[set(frequencies.keys()) for frequencies in word_frequencies_by_concept.values()])

        # Return the ordered set of common objects
        common_ordered_words = [obj for obj in ordered_words if obj in common_words]
        print("for all concepts, the most common caption words are (from most common to least: ", common_ordered_words)
        return common_ordered_words
    def find_unique_caption_words(ACs_list_name):
        word_frequencies_by_concept = calculate_caption_words_frequencies(ACs_list_name)

        unique_words_by_concept = {concept: set(frequencies.keys()) for concept, frequencies in
                                   word_frequencies_by_concept.items()}

        for concept, frequencies in word_frequencies_by_concept.items():
            for other_concept, other_frequencies in word_frequencies_by_concept.items():
                if concept != other_concept:
                    unique_words_by_concept[concept] -= set(other_frequencies.keys())

        for concept, word_set in unique_words_by_concept.items():
            if len(word_set) == 0:
                print(concept, "does not have any unique caption words")
            else:
                print(concept, "has the following unique caption words:", word_set)

        return unique_words_by_concept
    def find_relevant_caption_words_by_concept(ACs_list_name):
        word_frequencies_by_concept = calculate_caption_words_frequencies(ACs_list_name)
        # Calculate overall frequency of each object across all concepts
        all_word_frequencies = Counter()
        for word_frequencies in word_frequencies_by_concept.values():
            all_word_frequencies.update(word_frequencies)

        relevant_caption_words_by_concept = {}

        for concept, word_frequencies in word_frequencies_by_concept.items():
            # Calculate relative frequency (TF) for each object within the concept
            relative_frequencies = {
                obj: freq / all_word_frequencies[obj]
                for obj, freq in word_frequencies.items()
            }

            # Calculate inverse concept frequency (IDF) for each object
            num_concepts = len(word_frequencies_by_concept)
            inverse_concept_frequency = {word: num_concepts / sum(
                1 for concept_freqs in word_frequencies_by_concept.values() if word in concept_freqs)
                                         for word in word_frequencies}

            # Calculate relevance score for each object in the concept (TF * IDF)
            relevance_scores = {word: round(relative_frequencies[word] * inverse_concept_frequency[word], 3)
                                for word in word_frequencies}

            # Sort objects based on their relevance scores in descending order
            sorted_relevance_scores = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

            top_words = [word for word, score in sorted_relevance_scores[:15]]

            # Check if each object appears in only the current concept's top 10 list
            unique_top_words = []
            for word in top_words:
                unique_to_current_concept = True
                for other_concept, other_top_words in relevant_caption_words_by_concept.items():
                    if other_concept != concept and word in other_top_words:
                        unique_to_current_concept = False
                        break
                if unique_to_current_concept:
                    unique_top_words.append(word)

            relevant_caption_words_by_concept[concept] = unique_top_words
        #print(relevant_words_by_concept)
        # for concept, word_set in relevant_caption_words_by_concept.items():
        #     print(concept, "has relevant concepts: ", word_set)

        return relevant_caption_words_by_concept
    def find_top_caption_words_by_concept(ACs_list_name, k=15):
        word_frequencies_by_concept = calculate_caption_words_frequencies(ACs_list_name)
        top_caption_words_by_concept = {}

        for concept, frequencies in word_frequencies_by_concept.items():
            total_detected_words = sum(frequencies.values())
            words_percentages = {
                obj: (count / total_detected_words) * 100 for obj, count in frequencies.items()
            }
            sorted_words = sorted(words_percentages.items(), key=lambda x: x[1], reverse=True)
            top_words = dict(sorted_words[:k])
            top_caption_words_by_concept[concept] = top_words
        # print("Top words by concept: ", top_caption_words_by_concept)
        ordered_lists_by_concept = {}
        for concept, word_scores in top_caption_words_by_concept.items():
            # Sort objects based on their scores in descending order
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            # Extract the objects from the sorted list
            ordered_words = [obj for obj, score in sorted_words]
            ordered_lists_by_concept[concept] = ordered_words
        # for concept, word_set in ordered_lists_by_concept.items():
        #     print(concept, "has top concepts: ", word_set)
        return top_caption_words_by_concept
    def find_top_relevant_caption_words_by_concept(ACs_list_name, k=15):
        top_caption_words_by_concept = find_top_caption_words_by_concept(ACs_list_name)
        relevant_caption_words_by_concept = find_relevant_caption_words_by_concept(ACs_list_name)
        top_relevant_caption_words_by_concept = {}

        for concept, top_words in top_caption_words_by_concept.items():
            # Get the relevant concepts for the current concept
            relevant_words = relevant_caption_words_by_concept.get(concept, [])
            # Calculate the relevance scores for the relevant concepts
            relevance_scores = {obj: 1 for obj in relevant_words}
            # Convert top_concepts dictionary to a list of tuples (object, score)
            top_words_list = list(top_words.items())
            # Sort the top_concepts_list based on scores in descending order
            top_words_list.sort(key=lambda x: x[1], reverse=True)
            # Extract the objects from the sorted list
            top_words_sorted = [obj for obj, _ in top_words_list]
            # Ensure all objects in top_concepts_sorted have scores in relevance_scores dictionary
            relevance_scores.update({obj: 0 for obj in top_words_sorted if obj not in relevance_scores})
            # Calculate the Jaccard similarity between the top concepts and relevant concepts
            jaccard_scores = {obj: len(set([obj]).intersection(set(top_words_sorted))) / len(
                set([obj]).union(set(top_words_sorted))) for obj in relevant_words}
            # Sort objects based on Jaccard similarity scores in descending order
            sorted_words = sorted(jaccard_scores.items(), key=lambda x: x[1], reverse=True)
            # Take only the top k relevant concepts
            top_relevant_caption_words_by_concept[concept] = [obj for obj, _ in sorted_words[:k]]
            # Print the top relevant concepts
        # for concept, concepts in top_relevant_caption_words_by_concept.items():
        #     print(concept, "has top relevant concepts:", concepts)
        print(top_relevant_caption_words_by_concept)

        file_path = "top_relevant_jsons/top_relevant_caption_words_by_concept.json"
        # Using json.dump() to write the dictionary to a JSON file
        with open(file_path, "w") as json_file:
            json.dump(top_relevant_caption_words_by_concept, json_file)

        return top_relevant_caption_words_by_concept

    def find_top_relevant_objects_by_concept_w_freqs(ACs_list_name):
        top_relevant_caption_words_by_concept = find_top_relevant_caption_words_by_concept(ACs_list_name)
        word_frequencies_by_concept = calculate_caption_words_frequencies(ACs_list_name)

        top_relevant_caption_words_by_concept_w_freqs = {}

        for concept, top_relevant_words in top_relevant_caption_words_by_concept.items():
            # Get the object frequencies for the current concept
            word_frequencies = word_frequencies_by_concept.get(concept, {})

            # Fetch frequencies for top relevant objects and create a dictionary {object: frequency}
            words_with_freqs = {obj: word_frequencies.get(obj, 0) for obj in top_relevant_words}

            # Store the dictionary {object: frequency} for the current concept
            top_relevant_caption_words_by_concept_w_freqs[concept] = words_with_freqs

        # Print the top relevant concepts and their frequencies
        # for concept, words in top_relevant_caption_words_by_concept_w_freqs.items():
        #     print(concept, "has top relevant caption words and frequencies:", words)

        return top_relevant_caption_words_by_concept_w_freqs
    def create_caption_words_wordclouds(ACs_list_name):
        top_caption_words_by_concept = find_top_caption_words_by_concept(ACs_list_name)
        top_relevant_caption_words_by_concept_w_freqs = find_top_relevant_objects_by_concept_w_freqs(ACs_list_name)
        class GroupedColorFunc(object):
            def __init__(self, color_to_words, default_color):
                self.color_func_to_words = [
                    (get_single_color_func(color), set(words))
                    for (color, words) in color_to_words.items()]

                self.default_color_func = get_single_color_func(default_color)

            def get_color_func(self, word):
                """Returns a single_color_func associated with the word"""
                try:
                    color_func = next(
                        color_func for (color_func, words) in self.color_func_to_words
                        if word in words)
                except StopIteration:
                    color_func = self.default_color_func

                return color_func

            def __call__(self, word, **kwargs):
                return self.get_color_func(word)(word, **kwargs)

        font_color = '#0074D9'  # Use any shade of blue you prefer
        helvetica_font = 'Helvetica.ttf'  # Replace with the path to your Helvetica font file

        # Remove "person" from top_objects_by_concept and top_relevant_objects_by_concept
        for top_words in top_caption_words_by_concept.values():
            top_words.pop('person', None)

        for top_relevant_words in top_relevant_caption_words_by_concept_w_freqs.values():
            if 'person' in top_relevant_words:
                top_relevant_words.remove('person')

        # Generate word clouds for each concept
        for concept, top_words in top_caption_words_by_concept.items():
            # Create word cloud objects
            wc_top_words = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,  # Set the font to Helvetica
                color_func=lambda *args, **kwargs: font_color  # Set all words to blue color
            ).generate_from_frequencies(top_words)

            # Get the top relevant objects for the current concept
            top_relevant_words = top_relevant_caption_words_by_concept_w_freqs.get(concept, {})
            wc_top_relevant_words = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,  # Set the font to Helvetica
                color_func=lambda *args, **kwargs: font_color  # Set all words to blue color
            ).generate_from_frequencies(top_relevant_words)

            # Plot the word clouds side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(wc_top_words, interpolation="bilinear")
            axes[0].axis("off")
            axes[0].set_title(f"Top Caption Words for Concept: {concept}")

            axes[1].imshow(wc_top_relevant_words, interpolation="bilinear")
            axes[1].axis("off")
            axes[1].set_title(f"Top Relevant Caption Words for Concept: {concept}")

            plt.show()
            # Save the plot as an image
            save_filename = f"output_imgs/perceptual_data/image_captions/wordclouds/caption_words_{concept}_{ACs_list_name}_wordcloud.jpg"
            plt.savefig(save_filename)
            plt.show()

        return

    for ACs_list_name in ACs_list_names:
        ## EXECUTION
        # get_image_captions_by_concept(ACs_list_name)
        # get_captions_text_by_concept(ACs_list_name)
        # calculate_caption_words_frequencies(ACs_list_name)
        # plot_caption_words_frequencies(ACs_list_name)
        # find_common_words(ACs_list_name)
        # find_unique_caption_words(ACs_list_name)
        # find_relevant_caption_words_by_concept(ACs_list_name)
        # find_top_caption_words_by_concept(ACs_list_name)
        find_top_relevant_caption_words_by_concept(ACs_list_name)
        # find_top_relevant_objects_by_concept_w_freqs(ACs_list_name)
        # create_caption_words_wordclouds(ACs_list_name)
    return

def caption_words_co_occurences(ACs_list_names, consider_painting):

    def get_concept_caption_words(ACs_list_name, concept_of_interest, consider_painting):
        concept_caption_words = []
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        # Get the set of English stopwords from NLTK
        stop_words = set(stopwords.words('english'))
        # Initialize the Porter Stemmer from NLTK
        lemmatizer = WordNetLemmatizer()

        for img_id, image_info in merged_ARTstract.items():
            concept_name = None
            lemmatized_words = None

            # Find the concept_name and detected_objects dynamically
            for key, value in image_info['evoked_clusters'].items():
                if value.get('cluster_name') == concept_of_interest:
                    concept_name = value['cluster_name']

                    caption_dict = image_info['ic'].get("ARTstract_ic_2023_06_28", {})
                    caption = caption_dict['image_description']
                    caption_words = nltk.word_tokenize(caption)
                    words = [word.lower() for word in caption_words]
                    # Filter out the stop words from the list of words
                    words = [word for word in words if word not in stop_words]
                    # Stem each word to get its root form
                    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            if concept_name and lemmatized_words:
                concept_caption_words.append(lemmatized_words)

        # Flatten the list of lists to get all detected object names
        all_lemmatized_words = [lemmatized_word for sublist in concept_caption_words for lemmatized_word in sublist]

        # If consider_person is False, remove 'person' from the object names
        if not consider_painting:
            all_lemmatized_words = [lemmatized_word for lemmatized_word in all_lemmatized_words if
                                    lemmatized_word != 'painting']

        # Calculate word frequencies using Counter to get the top 60 lemmatized words
        word_frequencies = Counter(all_lemmatized_words)
        top_lemmatized_words = [word for word, _ in word_frequencies.most_common(60)]

        # Extract unique object names from the top 60 lemmatized words and sort alphabetically
        lemmatized_words = sorted(list(set(top_lemmatized_words)))
        # print('lemmatized words is a list with length', len(lemmatized_words))

        return concept_caption_words, lemmatized_words

    def create_co_occurrence_matrix(ACs_list_name, concept_of_interest, consider_painting):
        concept_caption_words, lemmatized_words = get_concept_caption_words(ACs_list_name, concept_of_interest, consider_painting)

        # Initialize an empty co-occurrence matrix
        num_lemmatized_words = len(lemmatized_words)
        co_occurrence_matrix = np.zeros((num_lemmatized_words, num_lemmatized_words), dtype=int)
        # Create a dictionary to map object names to matrix indices
        word_to_index = {word: index for index, word in enumerate(lemmatized_words)}

        # Populate the co-occurrence matrix based on the flattened list
        for img_caption_words in concept_caption_words:
            if not consider_painting:
                img_caption_words = [word for word in img_caption_words if word != 'painting']
            for word in img_caption_words:
                for other_word in img_caption_words:
                    if word != other_word:
                        # Check if the words exist in the word_to_index dictionary
                        if word in word_to_index and other_word in word_to_index:
                            # Increase the count for co-occurrence of word and other_word
                            i, j = word_to_index[word], word_to_index[other_word]
                            co_occurrence_matrix[i, j] += 1
        return co_occurrence_matrix, lemmatized_words

    def create_co_occurrence_heatmap(ACs_list_name, concept_of_interest, consider_painting):
        co_occurrence_matrix, lemmatized_words = create_co_occurrence_matrix(ACs_list_name, concept_of_interest, consider_painting)
        # Create a heatmap using seaborn
        plt.figure(figsize=(12, 10))
        sns.heatmap(co_occurrence_matrix, annot=False, fmt='d', cmap="YlGnBu", xticklabels=lemmatized_words,
                    yticklabels=lemmatized_words, cbar=True, cbar_kws={"label": "Co-occurrence count"})
        plt.title(f"Co-occurrence Heatmap for Concept: {concept_of_interest} in {ACs_list_name}")
        plt.xlabel("Caption Word")
        plt.ylabel("Caption Word")
        plt.show()
        plt.show()
        # Save the plot as an image
        save_filename = f"output_imgs/perceptual_data/image_captions/co_occurrence_heatmaps/{concept_of_interest}_{ACs_list_name}_caption_cooccurr_heatmap.jpg"
        plt.savefig(save_filename)
        plt.show()
        return

    def get_top_co_occurrences(ACs_list_name, concept_of_interest, consider_painting, top_n=60):
        concept_caption_words, lemmatized_words = get_concept_caption_words(ACs_list_name, concept_of_interest, consider_painting)
        # Create a dictionary to map object names to matrix indices
        word_to_index = {word: index for index, word in enumerate(lemmatized_words)}

        # Calculate the co-occurrence counts for each word pair
        co_occurrence_counts = {}
        for img_caption_words in concept_caption_words:
            if not consider_painting:
                img_caption_words = [word for word in img_caption_words if word != 'painting']
            for word in img_caption_words:
                for other_word in img_caption_words:
                    if word != other_word:
                        # Check if the words exist in the word_to_index dictionary
                        if word in word_to_index and other_word in word_to_index:
                            i, j = word_to_index[word], word_to_index[other_word]
                            word_pair = tuple(sorted([word, other_word]))
                            co_occurrence_counts[word_pair] = co_occurrence_counts.get(word_pair, 0) + 1

        # Sort the co-occurrence counts by the counts in descending order
        sorted_co_occurrences = sorted(co_occurrence_counts.items(), key=lambda x: x[1], reverse=True)
        top_sorted_co_occurrences = sorted_co_occurrences[:top_n]
        # print(f"Top {top_n} caption words co-occurrences for concept {concept_of_interest}:")
        # for (word1, word2), count in top_sorted_co_occurrences:
        #     print(f"{word1} - {word2}: {count}")
        # Return the top n co-occurrences
        return top_sorted_co_occurrences

    def get_relevant_cooccurrences(ACs_list_name, concepts_of_interest, consider_painting, top_n=60):
        top_co_occurrences_by_concept = {}
        for concept_of_interest in concepts_of_interest:
            top_co_occurrences = get_top_co_occurrences(ACs_list_name, concept_of_interest, consider_painting, top_n=60)
            top_co_occurrences_by_concept[concept_of_interest] = top_co_occurrences

        co_occurrence_relevance_by_concept = {}
        seen_co_occurrences = set()

        for concept_of_interest, top_co_occurrences in top_co_occurrences_by_concept.items():
            relevant_co_occurrences = []
            for (word1, word2), count in top_co_occurrences:
                co_occurrence_pair = (word1, word2)
                reverse_co_occurrence_pair = (word2, word1)

                if co_occurrence_pair not in seen_co_occurrences and reverse_co_occurrence_pair not in seen_co_occurrences:
                    relevant_co_occurrences.append(co_occurrence_pair)
                    seen_co_occurrences.add(co_occurrence_pair)

            co_occurrence_relevance_by_concept[concept_of_interest] = relevant_co_occurrences

        def print_relevant_cooccurrences(co_occurrence_relevance_by_concept):
            for concept_of_interest, relevant_co_occurrences in co_occurrence_relevance_by_concept.items():
                print(f"Concept: {concept_of_interest}")
                for word1, word2 in relevant_co_occurrences:
                    print(f"   - {word1}, {word2}")
                print()  # Empty line for separating concepts


        # Assuming you have already calculated and stored the co_occurrence_relevance_by_concept
        print_relevant_cooccurrences(co_occurrence_relevance_by_concept)
        return co_occurrence_relevance_by_concept

    def create_caption_cooccurrence_wordclouds(ACs_list_name, concepts_of_interest, consider_painting, top_n=60):
        co_occurrence_relevance_by_concept = get_relevant_cooccurrences(ACs_list_name, concepts_of_interest, consider_painting, top_n=60)
        for concept_of_interest, relevant_co_occurrences in co_occurrence_relevance_by_concept.items():
            words = [f"{word1}, {word2}" for word1, word2 in relevant_co_occurrences]
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Relevant Co-occurrence Pairs for Concept: {concept_of_interest}')
            plt.show()
            plt.show()
            # Save the plot as an image
            save_filename = f"output_imgs/perceptual_data/image_captions/wordclouds/{concept_of_interest}_{ACs_list_name}_caption_cooccurr_wordcloud.jpg"
            plt.savefig(save_filename)
            plt.show()

    # concepts_of_interest = ['safety']
    concepts_of_interest = ['comfort', 'danger', 'death', 'fitness', 'freedom', 'power', 'safety']
    for ACs_list_name in ACs_list_names:
        # get_relevant_cooccurrences(ACs_list_name, concepts_of_interest, consider_painting, top_n=30)
        create_caption_cooccurrence_wordclouds(ACs_list_name, concepts_of_interest, consider_painting, top_n=60)
        for concept_of_interest in concepts_of_interest:
           create_co_occurrence_heatmap(ACs_list_name, concept_of_interest, consider_painting)
    return

## Top colors
def stats_top_colors(ACs_list_names, filter_grays_out):
    def get_top_colors_by_concept(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        # for concept, list in concept_images.items():
            # print(concept, "has these many images ", len(list))
        top_colors_by_concept = {concept: [] for concept in concept_images}
        for concept, images in concept_images.items():
            for img in images:
                for image_id, image_info in merged_ARTstract.items():
                    if img == image_id:
                        top_colors_list = merged_ARTstract[img].get("color", {}).get("ARTstract_color_2023_06_26", [])
                        top_color_webcolor = [color_object["webcolor_name"] for color_object in
                                            top_colors_list]
                        top_colors_by_concept[concept].extend(top_color_webcolor)
        return top_colors_by_concept

    def calculate_colors_frequencies(ACs_list_name):
        top_colors_by_concept = get_top_colors_by_concept(ACs_list_name)
        color_frequencies_by_concept = {}
        all_detected_colors = []

        for concept, top_colors in top_colors_by_concept.items():
            color_frequencies = Counter(top_colors)
            color_frequencies_by_concept[concept] = color_frequencies
            all_detected_colors.extend(top_colors)

        all_color_frequencies = Counter(all_detected_colors)
        return color_frequencies_by_concept, all_color_frequencies

    def plot_colors_frequencies(ACs_list_name):
        color_frequencies_by_concept, all_color_frequencies = calculate_colors_frequencies(ACs_list_name)
        num_concepts = len(color_frequencies_by_concept)
        fig, axs = plt.subplots(num_concepts + 1, 1, figsize=(12, 5 * (num_concepts + 1)), sharex=True)

        # Calculate top 30 colors across all concepts, excluding the top colors with "gray" in it
        all_color_frequencies = {}
        for concept, color_frequencies in color_frequencies_by_concept.items():
            for color, freq in color_frequencies.items():
                if "gray" not in color:
                    all_color_frequencies[color] = all_color_frequencies.get(color, 0) + freq

        sorted_all_frequencies = sorted(all_color_frequencies.items(), key=lambda x: x[1], reverse=True)
        top_all_colors, top_all_frequencies = zip(*sorted_all_frequencies[:30])

        # Plot top 30 objects across all concepts, excluding the top colors with "gray" in it
        axs[0].bar(top_all_colors, top_all_frequencies)
        axs[0].set_title("Top 30 Colors Across All Concepts (excluding 'gray's)")
        axs[0].tick_params(axis='x', rotation=45, labelsize=8)

        for i, (concept, color_frequencies) in enumerate(color_frequencies_by_concept.items(), start=1):
            # Filter out colors containing "gray" in their names
            filtered_color_frequencies = {color: freq for color, freq in color_frequencies.items() if
                                          "gray" not in color.lower()}

            sorted_frequencies = sorted(filtered_color_frequencies.items(), key=lambda x: x[1], reverse=True)
            colors, frequencies = zip(
                *sorted_frequencies[:30])  # Taking top 30 objects, excluding the first color "gre"

            axs[i].bar(colors, frequencies)
            axs[i].set_title(f"Top Colors for Concept '{concept}'")
            axs[i].tick_params(axis='x', rotation=45, labelsize=8)

        plt.tight_layout()
        plt.show()

        # Save the plot as an image
        save_filename = f"output_imgs/perceptual_data/color/top_30_{ACs_list_name}.jpg"
        plt.savefig(save_filename)
        plt.show()

    def plot_colors_frequencies_percentages(ACs_list_name, filter_grays_out):
        color_frequencies_by_concept, all_color_frequencies = calculate_colors_frequencies(ACs_list_name)
        num_concepts = len(color_frequencies_by_concept)
        fig, axs = plt.subplots(num_concepts + 1, 1, figsize=(12, 5 * (num_concepts + 1)), sharex=True)

        # Calculate top 30 colors across all concepts, excluding the top colors with "gray" in it
        all_color_frequencies = {}
        for concept, color_frequencies in color_frequencies_by_concept.items():
            for color, freq in color_frequencies.items():
                if filter_grays_out == True:
                    if "gray" not in color:
                        all_color_frequencies[color] = all_color_frequencies.get(color, 0) + freq
                else:
                    all_color_frequencies[color] = all_color_frequencies.get(color, 0) + freq

        sorted_all_frequencies = sorted(all_color_frequencies.items(), key=lambda x: x[1], reverse=True)
        top_all_colors, top_all_frequencies = zip(*sorted_all_frequencies[:30])

        max_y_value = 0.18  # Set the maximum y-axis value to 1.0 (100%)

        # Calculate the total number of images for calculating percentages
        total_images = sum(all_color_frequencies.values())

        # Plot the total color frequencies subplot
        frequencies_total = [all_color_frequencies.get(color, 0) / total_images for color in top_all_colors]
        axs[0].bar(top_all_colors, frequencies_total)
        if filter_grays_out == True:
            axs[0].set_title("Overall Color Frequencies (excluding 'gray' colors)")
        else:
            axs[0].set_title("Overall Color Frequencies")
        axs[0].set_ylabel("Percentage of Images")
        axs[0].tick_params(axis='x', rotation=45, labelsize=8)
        axs[0].set_ylim(0, max_y_value)  # Set consistent y-axis limits

        # Plot the color frequencies for each concept subplot
        sorted_concepts = sorted(color_frequencies_by_concept.keys())  # Sort concepts alphabetically
        for i, concept in enumerate(sorted_concepts, start=1):
            color_frequencies = color_frequencies_by_concept[concept]
            # Filter out colors containing "gray" in their names
            if filter_grays_out == True:
                filtered_color_frequencies = {color: freq for color, freq in color_frequencies.items() if
                                              "gray" not in color.lower()}
                sorted_frequencies = sorted(filtered_color_frequencies.items(), key=lambda x: x[1], reverse=True)
            else:
                filtered_color_frequencies = {color: freq for color, freq in color_frequencies.items()}
                sorted_frequencies = sorted(filtered_color_frequencies.items(), key=lambda x: x[1], reverse=True)

            colors, frequencies = zip(*sorted_frequencies[:30])

            # Calculate the percentage of images for the current concept that have each color
            total_images_concept = sum(color_frequencies.values())
            frequencies_percentage = [freq / total_images_concept for freq in frequencies]

            axs[i].bar(colors, frequencies_percentage)
            if filter_grays_out == True:
                axs[i].set_title(f"Color Frequencies for Concept '{concept}' (excluding 'gray' colors)")
            else:
                axs[i].set_title(f"Color Frequencies for Concept '{concept}'")
            axs[i].set_title(f"Color Frequencies for Concept '{concept}'")
            axs[i].set_ylabel("Percentage of Images")
            axs[i].tick_params(axis='x', rotation=45, labelsize=8)
            axs[i].set_ylim(0, max_y_value)  # Set consistent y-axis limits

        plt.tight_layout()
        plt.show()

        # Save the plot as an image
        if filter_grays_out == True:
            save_filename = f"output_imgs/perceptual_data/color/filtered_grays_top_30_{ACs_list_name}.jpg"
        else:
            save_filename = f"output_imgs/perceptual_data/color/top_30_{ACs_list_name}.jpg"
        plt.savefig(save_filename)
        plt.show()
        return
    def palette_colors_frequencies_with_palettes(ACs_list_name, filter_grays_out):
        color_frequencies_by_concept, all_color_frequencies = calculate_colors_frequencies(ACs_list_name)
        # Sort the concepts alphabetically
        sorted_concepts = sorted(color_frequencies_by_concept.keys())
        num_concepts = len(sorted_concepts)
        fig, axs = plt.subplots(num_concepts + 1, 1, figsize=(12, 5 * (num_concepts + 1)), sharex=True)

        # Calculate top 30 colors across all concepts, excluding the top colors with "gray" in it
        all_color_frequencies = {}
        for concept, color_frequencies in color_frequencies_by_concept.items():
            for color, freq in color_frequencies.items():
                if filter_grays_out == True:
                    if "gray" not in color:
                        all_color_frequencies[color] = all_color_frequencies.get(color, 0) + freq
                else:
                    all_color_frequencies[color] = all_color_frequencies.get(color, 0) + freq
        sorted_all_frequencies = sorted(all_color_frequencies.items(), key=lambda x: x[1], reverse=True)
        top_all_colors, top_all_frequencies = zip(*sorted_all_frequencies[:30])

        # Convert color names to RGB values using webcolors
        top_all_colors_rgb = [webcolors.name_to_rgb(color) for color in top_all_colors]

        # Convert color names to RGB values using webcolors and normalize them
        top_all_colors_rgb_normalized = [tuple(value / 255.0 for value in color_rgb) for color_rgb in
                                         top_all_colors_rgb]

        # Create color patches for top colors with sizes based on frequencies
        color_patches = [
            Rectangle((0, 0), freq / max(top_all_frequencies), 1, fc=color_rgb)
            for color_rgb, freq in zip(top_all_colors_rgb_normalized, top_all_frequencies)
        ]
        axs[0].add_collection(PatchCollection(color_patches, match_original=True))
        if filter_grays_out == True:
            axs[0].set_title("Top 30 Colors Across All Concepts (excluding 'gray's)")
        else:
            axs[0].set_title("Top 30 Colors Across All Concepts")
        axs[0].set_xlim(0, 1)  # Set x-axis limit for the palette width

        for i, concept in enumerate(sorted_concepts, start=1):
            color_frequencies = color_frequencies_by_concept[concept]
            # Filter out colors containing "gray" in their names
            if filter_grays_out == True:
                filtered_color_frequencies = {color: freq for color, freq in color_frequencies.items() if
                                              "gray" not in color.lower()}
            else:
                filtered_color_frequencies = {color: freq for color, freq in color_frequencies.items()}

            sorted_frequencies = sorted(filtered_color_frequencies.items(), key=lambda x: x[1], reverse=True)
            colors, frequencies = zip(
                *sorted_frequencies[:30])  # Taking top 30 objects, excluding the first color "gre"

            # Convert color names to RGB values using webcolors
            colors_rgb = [webcolors.name_to_rgb(color) for color in colors]

            # Convert color names to RGB values using webcolors and normalize them
            colors_rgb_normalized = [tuple(value / 255.0 for value in color_rgb) for color_rgb in colors_rgb]

            # Create color patches for top colors with sizes based on frequencies
            color_patches = [
                plt.Rectangle((0, 0), freq / max(frequencies), 1, fc=color_rgb)
                for color_rgb, freq in zip(colors_rgb_normalized, frequencies)
            ]
            axs[i].add_collection(PatchCollection(color_patches, match_original=True))
            axs[i].set_title(f"Top Colors for Concept '{concept}'")

            if filter_grays_out == True:
                axs[i].set_title(f"Top Colors for Concept '{concept} (excluding 'gray's)")
            else:
                axs[i].set_title(f"Top Colors for Concept '{concept}'")
            axs[i].set_xlim(0, 1)  # Set x-axis limit for the palette width

        plt.tight_layout()
        plt.show()

        # Save the plot as an image
        if filter_grays_out == True:
            save_filename = f"output_imgs/perceptual_data/color/palette/filtered_grays_top_30_{ACs_list_name}.jpg"
        else:
            save_filename = f"output_imgs/perceptual_data/color/palette/top_30_{ACs_list_name}.jpg"
        plt.savefig(save_filename)
        plt.show()
        return

    def find_common_colors(ACs_list_name):
        color_frequencies_by_concept, all_color_frequencies = calculate_colors_frequencies(ACs_list_name)
        # Calculate overall frequency of each object across all concepts
        all_color_frequencies = {}
        for color_frequencies in color_frequencies_by_concept.values():
            for color, freq in color_frequencies.items():
                all_color_frequencies[color] = all_color_frequencies.get(color, 0) + freq

        # Sort objects based on their overall frequency in descending order
        sorted_overall_frequencies = sorted(all_color_frequencies.items(), key=lambda x: x[1], reverse=True)
        ordered_colors = [color for color, freq in sorted_overall_frequencies]

        # Get the set of common objects that are present in every concept
        common_colors = set.intersection(
            *[set(frequencies.keys()) for frequencies in color_frequencies_by_concept.values()])

        # Return the ordered set of common objects
        ordered_common_colors = [color for color in ordered_colors if color in common_colors]
        # common_objects = find_common_objects(object_frequencies_by_concept)
        print("for all concepts, the most common colors are (from most common to least: ", common_colors)
        return ordered_common_colors

    def find_unique_colors(ACs_list_name):
        color_frequencies_by_concept, all_color_frequencies = calculate_colors_frequencies(ACs_list_name)

        unique_colors_by_concept = {concept: set(frequencies.keys()) for concept, frequencies in
                                     color_frequencies_by_concept.items()}
        for concept, frequencies in color_frequencies_by_concept.items():
            for other_concept, other_frequencies in color_frequencies_by_concept.items():
                if concept != other_concept:
                    unique_colors_by_concept[concept] -= set(other_frequencies.keys())

        for concept, color_set in unique_colors_by_concept.items():
            if len(color_set) == 0:
                print(concept, "does not have any unique colors")
            else:
                print(concept, "is the only one with top colors: ", color_set)

        return unique_colors_by_concept

    def find_relevant_colors(ACs_list_name):
        color_frequencies_by_concept, all_color_frequencies = calculate_colors_frequencies(ACs_list_name)

        # Calculate overall frequency of each object across all concepts
        all_color_frequencies = Counter()
        for color_frequencies in color_frequencies_by_concept.values():
            all_color_frequencies.update(color_frequencies)

        relevant_colors_by_concept = {}

        for concept, color_frequencies in color_frequencies_by_concept.items():
            # Calculate relative frequency (TF) for each object within the concept
            relative_frequencies = {
                color: freq / all_color_frequencies[color]
                for color, freq in color_frequencies.items()
            }

            # Calculate inverse concept frequency (IDF) for each object
            num_concepts = len(color_frequencies_by_concept)
            inverse_concept_frequency = {color: num_concepts / sum(
                1 for concept_freqs in color_frequencies_by_concept.values() if color in concept_freqs)
                                         for color in color_frequencies}

            # Calculate relevance score for each object in the concept (TF * IDF)
            relevance_scores = {color: round(relative_frequencies[color] * inverse_concept_frequency[color], 3)
                                for color in color_frequencies}

            # Sort objects based on their relevance scores in descending order
            sorted_relevance_scores = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

            # Keep only the objects unique to the current concept's top 10 objects
            top_15_colors = [color for color, score in sorted_relevance_scores[:15]]

            # Check if each object appears in only the current concept's top 10 list
            unique_top_15_colors = []
            for color in top_15_colors:
                unique_to_current_concept = True
                for other_concept, other_top_15 in relevant_colors_by_concept.items():
                    if other_concept != concept and color in other_top_15:
                        unique_to_current_concept = False
                        break
                if unique_to_current_concept:
                    unique_top_15_colors.append(color)

            relevant_colors_by_concept[concept] = unique_top_15_colors
        # print(relevant_colors_by_concept)
        # for concept, color_set in relevant_colors_by_concept.items():
        #    print(concept, "has relevant top colors: ", color_set)
        return relevant_colors_by_concept

    def find_top_colors(ACs_list_name, num_top_objects=15):
        color_frequencies_by_concept, all_color_frequencies = calculate_colors_frequencies(ACs_list_name)

        top_colors_by_concept = {}
        for concept, frequencies in color_frequencies_by_concept.items():
            total_detected_colors = sum(frequencies.values())
            colors_percentages = {
                color: (count / total_detected_colors) * 100 for color, count in frequencies.items()
            }
            sorted_colors = sorted(colors_percentages.items(), key=lambda x: x[1], reverse=True)
            top_colors = dict(sorted_colors[:num_top_objects])
            top_colors_by_concept[concept] = top_colors
        print("Top colors by concept: ", top_colors_by_concept)
        ordered_lists_by_concept = {}
        for concept, color_scores in top_colors_by_concept.items():
            # Sort objects based on their scores in descending order
            sorted_colors = sorted(color_scores.items(), key=lambda x: x[1], reverse=True)
            # Extract the objects from the sorted list
            ordered_colors = [color for color, score in sorted_colors]
            ordered_lists_by_concept[concept] = ordered_colors
        # for concept, od_set in ordered_lists_by_concept.items():
        #     print(concept, "has top concepts: ", od_set)
        return top_colors_by_concept

    def find_top_relevant_colors(ACs_list_name, k=15):
        top_colors_by_concept = find_top_colors(ACs_list_name)
        relevant_colors_by_concept = find_relevant_colors(ACs_list_name)

        top_relevant_colors_by_concept = {}

        for concept, top_colors in top_colors_by_concept.items():
            # Get the relevant concepts for the current concept
            relevant_colors = relevant_colors_by_concept.get(concept, [])
            # Calculate the relevance scores for the relevant concepts
            relevance_scores = {color: 1 for color in relevant_colors}
            # Convert top_concepts dictionary to a list of tuples (object, score)
            top_colors_list = list(top_colors.items())
            # Sort the top_concepts_list based on scores in descending order
            top_colors_list.sort(key=lambda x: x[1], reverse=True)
            # Extract the objects from the sorted list
            top_colors_sorted = [color for color, _ in top_colors_list]
            # Ensure all objects in top_concepts_sorted have scores in relevance_scores dictionary
            relevance_scores.update({color: 0 for color in top_colors_sorted if color not in relevance_scores})
            # Calculate the Jaccard similarity between the top concepts and relevant concepts
            jaccard_scores = {color: len(set([color]).intersection(set(top_colors_sorted))) / len(
                set([color]).union(set(top_colors_sorted))) for color in relevant_colors}
            # Sort objects based on Jaccard similarity scores in descending order
            sorted_colors = sorted(jaccard_scores.items(), key=lambda x: x[1], reverse=True)
            # Take only the top k relevant concepts
            top_relevant_colors_by_concept[concept] = [obj for obj, _ in sorted_colors[:k]]
            # Print the top relevant concepts
        # for concept, concepts in top_relevant_objects_by_concept.items():
        #     print(concept, "has top relevant concepts:", concepts)
        # print(top_relevant_objects_by_concept)

        file_path = "top_relevant_jsons/top_relevant_colors_by_concept.json"
        # Using json.dump() to write the dictionary to a JSON file
        with open(file_path, "w") as json_file:
            json.dump(top_relevant_colors_by_concept, json_file)

        return top_relevant_colors_by_concept

    def find_top_relevant_colors_by_concept_w_freqs(ACs_list_name):
        top_relevant_colors_by_concept = find_top_relevant_colors(ACs_list_name)
        color_frequencies_by_concept, all_color_frequencies = calculate_colors_frequencies(ACs_list_name)
        top_relevant_colors_by_concept_w_freqs = {}

        for concept, top_relevant_colors in top_relevant_colors_by_concept.items():
            # Get the color frequencies for the current concept
            color_frequencies = color_frequencies_by_concept.get(concept, {})

            # Fetch frequencies for top relevant objects and create a dictionary {object: frequency}
            colors_with_freqs = {color: color_frequencies.get(color, 0) for color in top_relevant_colors}

            # Store the dictionary {object: frequency} for the current concept
            top_relevant_colors_by_concept_w_freqs[concept] = colors_with_freqs

        # Print the top relevant concepts and their frequencies
        # for concept, colors in top_relevant_colors_by_concept_w_freqs.items():
        #     print(concept, "has top relevant colors and frequencies:", colors)

        return top_relevant_colors_by_concept_w_freqs

    def create_colors_wordclouds(ACs_list_name):
        top_colors_by_concept = find_top_colors(ACs_list_name)
        top_relevant_colors_by_concept_w_freqs = find_top_relevant_colors_by_concept_w_freqs(ACs_list_name)
        class GroupedColorFunc(object):
            def __init__(self, color_to_words, default_color):
                self.color_func_to_words = [
                    (get_single_color_func(color), set(words))
                    for (color, words) in color_to_words.items()]

                self.default_color_func = get_single_color_func(default_color)

            def get_color_func(self, word):
                """Returns a single_color_func associated with the word"""
                try:
                    color_func = next(
                        color_func for (color_func, words) in self.color_func_to_words
                        if word in words)
                except StopIteration:
                    color_func = self.default_color_func

                return color_func

            def __call__(self, word, **kwargs):
                return self.get_color_func(word)(word, **kwargs)

        font_color = '#0074D9'  # Use any shade of blue you prefer
        helvetica_font = 'Helvetica.ttf'  # Replace with the path to your Helvetica font file

        # Generate word clouds for each concept
        for concept, top_colors in top_colors_by_concept.items():
            # Create word cloud objects
            wc_top_colors = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,  # Set the font to Helvetica
                color_func=lambda *args, **kwargs: font_color  # Set all words to blue color
            ).generate_from_frequencies(top_colors)

            # Get the top relevant objects for the current concept
            top_relevant_colors = top_relevant_colors_by_concept_w_freqs.get(concept, {})
            wc_top_relevant_colors = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,  # Set the font to Helvetica
                color_func=lambda *args, **kwargs: font_color  # Set all words to blue color
            ).generate_from_frequencies(top_relevant_colors)

            # Plot the word clouds side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(wc_top_colors, interpolation="bilinear")
            axes[0].axis("off")
            axes[0].set_title(f"Top Colors for Concept: {concept}")

            axes[1].imshow(wc_top_relevant_colors, interpolation="bilinear")
            axes[1].axis("off")
            axes[1].set_title(f"Top Relevant Colors for Concept: {concept}")

            plt.show()
            # Save the plot as an image
            save_filename = f"output_imgs/perceptual_data/color/wordclouds/{concept}_{ACs_list_name}_color_wordcloud.jpg"
            plt.savefig(save_filename)
            plt.show()

        return

    def create_colors_palettes(ACs_list_name):
        top_colors_by_concept = find_top_colors(ACs_list_name)
        top_relevant_colors_by_concept_w_freqs = find_top_relevant_colors_by_concept_w_freqs(ACs_list_name)
        font_color = '#0074D9'  # Use any shade of blue you prefer
        helvetica_font = 'Helvetica.ttf'  # Replace with the path to your Helvetica font file

        # Generate color palettes for each concept
        for concept, top_colors in top_colors_by_concept.items():
            # Convert color names to RGB values using webcolors and normalize them
            top_colors_rgb_normalized = [tuple(value / 255.0 for value in webcolors.name_to_rgb(color)) for color in
                                         top_colors]
            sorted_color_patches = [
                (color_rgb, freq)
                for color_rgb, freq in zip(top_colors_rgb_normalized, top_colors.values())
            ]
            sorted_color_patches.sort(key=lambda x: x[1], reverse=True)

            # Plot the color palettes
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            x_position = 0
            for color_rgb, freq in sorted_color_patches:
                rect = plt.Rectangle((x_position, 0), freq, 1, fc=color_rgb)
                axs[0].add_patch(rect)
                x_position += freq

            axs[0].axis("off")
            axs[0].set_xlim(0, x_position)  # Set x-axis limit for the palette width
            axs[0].set_title(f"Top Colors for Concept: {concept}")

            # Get the top relevant colors for the current concept
            top_relevant_colors = top_relevant_colors_by_concept_w_freqs.get(concept, {})
            top_relevant_colors_rgb_normalized = [tuple(value / 255.0 for value in webcolors.name_to_rgb(color)) for
                                                  color in top_relevant_colors]
            sorted_relevant_color_patches = [
                (color_rgb, freq)
                for color_rgb, freq in zip(top_relevant_colors_rgb_normalized, top_relevant_colors.values())
            ]
            sorted_relevant_color_patches.sort(key=lambda x: x[1], reverse=True)

            x_position = 0
            for color_rgb, freq in sorted_relevant_color_patches:
                rect = plt.Rectangle((x_position, 0), freq, 1, fc=color_rgb)
                axs[1].add_patch(rect)
                x_position += freq

            axs[1].axis("off")
            axs[1].set_xlim(0, x_position)  # Set x-axis limit for the palette width
            axs[1].set_title(f"Top Relevant Colors for Concept: {concept}")

            plt.show()

            # Save the plot as an image
            save_filename = f"output_imgs/perceptual_data/color/palette/{concept}_{ACs_list_name}_color_palette.jpg"
            plt.savefig(save_filename)
            plt.show()

        return

    for ACs_list_name in ACs_list_names:
        ## EXECUTION
        # get_top_colors_by_concept(ACs_list_name)
        # calculate_colors_frequencies(ACs_list_name)
        # plot_colors_frequencies_percentages(ACs_list_name, filter_grays_out=False)
        # palette_colors_frequencies_with_palettes(ACs_list_name,  filter_grays_out)
        # find_common_colors(ACs_list_name)
        # find_unique_colors(ACs_list_name)
        # find_relevant_colors(ACs_list_name)
        # find_top_colors(ACs_list_name)
        find_top_relevant_colors(ACs_list_name)
        # find_top_relevant_colors_by_concept_w_freqs(ACs_list_name)
        # create_colors_wordclouds(ACs_list_name)
        # create_colors_palettes(ACs_list_name)
    return

def colors_co_occurrences(ACs_list_names, filter_grays_out):
    def create_co_occurrence_matrix(concept_top_colors, filter_grays_out):
        # Flatten the list of color lists to get all color names
        all_colors = [color_name for color_list in concept_top_colors for color_name in color_list]
        color_names = sorted(list(set(all_colors)))

        # Initialize an empty co-occurrence matrix
        num_colors = len(color_names)
        co_occurrence_matrix = np.zeros((num_colors, num_colors), dtype=int)

        # Create a dictionary to map color names to matrix indices
        color_to_index = {color_name: index for index, color_name in enumerate(color_names)}

        # Populate the co-occurrence matrix based on the top color lists
        for top_colors in concept_top_colors:
            for color_name in top_colors:
                for other_color_name in top_colors:
                    if color_name != other_color_name:
                        # Increase the count for co-occurrence of color_name and other_color_name
                        i, j = color_to_index[color_name], color_to_index[other_color_name]
                        co_occurrence_matrix[i, j] += 1

        return co_occurrence_matrix, color_names
    def create_heatmap(concept_of_interest, co_occurrence_matrix, color_names):
        # Create a heatmap using seaborn
        plt.figure(figsize=(12, 10))
        sns.heatmap(co_occurrence_matrix, annot=False, fmt='d', cmap="YlGnBu", xticklabels=color_names,
                    yticklabels=color_names, cbar=True, cbar_kws={"label": "Co-occurrence count"})
        plt.title(f"Co-occurrence Heatmap for Concept: {concept_of_interest}")
        plt.xlabel("Colors")
        plt.ylabel("Colors")
        plt.show()
        # Save the plot as an image
        save_filename = f"output_imgs/perceptual_data/color/co_occurrence_heatmaps/{concept_of_interest}_{ACs_list_name}_color_cooccurr_heatmap.jpg"
        plt.savefig(save_filename)
        plt.show()

    def get_occurrence_pairs(co_occurrence_matrix, color_names, concept_of_interest, filter_grays):
        if filter_grays:
            # Exclude colors containing "gray" in their names
            filtered_indices = [i for i, name in enumerate(color_names) if "gray" not in name.lower()]
            co_occurrence_matrix = co_occurrence_matrix[filtered_indices][:, filtered_indices]
            color_names = [color_names[i] for i in filtered_indices]

        # Get the top 15 co-occurrences
        top_indices = np.unravel_index(np.argsort(co_occurrence_matrix.ravel())[-15:], co_occurrence_matrix.shape)
        top_color_pairs = [(color_names[i], color_names[j]) for i, j in zip(*top_indices)]

        # Create the heatmap visualization
        fig, axs = plt.subplots(len(top_color_pairs), 2, figsize=(6, 2 * len(top_color_pairs)))
        fig.suptitle(f'Co-occurrences for Concept: {concept_of_interest}', fontsize=16)

        for idx, (color1, color2) in enumerate(top_color_pairs):
            ax = axs[idx, 0]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(color1)
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=color1))

            ax = axs[idx, 1]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(color2)
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=color2))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the plot as an image
        save_filename = f"output_imgs/perceptual_data/color/co_occurrence_pairs/{concept_of_interest}_{ACs_list_name}_color_cooccurr_heatmap.jpg"
        plt.savefig(save_filename)
        plt.show()
    def set_occurrence_heatmaps(ACs_list_name, concept_of_interest, filter_grays_out):
        concept_detected_colors = []
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        for img_id, image_info in merged_ARTstract.items():
            concept_name = None
            detected_colors = None

            # Find the concept_name and detected_objects dynamically
            for key, value in image_info['evoked_clusters'].items():
                if value.get('cluster_name') == concept_of_interest:
                    concept_name = value['cluster_name']
                    detected_colors_list = image_info['color'].get("ARTstract_color_2023_06_26", [])
                    detected_colors = [color_object["webcolor_name"] for color_object in
                                            detected_colors_list]
            if concept_name and detected_colors:
                concept_detected_colors.append(detected_colors)

        co_occurrence_matrix, color_names = create_co_occurrence_matrix(concept_detected_colors, filter_grays_out)
        get_occurrence_pairs(co_occurrence_matrix, color_names, concept_of_interest, filter_grays=True)
        create_heatmap(concept_of_interest, co_occurrence_matrix, color_names)

    concepts_of_interest = ['comfort', 'danger', 'death', 'fitness', 'freedom', 'power', 'safety']
    for concept_of_interest in concepts_of_interest:
        for ACs_list_name in ACs_list_names:
            set_occurrence_heatmaps(ACs_list_name, concept_of_interest, filter_grays_out)
    return

## Emotions
def stats_emotions(ACs_list_names):
    def get_emotions_by_concept(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        emotions_by_concept = {concept: [] for concept in concept_images}
        emotions_by_concept_w_strengths = {concept: [] for concept in concept_images}

        for concept, images in concept_images.items():
            for img in images:
                for image_id, image_info in merged_ARTstract.items():
                    if img == image_id:
                        emotion_object = merged_ARTstract[img].get("em", {}).get("ARTstract_em_2023_06_26", {})
                        emotion = emotion_object['emotion']
                        em_strength = emotion_object['annotation_strength']
                        emotions_by_concept[concept].append(emotion)
                        emotions_by_concept_w_strengths[concept].append((emotion, em_strength))
        return emotions_by_concept, emotions_by_concept_w_strengths

    def calculate_emotion_frequencies(ACs_list_name):
        emotions_by_concept, emotions_by_concept_w_strengths = get_emotions_by_concept(ACs_list_name)
        emotion_frequencies_by_concept = {}
        all_detected_emotions = []

        for concept, emotions_list in emotions_by_concept.items():
            emotion_frequencies = Counter(emotions_list)
            emotion_frequencies_by_concept[concept] = emotion_frequencies
            all_detected_emotions.extend(emotions_list)

        all_emotions_frequencies = Counter(all_detected_emotions)
        return emotion_frequencies_by_concept, all_emotions_frequencies

    def plot_emotion_frequencies(ACs_list_name):
        emotion_frequencies_by_concept, all_emotions_frequencies = calculate_emotion_frequencies(ACs_list_name)
        num_concepts = len(emotion_frequencies_by_concept)
        emotions = ["amusement", "awe", "anger", "contentment", "fear", "excitement", "sadness", "disgust",
                    "something else"]
        fig, axs = plt.subplots(num_concepts + 1, 1, figsize=(12, 5 * (num_concepts + 1)), sharex=True)

        # Calculate the total number of images for each concept
        total_images_by_concept = {concept: sum(emotion_frequencies.values()) for concept, emotion_frequencies in
                                   emotion_frequencies_by_concept.items()}

        # Calculate the overall emotion frequencies across all concepts
        all_emotion_frequencies = Counter()
        for emotion_frequencies in emotion_frequencies_by_concept.values():
            all_emotion_frequencies.update(emotion_frequencies)

        # Calculate the percentage of images with each emotion
        for emotion in emotions:
            all_emotion_frequencies[emotion] /= sum(total_images_by_concept.values())

        # Sort emotions based on their overall frequency in descending order
        sorted_all_emotion_frequencies = sorted(all_emotion_frequencies.items(), key=lambda x: x[1], reverse=True)
        ordered_emotions = [emotion for emotion, freq in sorted_all_emotion_frequencies]

        max_y_value = 0.75  # Set the maximum y-axis value to 0.75 (75%)

        # Plot the total emotion frequencies subplot
        frequencies_total = [all_emotion_frequencies.get(emotion, 0) for emotion in ordered_emotions]
        axs[0].bar(ordered_emotions, frequencies_total)
        axs[0].set_title("Overall Emotion Frequencies")
        axs[0].set_ylabel("Percentage of Images")
        axs[0].tick_params(axis='x', rotation=45, labelsize=8)
        axs[0].set_ylim(0, max_y_value)  # Set consistent y-axis limits

        # Plot the emotion frequencies for each concept subplot
        sorted_concepts = sorted(emotion_frequencies_by_concept.keys())  # Sort concepts alphabetically
        for i, concept in enumerate(sorted_concepts, start=1):
            emotion_frequencies = emotion_frequencies_by_concept[concept]
            frequencies = [emotion_frequencies.get(emotion, 0) / total_images_by_concept[concept] for emotion in
                           ordered_emotions]
            axs[i].bar(ordered_emotions, frequencies)
            axs[i].set_title(f"Emotion Frequencies for Concept '{concept}'")
            axs[i].set_ylabel("Percentage of Images")
            axs[i].tick_params(axis='x', rotation=45, labelsize=8)
            axs[i].set_ylim(0, max_y_value)  # Set consistent y-axis limits

        plt.tight_layout()
        plt.show()

        # Save the plot as an image
        save_filename = f"output_imgs/perceptual_data/emotion/emotion_frequencies_{ACs_list_name}.jpg"
        plt.savefig(save_filename)
        plt.show()

    def find_relevant_emotions(ACs_list_name):
        emotion_frequencies_by_concept, all_emotions_frequencies = calculate_emotion_frequencies(ACs_list_name)
        emotions = ["amusement", "awe", "anger", "contentment", "fear", "excitement", "sadness", "disgust",
                    "something else"]

        # Calculate overall frequency of each emotion across all concepts
        all_emotions_frequencies = Counter()
        for emotion_frequencies in emotion_frequencies_by_concept.values():
            all_emotions_frequencies.update(emotion_frequencies)

        relevant_emotions_by_concept = {}

        for concept, emotion_frequencies in emotion_frequencies_by_concept.items():
            # Calculate relative frequency (TF) for each emotion within the concept
            relative_frequencies = {
                emotion: freq / sum(emotion_frequencies.values())
                for emotion, freq in emotion_frequencies.items()
            }

            # Calculate inverse concept frequency (IDF) for each emotion
            num_concepts = len(emotion_frequencies_by_concept)
            inverse_concept_frequency = {emotion: num_concepts / sum(
                1 for emotion_frequencies in emotion_frequencies_by_concept.values() if emotion in emotion_frequencies)
                                         for emotion in emotion_frequencies}

            # Calculate relevance score for each emotion in the concept (TF * IDF)
            relevance_scores = {emotion: round(relative_frequencies[emotion] * inverse_concept_frequency[emotion], 3)
                                for emotion in emotion_frequencies}

            # Sort emotions based on their relevance scores in descending order
            sorted_relevance_scores = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

            # Keep the top 4 emotions based on frequencies and the top 4 relevant emotions
            top_frequencies_emotions = [emotion for emotion, score in
                                        sorted(emotion_frequencies.items(), key=lambda x: x[1], reverse=True)[:4]]
            top_relevance_emotions = [emotion for emotion, score in sorted_relevance_scores[:5]]
            relevant_emotions_by_concept[concept] = (top_frequencies_emotions, top_relevance_emotions)

        # print(relevant_emotions_by_concept)
        # for concept, (freq_emotion_set, rel_emotion_set) in relevant_emotions_by_concept.items():
        #     print(concept, "has top frequency emotions:", freq_emotion_set)
        #     print(concept, "has top relevance emotions:", rel_emotion_set)





        file_path = "top_relevant_jsons/top_relevant_emotions_by_concept.json"
        # Using json.dump() to write the dictionary to a JSON file
        with open(file_path, "w") as json_file:
            top_relevant_emotions_by_concept = {key: value[1] for key, value in relevant_emotions_by_concept.items() if value}

            json.dump(top_relevant_emotions_by_concept, json_file)

        return relevant_emotions_by_concept

    def create_emotion_wordclouds(ACs_list_name, threshold=0.5):
        emotions_by_concept, emotions_by_concept_w_strengths = get_emotions_by_concept(ACs_list_name)
        font_color = '#0074D9'  # Use any shade of blue you prefer
        helvetica_font = 'Helvetica.ttf'  # Replace with the path to your Helvetica font file

        for concept, emotions_with_strengths in emotions_by_concept_w_strengths.items():
            emotions = [(emotion, strength) for emotion, strength in emotions_with_strengths]
            emotions = {emotion: strength for emotion, strength in emotions}

            emotions_above_threshold = [(emotion, strength) for emotion, strength in emotions_with_strengths if
                                        strength >= threshold]
            emotions_above_threshold = {emotion: strength for emotion, strength in emotions_above_threshold}

            wc_emotions = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,
                color_func=lambda *args, **kwargs: font_color
            ).generate_from_frequencies(emotions)

            wc_emotions_1 = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,
                color_func=lambda *args, **kwargs: font_color
            ).generate_from_frequencies(emotions_above_threshold)

            # Plot the word clouds side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(wc_emotions, interpolation="bilinear")
            axes[0].axis("off")
            axes[0].set_title(f"Emotions Word Cloud for: {concept} (no threshold)")

            axes[1].imshow(wc_emotions_1, interpolation="bilinear")
            axes[1].axis("off")
            axes[1].set_title(f"Emotion Word Cloud for: {concept} ( threshold {threshold})")

            plt.show()
            save_filename = f"output_imgs/perceptual_data/emotion/wordclouds/emotion_wordcloud_{concept}_{ACs_list_name}.jpg"
            plt.savefig(save_filename)
        return

    # List of concepts of interest
    concepts_of_interest = ['comfort', 'danger', 'death', 'fitness', 'freedom', 'power', 'safety']
    for ACs_list_name in ACs_list_names:
        # get_emotions_by_concept(ACs_list_name)
        # plot_emotion_frequencies(ACs_list_name)
        # create_emotion_wordclouds(ACs_list_name)
        find_relevant_emotions(ACs_list_name)

    return

## Age
def stats_age(ACs_list_names):
    def get_ages_by_concept(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        ages_by_concept = {concept: [] for concept in concept_images}
        ages_by_concept_w_strengths = {concept: [] for concept in concept_images}

        for concept, images in concept_images.items():
            for img in images:
                for image_id, image_info in merged_ARTstract.items():
                    if img == image_id:
                        age_object = merged_ARTstract[img].get("age", {}).get("ARTstract_age_2023_06_26", {})
                        age = age_object['age_tier']
                        age_strength = age_object['annotation_strength']
                        ages_by_concept[concept].append(age)
                        ages_by_concept_w_strengths[concept].append((age, age_strength))
        return ages_by_concept, ages_by_concept_w_strengths

    def calculate_age_frequencies(ACs_list_name):
        ages_by_concept, ages_by_concept_w_strengths = get_ages_by_concept(ACs_list_name)
        age_frequencies_by_concept = {}
        all_detected_ages = []

        for concept, ages_list in ages_by_concept.items():
            age_frequencies = Counter(ages_list)
            age_frequencies_by_concept[concept] = age_frequencies
            all_detected_ages.extend(ages_list)

        all_ages_frequencies = Counter(all_detected_ages)
        return age_frequencies_by_concept, all_ages_frequencies

    def plot_age_frequencies(ACs_list_name):
        age_frequencies_by_concept, all_ages_frequencies = calculate_age_frequencies(ACs_list_name)
        num_concepts = len(age_frequencies_by_concept)
        ages = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        fig, axs = plt.subplots(num_concepts + 1, 1, figsize=(12, 5 * (num_concepts + 1)), sharex=True)

        # Calculate the total number of images for each concept
        total_images_by_concept = {concept: sum(age_frequencies.values()) for concept, age_frequencies in
                                   age_frequencies_by_concept.items()}

        max_y_value = 1  # Set the maximum y-axis value to 1

        # Calculate the total number of images across all concepts
        total_images = sum(total_images_by_concept.values())

        # Calculate the total age frequencies subplot as a percentage of total images
        frequencies_total = [all_ages_frequencies.get(age, 0) / total_images * 100 for age in ages]

        axs[0].bar(ages, frequencies_total)
        axs[0].set_title("Overall Age Frequencies")
        axs[0].set_ylabel("Percentage of Images")
        axs[0].tick_params(axis='x', rotation=45, labelsize=8)
        axs[0].set_ylim(0, 100)  # Set y-axis limits to 0-100 for percentage scale

        # Plot the age frequencies for each concept subplot
        sorted_concepts = sorted(age_frequencies_by_concept.keys())  # Sort concepts alphabetically
        ordered_ages = ages  # Use the predefined order of ages

        for i, concept in enumerate(sorted_concepts, start=1):
            age_frequencies = age_frequencies_by_concept[concept]
            frequencies = [age_frequencies.get(age, 0) / total_images_by_concept[concept] for age in ordered_ages]
            axs[i].bar(ordered_ages, frequencies)
            axs[i].set_title(f"Age Frequencies for Concept '{concept}'")
            axs[i].set_ylabel("Percentage of Images")
            axs[i].tick_params(axis='x', rotation=45, labelsize=8)
            axs[i].set_ylim(0, max_y_value)  # Set consistent y-axis limits

        plt.tight_layout()
        plt.show()

        # Save the plot as an image
        save_filename = f"output_imgs/perceptual_data/age/age_frequencies_{ACs_list_name}.jpg"
        plt.savefig(save_filename)
        plt.show()
        return

    def find_relevant_ages(ACs_list_name):
        age_frequencies_by_concept, all_ages_frequencies = calculate_age_frequencies(ACs_list_name)
        ages = ['0-2','3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

        # Calculate the overall age frequencies across all concepts
        all_age_frequencies = Counter()
        for age_frequencies in age_frequencies_by_concept.values():
            all_age_frequencies.update(age_frequencies)

        relevant_ages_by_concept = {}

        for concept, age_frequencies in age_frequencies_by_concept.items():
            # Calculate relative frequency (TF) for each emotion within the concept
            relative_frequencies = {
                age: freq / sum(age_frequencies.values())
                for age, freq in age_frequencies.items()
            }

            # Calculate inverse concept frequency (IDF) for each emotion
            num_concepts = len(age_frequencies_by_concept)
            inverse_concept_frequency = {age: num_concepts / sum(
                1 for age_frequencies in age_frequencies_by_concept.values() if age in age_frequencies)
                                         for age in age_frequencies}

            # Calculate relevance score for each emotion in the concept (TF * IDF)
            relevance_scores = {age: round(relative_frequencies[age] * inverse_concept_frequency[age], 3)
                                for age in age_frequencies}

            # Sort emotions based on their relevance scores in descending order
            sorted_relevance_scores = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

            # Keep the top 4 emotions based on frequencies and the top 4 relevant emotions
            top_frequencies_ages = [age for age, score in
                                        sorted(age_frequencies.items(), key=lambda x: x[1], reverse=True)[:4]]
            top_relevance_ages = [age for age, score in sorted_relevance_scores[:5]]
            relevant_ages_by_concept[concept] = (top_frequencies_ages, top_relevance_ages)
        #
        # print(relevant_ages_by_concept)
        # for concept, (freq_age_set, rel_age_set) in relevant_ages_by_concept.items():
        #     print(concept, "has top frequency ages:", freq_age_set)
        #     print(concept, "has top relevance ages:", rel_age_set)
        #


        file_path = "top_relevant_jsons/top_relevant_ages_by_concept.json"
        # Using json.dump() to write the dictionary to a JSON file
        with open(file_path, "w") as json_file:
            top_relevant_ages_by_concept = {key: value[1] for key, value in relevant_ages_by_concept.items() if value}

            json.dump(top_relevant_ages_by_concept, json_file)

        return relevant_ages_by_concept

    def find_thresh_relevant_ages(ACs_list_name, threshold=0.5):
        age_frequencies_by_concept, all_ages_frequencies = calculate_age_frequencies(ACs_list_name)

        ages = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

        # Calculate the overall age frequencies across all concepts
        all_age_frequencies = Counter()
        for age_frequencies in age_frequencies_by_concept.values():
            all_age_frequencies.update(age_frequencies)

        relevant_ages_by_concept = {}

        for concept, age_frequencies in age_frequencies_by_concept.items():
            # Calculate relative frequency (TF) for each age tier within the concept
            relative_frequencies = {
                age: freq / sum(age_frequencies.values())
                for age, freq in age_frequencies.items()
            }

            # Calculate inverse concept frequency (IDF) for each age tier
            num_concepts = len(age_frequencies_by_concept)
            inverse_concept_frequency = {age: num_concepts / sum(
                1 for age_frequencies in age_frequencies_by_concept.values() if age in age_frequencies)
                                         for age in age_frequencies}

            # Calculate relevance score for each age tier in the concept (TF * IDF)
            relevance_scores = {age: round(relative_frequencies[age] * inverse_concept_frequency[age], 3)
                                for age in age_frequencies}

            # Filter age tiers based on the threshold
            top_relevance_ages = [age for age, score in relevance_scores.items() if score >= threshold]

            # Sort top relevance age tiers by their relevance scores in descending order
            top_relevance_ages.sort(key=lambda age: relevance_scores[age], reverse=True)

            # Keep the top 4 age tiers based on relevance score (or all if fewer than 4)
            top_relevance_ages = top_relevance_ages[:5]

            relevant_ages_by_concept[concept] = top_relevance_ages
        file_path = "top_relevant_jsons/top_thresh_relevant_ages_by_concept.json"
        # Using json.dump() to write the dictionary to a JSON file
        with open(file_path, "w") as json_file:
            top_relevant_ages_by_concept = relevant_ages_by_concept

            json.dump(top_relevant_ages_by_concept, json_file)

        return relevant_ages_by_concept

    def create_age_wordclouds(ACs_list_name, threshold=0.5):
        ages_by_concept, ages_by_concept_w_strengths = get_ages_by_concept(ACs_list_name)
        font_color = '#0074D9'  # Use any shade of blue you prefer
        helvetica_font = 'Helvetica.ttf'

        for concept, ages_with_strengths in ages_by_concept_w_strengths.items():
            ages = [(age, strength) for age, strength in ages_with_strengths]
            ages = {age: strength for age, strength in ages}

            ages_above_threshold = [(age, strength) for age, strength in ages_with_strengths if
                                        strength >= threshold]
            ages_above_threshold = {age: strength for age, strength in ages_above_threshold}

            wc_emotions = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,
                color_func=lambda *args, **kwargs: font_color
            ).generate_from_frequencies(ages)

            wc_emotions_1 = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,
                color_func=lambda *args, **kwargs: font_color
            ).generate_from_frequencies(ages_above_threshold)

            # Plot the word clouds side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(wc_emotions, interpolation="bilinear")
            axes[0].axis("off")
            axes[0].set_title(f"Age Tiers Word Cloud for: {concept} (no threshold)")

            axes[1].imshow(wc_emotions_1, interpolation="bilinear")
            axes[1].axis("off")
            axes[1].set_title(f"Age Tiers Word Cloud for: {concept} ( threshold {threshold})")

            plt.show()
            save_filename = f"output_imgs/perceptual_data/age/wordclouds/age_wordcloud_{concept}_{ACs_list_name}.jpg"
            plt.savefig(save_filename)
        return

    # List of concepts of interest
    concepts_of_interest = ['comfort', 'danger', 'death', 'fitness', 'freedom', 'power', 'safety']
    for ACs_list_name in ACs_list_names:
        # get_ages_by_concept(ACs_list_name)
        # plot_age_frequencies(ACs_list_name)
        find_relevant_ages(ACs_list_name)
        # create_age_wordclouds(ACs_list_name)

    return

## Art Style
def stats_art_style(ACs_list_names):
    def get_art_styles_by_concept(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        art_styles_by_concept = {concept: [] for concept in concept_images}
        art_styles_by_concept_w_strengths = {concept: [] for concept in concept_images}

        for concept, images in concept_images.items():
            for img in images:
                for image_id, image_info in merged_ARTstract.items():
                    if img == image_id:
                        art_style_object = merged_ARTstract[img].get("as", {}).get("ARTstract_as_2023_06_26", {})
                        art_style = art_style_object['art_style']
                        art_style_strength = art_style_object['annotation_strength']
                        art_styles_by_concept[concept].append(art_style)
                        art_styles_by_concept_w_strengths[concept].append((art_style, art_style_strength))
        return art_styles_by_concept, art_styles_by_concept_w_strengths

    def calculate_art_style_frequencies(ACs_list_name):
        art_styles_by_concept, art_styles_by_concept_w_strengths = get_art_styles_by_concept(ACs_list_name)
        art_style_frequencies_by_concept = {}
        all_detected_art_styles = []

        for concept, art_styles_list in art_styles_by_concept.items():
            art_style_frequencies = Counter(art_styles_list)
            art_style_frequencies_by_concept[concept] = art_style_frequencies
            all_detected_art_styles.extend(art_styles_list)

        all_art_styles_frequencies = Counter(all_detected_art_styles)
        return art_style_frequencies_by_concept, all_art_styles_frequencies

    def plot_art_style_frequencies(ACs_list_name):
        art_style_frequencies_by_concept, all_art_styles_frequencies = calculate_art_style_frequencies(ACs_list_name)
        num_concepts = len(art_style_frequencies_by_concept)
        art_styles = ["Art Nouveau", "Baroque", "Expressionism", "Impressionism", "Post-Impressionism", "Realism",
                      "Renaissance", "Romanticism", "Surrealism", "Ukiyo-e"]
        fig, axs = plt.subplots(num_concepts + 1, 1, figsize=(12, 5 * (num_concepts + 1)), sharex=True)

        # Calculate the total number of images for each concept
        total_images_by_concept = {concept: sum(art_style_frequencies.values()) for concept, art_style_frequencies in
                                   art_style_frequencies_by_concept.items()}

        max_y_value = 1  # Set the maximum y-axis value to 1

        # Calculate the total number of images across all concepts
        total_images = sum(total_images_by_concept.values())

        # Calculate the total art style frequencies subplot as a percentage of total images
        frequencies_total = [all_art_styles_frequencies.get(art_style, 0) / total_images * 100 for art_style in
                             art_styles]

        axs[0].bar(art_styles, frequencies_total)
        axs[0].set_title("Overall Art Style Frequencies")
        axs[0].set_ylabel("Percentage of Images")
        axs[0].tick_params(axis='x', rotation=45, labelsize=8)
        axs[0].set_ylim(0, 100)  # Set y-axis limits to 0-100 for percentage scale

        # Plot the art style frequencies for each concept subplot
        sorted_concepts = sorted(art_style_frequencies_by_concept.keys())  # Sort concepts alphabetically
        ordered_art_styles = art_styles  # Use the predefined order of art styles

        for i, concept in enumerate(sorted_concepts, start=1):
            art_style_frequencies = art_style_frequencies_by_concept[concept]
            frequencies = [art_style_frequencies.get(art_style, 0) / total_images_by_concept[concept] for art_style in
                           ordered_art_styles]
            axs[i].bar(ordered_art_styles, frequencies)
            axs[i].set_title(f"Art Style Frequencies for Concept '{concept}'")
            axs[i].set_ylabel("Percentage of Images")
            axs[i].tick_params(axis='x', rotation=45, labelsize=8)
            axs[i].set_ylim(0, max_y_value)  # Set consistent y-axis limits

        plt.tight_layout()
        plt.show()

        # Save the plot as an image
        save_filename = f"output_imgs/perceptual_data/art_style/art_style_frequencies_{ACs_list_name}.jpg"
        plt.savefig(save_filename)
        plt.show()

    def find_relevant_art_styles(ACs_list_name):
        art_style_frequencies_by_concept, all_art_styles_frequencies = calculate_art_style_frequencies(ACs_list_name)

        # Calculate the overall art style frequencies across all concepts
        all_art_style_frequencies = Counter()
        for art_style_frequencies in art_style_frequencies_by_concept.values():
            all_art_style_frequencies.update(art_style_frequencies)

        relevant_art_styles_by_concept = {}

        for concept, art_style_frequencies in art_style_frequencies_by_concept.items():
            # Calculate relative frequency (TF) for each art style within the concept
            relative_frequencies = {
                art_style: freq / sum(art_style_frequencies.values())
                for art_style, freq in art_style_frequencies.items()
            }

            # Calculate inverse concept frequency (IDF) for each art style
            num_concepts = len(art_style_frequencies_by_concept)
            inverse_concept_frequency = {art_style: num_concepts / sum(
                1 for art_style_frequencies in art_style_frequencies_by_concept.values() if art_style in art_style_frequencies)
                                         for art_style in art_style_frequencies}

            # Calculate relevance score for each art style in the concept (TF * IDF)
            relevance_scores = {art_style: round(relative_frequencies[art_style] * inverse_concept_frequency[art_style], 3)
                                for art_style in art_style_frequencies}

            # Sort art styles based on their relevance scores in descending order
            sorted_relevance_scores = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

            # Keep the top 4 art styles based on frequencies and the top 4 relevant art styles
            top_frequencies_art_styles = [art_style for art_style, score in
                                          sorted(art_style_frequencies.items(), key=lambda x: x[1], reverse=True)[:4]]
            top_relevance_art_styles = [art_style for art_style, score in sorted_relevance_scores[:4]]
            relevant_art_styles_by_concept[concept] = (top_frequencies_art_styles, top_relevance_art_styles)

        for concept, (freq_art_style_set, rel_art_style_set) in relevant_art_styles_by_concept.items():
            print(concept, "has top frequency art styles:", freq_art_style_set)
            print(concept, "has top relevance art styles:", rel_art_style_set)


        file_path = "top_relevant_jsons/top_relevant_art_styles_by_concept.json"
        # Using json.dump() to write the dictionary to a JSON file
        with open(file_path, "w") as json_file:
            top_relevant_art_styles_by_concept = {key: value[1] for key, value in relevant_art_styles_by_concept.items() if value}

            json.dump(top_relevant_art_styles_by_concept, json_file)

        return relevant_art_styles_by_concept

    def create_art_style_wordclouds(ACs_list_name, threshold=0.5):
        art_styles_by_concept, art_styles_by_concept_w_strengths = get_art_styles_by_concept(ACs_list_name)
        font_color = '#0074D9'  # Use any shade of blue you prefer
        helvetica_font = 'Helvetica.ttf'

        for concept, art_styles_with_strengths in art_styles_by_concept_w_strengths.items():
            art_styles = [(art_style, strength) for art_style, strength in art_styles_with_strengths]
            art_styles = {art_style: strength for art_style, strength in art_styles}

            art_styles_above_threshold = [(art_style, strength) for art_style, strength in art_styles_with_strengths if
                                        strength >= threshold]
            art_styles_above_threshold = {art_style: strength for art_style, strength in art_styles_above_threshold}

            wc_art_styles = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,
                color_func=lambda *args, **kwargs: font_color
            ).generate_from_frequencies(art_styles)

            wc_art_styles_1 = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,
                color_func=lambda *args, **kwargs: font_color
            ).generate_from_frequencies(art_styles_above_threshold)

            # Plot the word clouds side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(wc_art_styles, interpolation="bilinear")
            axes[0].axis("off")
            axes[0].set_title(f"Art Styles Word Cloud for: {concept} (no threshold)")

            axes[1].imshow(wc_art_styles_1, interpolation="bilinear")
            axes[1].axis("off")
            axes[1].set_title(f"Art Style Word Cloud for: {concept} ( threshold {threshold})")

            plt.show()
            save_filename = f"output_imgs/perceptual_data/art_style/wordclouds/art_style_wordcloud_{concept}_{ACs_list_name}.jpg"
            plt.savefig(save_filename)
        return

    # List of concepts of interest
    concepts_of_interest = ['comfort', 'danger', 'death', 'fitness', 'freedom', 'power', 'safety']
    for ACs_list_name in ACs_list_names:
        # get_art_styles_by_concept(ACs_list_name)
        # plot_art_style_frequencies(ACs_list_name)
        find_relevant_art_styles(ACs_list_name)
        # create_art_style_wordclouds(ACs_list_name)

    return

## Action
def stats_action(ACs_list_names):
    def get_actions_by_concept(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        actions_by_concept = {concept: [] for concept in concept_images}
        actions_by_concept_w_strengths = {concept: [] for concept in concept_images}

        for concept, images in concept_images.items():
            for img in images:
                for image_id, image_info in merged_ARTstract.items():
                    if img == image_id:
                        action_object = merged_ARTstract[img].get("act", {}).get("ARTstract_act_2023_06_28", {})
                        action = action_object['action_label']
                        action_strength = action_object['annotation_strength']
                        actions_by_concept[concept].append(action)
                        actions_by_concept_w_strengths[concept].append((action, action_strength))
        return actions_by_concept, actions_by_concept_w_strengths

    def calculate_action_frequencies(ACs_list_name):
        actions_by_concept, actions_by_concept_w_strengths = get_actions_by_concept(ACs_list_name)
        action_frequencies_by_concept = {}
        all_detected_actions = []

        for concept, actions_list in actions_by_concept.items():
            action_frequencies = Counter(actions_list)
            action_frequencies_by_concept[concept] = action_frequencies
            all_detected_actions.extend(actions_list)

        all_actions_frequencies = Counter(all_detected_actions)
        return action_frequencies_by_concept, all_actions_frequencies

    def plot_action_frequencies(ACs_list_name):
        action_frequencies_by_concept, all_actions_frequencies = calculate_action_frequencies(ACs_list_name)
        num_concepts = len(action_frequencies_by_concept)
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
        fig, axs = plt.subplots(num_concepts + 1, 1, figsize=(12, 5 * (num_concepts + 1)), sharex=True)

        # Calculate the total number of images for each concept
        total_images_by_concept = {concept: sum(action_frequencies.values()) for concept, action_frequencies in
                                   action_frequencies_by_concept.items()}

        max_y_value = 1  # Set the maximum y-axis value to 1

        # Calculate the total number of images across all concepts
        total_images = sum(total_images_by_concept.values())

        # Calculate the total action frequencies subplot as a percentage of total images
        frequencies_total = [all_actions_frequencies.get(action, 0) / total_images * 100 for action in actions]

        axs[0].bar(actions, frequencies_total)
        axs[0].set_title("Overall Action Frequencies")
        axs[0].set_ylabel("Percentage of Images")
        axs[0].tick_params(axis='x', rotation=45, labelsize=8)
        axs[0].set_ylim(0, 100)  # Set y-axis limits to 0-100 for percentage scale

        # Plot the action frequencies for each concept subplot
        sorted_concepts = sorted(action_frequencies_by_concept.keys())  # Sort concepts alphabetically
        ordered_actions = actions  # Use the predefined order of actions

        for i, concept in enumerate(sorted_concepts, start=1):
            action_frequencies = action_frequencies_by_concept[concept]
            frequencies = [action_frequencies.get(action, 0) / total_images_by_concept[concept] for action in
                           ordered_actions]
            axs[i].bar(ordered_actions, frequencies)
            axs[i].set_title(f"Action Frequencies for Concept '{concept}'")
            axs[i].set_ylabel("Percentage of Images")
            axs[i].tick_params(axis='x', rotation=45, labelsize=8)
            axs[i].set_ylim(0, max_y_value)  # Set consistent y-axis limits

        plt.tight_layout()
        plt.show()

        # Save the plot as an image
        save_filename = f"output_imgs/perceptual_data/action/action_frequencies_{ACs_list_name}.jpg"
        plt.savefig(save_filename)
        plt.show()

    # def find_relevant_actions_normalized(ACs_list_name):
    #     action_frequencies_by_concept, all_actions_frequencies = calculate_action_frequencies(ACs_list_name)
    #
    #     # Calculate the overall action frequencies across all concepts
    #     all_action_frequencies = Counter()
    #     for action_frequencies in action_frequencies_by_concept.values():
    #         all_action_frequencies.update(action_frequencies)
    #
    #     relevant_actions_by_concept = {}
    #
    #     for concept, action_frequencies in action_frequencies_by_concept.items():
    #         # Calculate the sum of frequencies for normalization
    #         sum_of_frequencies = sum(action_frequencies.values())
    #
    #         # Calculate relative frequency (TF) for each action within the concept and normalize
    #         relative_frequencies = {
    #             action: freq / sum_of_frequencies
    #             for action, freq in action_frequencies.items()
    #         }
    #
    #         # Calculate inverse concept frequency (IDF) for each action
    #         num_concepts = len(action_frequencies_by_concept)
    #         inverse_concept_frequency = {action: num_concepts / sum(
    #             1 for action_frequencies in action_frequencies_by_concept.values() if action in action_frequencies)
    #                                      for action in action_frequencies}
    #
    #         # Calculate relevance score for each action in the concept (TF * IDF)
    #         relevance_scores = {action: round(relative_frequencies[action] * inverse_concept_frequency[action], 3)
    #                             for action in action_frequencies}
    #
    #         # Sort actions based on their relevance scores in descending order
    #         sorted_relevance_scores = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    #
    #         # Keep the top 4 actions based on frequencies and the top 4 relevant actions
    #         top_frequencies_actions = [action for action, score in
    #                                    sorted(action_frequencies.items(), key=lambda x: x[1], reverse=True)[:4]]
    #         top_relevance_actions = [action for action, score in sorted_relevance_scores[:15]]
    #         relevant_actions_by_concept[concept] = (top_frequencies_actions, top_relevance_actions)
    #
    #     # Find actions that appear in at least half of the top action lists
    #     actions_to_remove = set()
    #     concept_count = len(relevant_actions_by_concept)
    #     action_counts = Counter()
    #     for concept, (_, rel_action_set) in relevant_actions_by_concept.items():
    #         action_counts.update(rel_action_set)
    #
    #     for action, count in action_counts.items():
    #         if count >= 10:
    #             actions_to_remove.add(action)
    #
    #     # Remove common actions from all concepts
    #     for concept in relevant_actions_by_concept:
    #         freq_action_set, rel_action_set = relevant_actions_by_concept[concept]
    #         relevant_actions_by_concept[concept] = (
    #             [action for action in freq_action_set if action not in actions_to_remove],
    #             [action for action in rel_action_set if action not in actions_to_remove]
    #         )
    #
    #     for concept, (freq_action_set, rel_action_set) in relevant_actions_by_concept.items():
    #         print(concept, "has top frequency actions:", freq_action_set)
    #         print(concept, "has top relevance actions:", rel_action_set)
    #
    #     file_path = "top_relevant_jsons/top_relevant_actions_by_concept.json"
    #     # Using json.dump() to write the dictionary to a JSON file
    #     with open(file_path, "w") as json_file:
    #         top_relevant_actions_by_concept = {key: value[1] for key, value in relevant_actions_by_concept.items() if
    #                                            value}
    #         json.dump(top_relevant_actions_by_concept, json_file)
    #         print(top_relevant_actions_by_concept)
    #     return relevant_actions_by_concept

    def find_top_concepts_for_actions_normalized(ACs_list_name, top_n=3):
        actions_by_concept, _ = get_actions_by_concept(ACs_list_name)
        action_concept_scores = defaultdict(dict)

        # Calculate concept scores for each action and normalize by concept size
        concept_sizes = {concept: len(actions_list) for concept, actions_list in actions_by_concept.items()}

        for concept, actions_list in actions_by_concept.items():
            for action in actions_list:
                if action not in action_concept_scores:
                    action_concept_scores[action] = {}
                if concept in action_concept_scores[action]:
                    action_concept_scores[action][concept] += 1 / concept_sizes[concept]
                else:
                    action_concept_scores[action][concept] = 1 / concept_sizes[concept]

        top_concepts_for_actions = {}

        # Find the top N concepts for each action
        for action, concept_scores in action_concept_scores.items():
            sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
            top_concepts = [concept for concept, _ in sorted_concepts[:top_n]]
            top_concepts_for_actions[action] = top_concepts


        result_dict = {concept: [] for concept in
                       set(action for actions in top_concepts_for_actions.values() for action in actions)}

        # Populate the result dictionary
        for action, concepts in top_concepts_for_actions.items():
            for concept in concepts:
                result_dict[concept].append(action)

        # Sort the actions for each concept alphabetically
        for concept, actions in result_dict.items():
            result_dict[concept] = sorted(actions)

        file_path = "top_relevant_jsons/top_relevant_actions_by_concept.json"
        # Using json.dump() to write the dictionary to a JSON file
        with open(file_path, "w") as json_file:
            json.dump(result_dict, json_file)
        print(result_dict)

        return result_dict

    def create_action_wordclouds(ACs_list_name, threshold=0.9):
        actions_by_concept, actions_by_concept_w_strengths = get_actions_by_concept(ACs_list_name)
        font_color = '#0074D9'  # Use any shade of blue you prefer
        helvetica_font = 'Helvetica.ttf'

        for concept, actions_with_strengths in actions_by_concept_w_strengths.items():
            actions = [(action, strength) for action, strength in actions_with_strengths]
            actions = {action: strength for action, strength in actions}

            actions_above_threshold = [(action, strength) for action, strength in actions_with_strengths if
                                       strength >= threshold]
            actions_above_threshold = {action: strength for action, strength in actions_above_threshold}

            wc_actions = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,
                color_func=lambda *args, **kwargs: font_color
            ).generate_from_frequencies(actions)

            wc_actions_1 = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,
                color_func=lambda *args, **kwargs: font_color
            ).generate_from_frequencies(actions_above_threshold)

            # Plot the word clouds side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(wc_actions, interpolation="bilinear")
            axes[0].axis("off")
            axes[0].set_title(f"Actions Word Cloud for: {concept} (no threshold)")

            axes[1].imshow(wc_actions_1, interpolation="bilinear")
            axes[1].axis("off")
            axes[1].set_title(f"Actions Word Cloud for: {concept} ( threshold {threshold})")

            plt.show()
            save_filename = f"output_imgs/perceptual_data/action/wordclouds/action_wordcloud_{concept}_{ACs_list_name}.jpg"
            plt.savefig(save_filename)
        return

    # List of concepts of interest
    concepts_of_interest = ['comfort', 'danger', 'death', 'fitness', 'freedom', 'power', 'safety']
    for ACs_list_name in ACs_list_names:
        # get_action_frequencies(ACs_list_name)
        # plot_action_frequencies(ACs_list_name)
        # find_relevant_actions_normalized(ACs_list_name)
        find_top_concepts_for_actions_normalized(ACs_list_name)
        # create_action_wordclouds(ACs_list_name)
    return

## Human Presence
def stats_hp(ACs_list_names):
    def get_human_presence_by_concept(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        human_presence_by_concept = {concept: [] for concept in concept_images}
        human_presence_by_concept_w_strengths = {concept: [] for concept in concept_images}
        for concept, images in concept_images.items():
            for img in images:
                for image_id, image_info in merged_ARTstract.items():
                    if img == image_id:
                        hp_object = merged_ARTstract[img].get("hp", {}).get("ARTstract_hp_2023_06_26", {})
                        hp = hp_object['human_presence']
                        hp_strength = hp_object['annotation_strength']
                        human_presence_by_concept[concept].append(hp)
                        human_presence_by_concept_w_strengths[concept].append((hp, hp_strength))
        return human_presence_by_concept, human_presence_by_concept_w_strengths

    def calculate_human_presence_frequencies(ACs_list_name):
        human_presence_by_concept, human_presence_by_concept_w_strengths = get_human_presence_by_concept(ACs_list_name)
        human_presence_frequencies_by_concept = {}

        for concept, human_presence_list in human_presence_by_concept.items():
            human_presence_frequencies = Counter(human_presence_list)
            human_presence_frequencies_by_concept[concept] = human_presence_frequencies

        return human_presence_frequencies_by_concept

    def plot_human_presence_frequencies(ACs_list_name):
        human_presence_frequencies_by_concept = calculate_human_presence_frequencies(ACs_list_name)
        num_concepts = len(human_presence_frequencies_by_concept)
        presence_labels = ["False", "True"]
        fig, axs = plt.subplots(num_concepts + 1, 1, figsize=(12, 5 * (num_concepts + 1)), sharex=True)

        # Calculate the total number of images for each concept
        total_images_by_concept = {concept: sum(presence_frequencies.values()) for concept, presence_frequencies in
                                   human_presence_frequencies_by_concept.items()}

        max_y_value = 100  # Set the maximum y-axis value to 100 to represent percentages

        # Calculate the total number of images across all concepts
        total_images = sum(total_images_by_concept.values())

        # Calculate the total human presence frequencies subplot as a percentage of total images
        frequencies_total = [
            sum(presence_frequencies.get(presence_label, 0) for presence_frequencies in
                human_presence_frequencies_by_concept.values()) / total_images * 100
            for presence_label in presence_labels]

        axs[0].bar(presence_labels, frequencies_total)
        axs[0].set_title("Overall Human Presence Frequencies")
        axs[0].set_ylabel("Percentage of Images")
        axs[0].tick_params(axis='x', rotation=45, labelsize=8)
        axs[0].set_ylim(0, max_y_value)  # Set y-axis limits to 0-100 for percentage scale

        for i, (concept, presence_frequencies) in enumerate(human_presence_frequencies_by_concept.items(), start=1):
            frequencies = [presence_frequencies.get(presence_label, 0) / total_images_by_concept[concept] * 100 for
                           presence_label in presence_labels]
            axs[i].bar(presence_labels, frequencies)
            axs[i].set_title(f"Human Presence Frequencies for Concept '{concept}'")
            axs[i].set_ylabel("Percentage of Images")
            axs[i].tick_params(axis='x', rotation=45, labelsize=8)
            axs[i].set_ylim(0, max_y_value)  # Set consistent y-axis limits

        plt.tight_layout()
        plt.show()

        # Save the plot as an image
        save_filename = f"output_imgs/perceptual_data/human_presence/human_presence_frequencies_{ACs_list_name}.jpg"
        plt.savefig(save_filename)
        plt.show()

    # List of concepts of interest
    concepts_of_interest = ['comfort', 'danger', 'death', 'fitness', 'freedom', 'power', 'safety']
    for ACs_list_name in ACs_list_names:
        plot_human_presence_frequencies(ACs_list_name)

    return

# Execution input examples
# ACs_list_names = ["ARTstract_ACs_lists", "Balanced_ARTstract_ACs_lists"]
ACs_list_names = ["ARTstract_ACs_lists"]
dataset_colors = ['#00BFFF', '#FF6F61', '#9370DB', '#2E8B57']
concept_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
plot_type = "bar"


############# EXECUTION

### Concept frequencies over different source datasets
# stats_concept_frequencies(ACs_list_names, dataset_colors, concept_colors)

### Evocation strengths
# stats_evocation_strengths(ACs_list_names, dataset_colors, concept_colors, plot_type)

### Detected objects
# stats_num_detected_objects(ACs_list_names, dataset_colors, concept_colors, plot_type)
# stats_detected_objects(ACs_list_names)
object_co_occurence_heatmaps(ACs_list_names, consider_person=False)

### Image captions
# stats_image_captions(ACs_list_names)
# caption_words_co_occurences(ACs_list_names, consider_painting=False)

### Top colors
# stats_top_colors(ACs_list_names, filter_grays_out=True)
# colors_co_occurrences(ACs_list_names, filter_grays_out=False)

### Emotions
# stats_emotions(ACs_list_names)

### Ages
# stats_age(ACs_list_names)

### Art Style
# stats_art_style(ACs_list_names)

### Actions
# stats_action(ACs_list_names)

## Human Presence
# stats_hp(ACs_list_names)