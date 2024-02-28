import json

# manually done descriptions

## art style description
# print(f":as_detection_desc rdf:type :ImageAnnotationDescription ;")
# print(f"                        rdfs:comment 'Art style detections are annotation situations in which annotations play the role of detected_art_style, assigned by an Annotator according to a certain detection threshold or heuristic'^^xsd:string ;")
# print(f"                        :defines detected_art_style .")
# print(f":detected_art_style rdf:type :ImageAnnotationRole .")

annotation_type_tuples = [("as", "ArtStyle"), ("act", "Action"), ("age","Age"), ("color", "Color"), ("em", "Emotion"), ("ic", "ImageCaption"), ("hp", "HumanPresence"), ("od", "Object")]

prefixes = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "": "https://w3id.org/situannotate#"
}
# Define the TTL file path
ttl_file_path = "output.ttl"

with open('toy-real-annotation-situations.json', 'r') as json_file:
    # Load the JSON data into a Python dictionary
    data = json.load(json_file)

    # Create and open the TTL file for writing
    with open(ttl_file_path, "w") as ttl_file:
        seen_triples = set()
        for prefix, namespace_uri in prefixes.items():
            ttl_file.write(f"@prefix {prefix}: <{namespace_uri}> .\n")
        for situation_name, details in data.items():
            annotation_type = details["annotation_type"]
            annotator_type = details["annotator_type"]
            annotator_name = details["annotator"].replace("/", "_")  # Convert '/' to '_'
            annotated_dataset = details["annotated_dataset"]
            model_architecture = details["annotator_model_arch"].replace(" ", "_")
            dataset_of_pretraining = details["annotator_pretrained_on"]
            annotation_place = details["annotation_place"]
            annotation_time = details["annotation_time"].replace("_", "-")
            detection_threshold = details["detection_threshold"].replace(" ", "_")
            satisfies_description = details["satisfies_description"]

            # Write RDF triples to the TTL file
            ttl_file.write(f":{annotated_dataset} rdf:type :Dataset .\n")
            ttl_file.write(f":{annotation_place} rdf:type :Country .\n")

            # Create RDF triples based on annotator type
            if annotator_type == "artificial":
                ttl_file.write(f":{annotator_name} rdf:type :ArtificialAnnotator ;\n")
                ttl_file.write(f"                            :hasModelArchitecture :{model_architecture} ;\n")
                ttl_file.write(f"                            :pretrainedOnDataset :{dataset_of_pretraining} .\n")
                ttl_file.write(f":{model_architecture} rdf:type :ModelArchitecture .\n")
                ttl_file.write(f":{dataset_of_pretraining} rdf:type :Dataset .\n")

            elif annotator_type == "individual human annotator":
                ttl_file.write(f":{annotator_name} rdf:type :IndividualHumanAnnotator ;\n")

            elif annotator_type == "human annotator community":
                ttl_file.write(f":{annotator_name} rdf:type :HumanAnnotatorCommunity ;\n")

            annotation_type_value = next(value for key, value in annotation_type_tuples if key == annotation_type)
            # Write the remaining RDF triples to the TTL file
            ttl_file.write(f":{situation_name} rdf:type :{annotation_type_value}AnnotationSituation ;\n")
            ttl_file.write(f"                        :satisfies :{satisfies_description} ;\n")
            ttl_file.write(f"                        :involvesAnnotator :{annotator_name} ;\n")
            ttl_file.write(f"                        :involvesDataset :{annotated_dataset} ;\n")
            ttl_file.write(f"                        :atPlace :{annotation_place} ;\n")
            ttl_file.write(f'                        :onDate "{annotation_time}"^^xsd:date ;\n')
            ttl_file.write(f'                        :hasDetectionThreshold "{detection_threshold}"^^xsd:string .\n')

    print(f"Triples have been saved to {ttl_file_path}")






# for situation_name, details in data.items():
#     annotation_type = details["annotation_type"]
#     annotator_type = details["annotator_type"]
#     annotator_name = details["annotator"].replace("/", "_")  # Convert '/' to '_'
#     annotated_dataset = details["annotated_dataset"]
#     model_architecture = details["annotator_model_arch"].replace(" ", "_")
#     dataset_of_pretraining = details["annotator_pretrained_on"]
#     annotation_place = details["annotation_place"]
#     annotation_time = details["annotation_time"]
#     detection_threshold = details["detection_threshold"].replace(" ", "_")
#     satisfies_description = details["satisfies_description"]
#
#     print(f":{annotated_dataset} rdf:type :Dataset .")
#     print(f":{annotation_place} rdf:type :Country .")
#
#     # Create RDF triples based on annotator type
#     if annotator_type == "artificial":
#         print(f":{annotator_name} rdf:type :ArtificialAnnotator ;")
#         print(f"                            :hasModelArchitecture :{model_architecture} ;")
#         print(f"                            :pretrainedOnDataset :{dataset_of_pretraining} .")
#         print(f":{model_architecture} rdf:type :ModelArchitecture .")
#         print(f":{dataset_of_pretraining} rdf:type :Dataset .")
#
#     elif annotator_type == "individual human annotator":
#         print(f":{annotator_name} rdf:type :IndividualHumanAnnotator ;")
#
#     elif annotator_type == "human annotator community":
#         print(f":{annotator_name} rdf:type :HumanAnnotatorCommunity ;")
#
#     # Find the matching tuple for annotation type and get the corresponding value
#     annotation_type_value = next(value for key, value in annotation_type_tuples if key == annotation_type)
#     print(f":{situation_name} rdf:type :{annotation_type_value}AnnotationSituation ;")
#     print(f"                        :satisfies :{satisfies_description} ;")
#     print(f"                        :involvesAnnotator :{annotator_name} ;")
#     print(f"                        :involvesDataset :{annotated_dataset} ;")
#     print(f"                        :atPlace :{annotation_place} ;")
#     print(f'                        :onDate "{annotation_time}"^^xsd:date ;')
#     print(f'                        :hasDetectionThreshold "{detection_threshold}"^^xsd:string')
