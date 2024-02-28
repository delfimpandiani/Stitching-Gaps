import json
from rdflib import Graph, Namespace, Literal, URIRef, XSD
from rdflib.namespace import XSD

# Mapping of codes to (image) annotation types
annotation_type_tuples = [("as", "ArtStyle"), ("act", "Action"), ("age","Age"), ("color", "Color"), ("em", "Emotion"), ("ic", "ImageCaption"), ("hp", "HumanPresence"), ("od", "Object")]

# Define namespaces for your prefixes
base = "https://w3id.org/situannotate#"
rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
xsd = Namespace("http://www.w3.org/2001/XMLSchema#")
situannotate = Namespace("https://w3id.org/situannotate#")

# Create an RDF graph
g = Graph()

### Create triples for Annotation Situation Descriptions
g.add((situannotate["ac_visual_evocation"], rdf.type, situannotate.ImageAnnotationDescription))

## add comments for the Annotation Situation Descriptions
g.add((situannotate["act_detection_desc"], rdfs.comment, Literal("Action detections are annotation situations in which annotations play the role of detected_action, assigned by an Annotator according to a certain detection threshold or heuristic", datatype=XSD.string)))
g.add((situannotate["age_detection_desc"], rdfs.comment, Literal("Age detections are annotation situations in which annotations play the role of detected_age, assigned by an Annotator according to a certain detection threshold or heuristic", datatype=XSD.string)))
g.add((situannotate["as_detection_desc"], rdfs.comment, Literal("Art style detections are annotation situations in which annotations play the role of detected_art_style, assigned by an Annotator according to a certain detection threshold or heuristic", datatype=XSD.string)))
g.add((situannotate["color_detection_desc"], rdfs.comment, Literal("Color detections are annotation situations in which annotations play the role of detected_color, assigned by an Annotator according to a certain detection threshold or heuristic", datatype=XSD.string)))
g.add((situannotate["em_detection_desc"], rdfs.comment, Literal("Emotion detections are annotation situations in which annotations play the role of detected_emotion, assigned by an Annotator according to a certain detection threshold or heuristic", datatype=XSD.string)))
g.add((situannotate["ic_detection_desc"], rdfs.comment, Literal("Image caption detections are annotation situations in which annotations play the role of detected_image_caption, assigned by an Annotator according to a certain detection threshold or heuristic", datatype=XSD.string)))
g.add((situannotate["hp_detection_desc"], rdfs.comment, Literal("Human presence detections are annotation situations in which annotations play the role of detected_human_presence, assigned by an Annotator according to a certain detection threshold or heuristic", datatype=XSD.string)))
g.add((situannotate["od_detection_desc"], rdfs.comment, Literal("Object detections are annotation situations in which annotations play the role of detected_object, assigned by an Annotator according to a certain detection threshold or heuristic", datatype=XSD.string)))

### Create triples for Image Annotation Roles
g.add((situannotate["detected_action"], rdf.type, situannotate.ImageAnnotationRole))
g.add((situannotate["detected_age"], rdf.type, situannotate.ImageAnnotationRole))
g.add((situannotate["detected_art_style"], rdf.type, situannotate.ImageAnnotationRole))
g.add((situannotate["detected_color"], rdf.type, situannotate.ImageAnnotationRole))
g.add((situannotate["detected_emotion"], rdf.type, situannotate.ImageAnnotationRole))
g.add((situannotate["detected_image_caption"], rdf.type, situannotate.ImageAnnotationRole))
g.add((situannotate["detected_human_presence"], rdf.type, situannotate.ImageAnnotationRole))
g.add((situannotate["detected_object"], rdf.type, situannotate.ImageAnnotationRole))

### Create triples for Annotation Situation Descriptions
g.add((situannotate["act_detection_desc"], situannotate.defines, situannotate["detected_action"]))
g.add((situannotate["age_detection_desc"], situannotate.defines, situannotate["detected_age"]))
g.add((situannotate["as_detection_desc"], situannotate.defines, situannotate["detected_art_style"]))
g.add((situannotate["color_detection_desc"], situannotate.defines, situannotate["detected_color"]))
g.add((situannotate["em_detection_desc"], situannotate.defines, situannotate["detected_emotion"]))
g.add((situannotate["ic_detection_desc"], situannotate.defines, situannotate["detected_image_caption"]))
g.add((situannotate["hp_detection_desc"], situannotate.defines, situannotate["detected_human_presence"]))
g.add((situannotate["od_detection_desc"], situannotate.defines, situannotate["detected_object"]))

### Create triples for Annotation Situations
with open('input/real-annotation-situations.json', 'r') as json_file:
    # Load the JSON data into a Python dictionary
    data = json.load(json_file)

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

        # Create RDF triples
        g.add((situannotate[annotated_dataset], rdf.type, situannotate.Dataset))
        g.add((situannotate[annotation_place], rdf.type, situannotate.Country))

        if annotator_type == "artificial":
            g.add((situannotate[annotator_name], rdf.type, situannotate.ArtificialAnnotator))
            g.add((situannotate[annotator_name], situannotate.hasModelArchitecture, situannotate[model_architecture]))
            g.add((situannotate[annotator_name], situannotate.pretrainedOnDataset, situannotate[dataset_of_pretraining]))
            g.add((situannotate[model_architecture], rdf.type, situannotate.ModelArchitecture))
            g.add((situannotate[dataset_of_pretraining], rdf.type, situannotate.Dataset))

        elif annotator_type == "individual human annotator":
            g.add((situannotate[annotator_name], rdf.type, situannotate.IndividualHumanAnnotator))

        elif annotator_type == "human annotator community":
            g.add((situannotate[annotator_name], rdf.type, situannotate.HumanAnnotatorCommunity))

        annotation_type_value = next(value for key, value in annotation_type_tuples if key == annotation_type)
        g.add((situannotate[situation_name], rdf.type, situannotate[annotation_type_value + "AnnotationSituation"]))
        g.add((situannotate[situation_name], situannotate.satisfies, situannotate[satisfies_description]))
        g.add((situannotate[situation_name], situannotate.involvesAnnotator, situannotate[annotator_name]))
        g.add((situannotate[situation_name], situannotate.involvesDataset, situannotate[annotated_dataset]))

        g.add((situannotate[situation_name], situannotate.atPlace, situannotate[annotation_place]))


        g.add((situannotate[situation_name], situannotate.onDate, Literal(annotation_time, datatype=XSD.date)))
        g.add((situannotate[situation_name], situannotate.hasDetectionThreshold, Literal(detection_threshold)))

# Serialize the RDF graph to Turtle format
turtle_data = g.serialize(format="turtle")

# Print the Turtle data
print(turtle_data)

# Save the Turtle RDF data to a file
with open("output/real_situations_kg.ttl", "w") as outfile:  # Open in regular text mode (not binary mode)
    outfile.write(turtle_data)

