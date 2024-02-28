import json
from rdflib import Graph, Namespace, Literal, URIRef, XSD
from rdflib.namespace import XSD

# Mapping of codes to (image) annotation types
annotation_type_tuples = [("as", "ArtStyle"), ("acve", "ACVisualEvocation"), ("act", "Action"), ("age","Age"), ("color", "Color"), ("em", "Emotion"), ("ic", "ImageCaption"), ("hp", "HumanPresence"), ("od", "Object")]

# Define namespaces for your prefixes
base = "https://w3id.org/situannotate#"
rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
xsd = Namespace("http://www.w3.org/2001/XMLSchema#")
situannotate = Namespace("https://w3id.org/situannotate#")
artstract = Namespace("https://w3id.org/artstract#")

# Create an RDF graph
g = Graph()

### Create triples for Annotation Situation Descriptions
g.add((situannotate["ac_visual_evocation"], rdf.type, situannotate.ImageAnnotationDescription))

## add comments for the Annotation Situation Descriptions
g.add((situannotate["ac_visual_evocation"], rdfs.comment, Literal("Abstract Concept Visual Evocation (ACVE) situations are annotation situations in which annotations play the role of evoked_abstract_concept, assigned by an Annotator, according to a certain detection threshold or heuristic", datatype=XSD.string)))

### Create triples for Image Annotation Roles
g.add((situannotate["evoked_abstract_concept"], rdf.type, situannotate.ImageAnnotationRole))

### Create triples for Annotation Situation Descriptions
g.add((situannotate["ac_visual_evocation"], situannotate.defines, situannotate["evoked_abstract_concept"]))

### Create triples for Annotation Situations
with open('input/real-acve-annotation-situations.json', 'r') as json_file:
    # Load the JSON data into a Python dictionary
    data = json.load(json_file)

    for situation_name, details in data.items():
        annotation_type = details["annotation_type"]
        annotator_type = details["annotator_type"]
        annotator_name = details["annotator"].replace("/", "_")  # Convert '/' to '_'
        annotated_dataset = details["annotated_dataset"]
        annotation_place = details["annotation_place"]
        annotation_time = details["annotation_time"].replace("_", "-")
        detection_threshold = details["detection_threshold"].replace(" ", "_")
        satisfies_description = details["satisfies_description"]

        # Create RDF triples
        g.add((situannotate[annotated_dataset], rdf.type, situannotate.Dataset))
        g.add((situannotate[annotation_place], rdf.type, situannotate.Country))

        if annotator_type == "individual human annotator":
            g.add((situannotate[annotator_name], rdf.type, situannotate.IndividualHumanAnnotator))

        elif annotator_type == "human annotator community":
            g.add((situannotate[annotator_name], rdf.type, situannotate.HumanAnnotatorCommunity))

        annotation_type_value = next(value for key, value in annotation_type_tuples if key == annotation_type)
        g.add((situannotate[situation_name], rdf.type, situannotate[annotation_type_value + "AnnotationSituation"]))
        g.add((situannotate[annotation_type_value + "AnnotationSituation"], rdf.subClassOf, situannotate.ImageAnnotationSituation))
        g.add((situannotate[situation_name], situannotate.satisfies, situannotate[satisfies_description]))
        g.add((situannotate[situation_name], situannotate.involvesAnnotator, situannotate[annotator_name]))
        g.add((situannotate[situation_name], situannotate.involvesDataset, situannotate[annotated_dataset]))

        g.add((situannotate[situation_name], situannotate.atPlace, situannotate[annotation_place]))


        g.add((situannotate[situation_name], situannotate.onDate, Literal(annotation_time)))
        g.add((situannotate[situation_name], situannotate.hasDetectionThreshold, Literal(detection_threshold)))

# Serialize the RDF graph to Turtle format
turtle_data = g.serialize(format="turtle")

# Print the Turtle data
print(turtle_data)

# Save the Turtle RDF data to a file
with open("output/real_acve_situations_kg.ttl", "w") as outfile:  # Open in regular text mode (not binary mode)
    outfile.write(turtle_data)

