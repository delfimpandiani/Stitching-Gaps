import json
from rdflib import Graph, Namespace, Literal, URIRef, XSD
from rdflib.namespace import XSD

# Mapping of codes to (image) annotation types
annotation_codes_classes = [("evoked_clusters", "ACVisualEvocation"), ("as", "ArtStyle"), ("act", "Action"), ("age","Age"), ("color", "Color"), ("em", "Emotion"), ("ic", "ImageCaption"), ("hp", "HumanPresence"), ("od", "Object")]
annotation_codes_jsonnames = [("evoked_clusters", "evoked_abstract_concept"), ("as", "art_style"), ("act", "action_label"), ("age","age_tier"), ("color", "webcolor_name"), ("em", "emotion"), ("ic", "image_description"), ("hp", "human_presence"), ("od", "detected_object")]
annotation_codes_roles = [("acve", "evoked_abstract_concept"), ("as", "detected_art_style"), ("act", "detected_action"), ("age","detected_age"), ("color", "detected_color"), ("em", "detected_emotion"), ("ic", "detected_image_caption"), ("hp", "detected_human_presence"), ("od", "detected_object")]


# Define namespaces for your prefixes
base = "https://w3id.org/situannotate#"
rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
xsd = Namespace("http://www.w3.org/2001/XMLSchema#")
situannotate = Namespace("https://w3id.org/situannotate#")
conceptnet = Namespace("http://etna.istc.cnr.it/framester2/conceptnet/5.7.0/c/en/")

# Create an RDF graph
g = Graph()

dataset = "ARTstract"
### Create triples for Annotation Situations
with open('input/real-img-data.json', 'r') as json_file:
    # Load the JSON data into a Python dictionary
    data = json.load(json_file)

    for image_id, details in data.items():
        image_instance = str(dataset + "_" + image_id)
        g.add((situannotate[image_instance], rdf.type, situannotate.Image))
        source_dataset = details["source_dataset"]
        source_id = details["source_id"]
        evoked_clusters = details['evoked_clusters']
        first_cluster = next(iter(evoked_clusters.values()))

        annotation_class = "ACVisualEvocation"
        annotation_class = str(annotation_class) + "Annotation"
        situation_name = source_dataset + "_acve"
        annotation_role = "evoked_abstract_concept"
        annotation_id = image_instance + "_" + situation_name
        cluster_name = first_cluster["cluster_name"]
        evocation_context = first_cluster["evocation_context"]

        # declare triple between the image and the annotation situation
        g.add((situannotate[image_instance], situannotate.isInvolvedInAnnotationSituation, situannotate[situation_name]))

        # triples for each annotation
        g.add((situannotate[annotation_id], rdf.type, situannotate[annotation_class]))

        g.add((situannotate[annotation_id], situannotate.isAnnotationInvolvedInSituation, situannotate[situation_name]))
        g.add((situannotate[annotation_id], situannotate.isClassifiedBy, situannotate[annotation_role]))
        g.add((situannotate[annotation_id], situannotate.aboutAnnotatedEntity, situannotate[image_instance]))
        g.add((situannotate[annotation_id], situannotate.typedByConcept, conceptnet[cluster_name]))
        g.add((situannotate[annotation_id], situannotate.annotationWithLexicalEntry, situannotate[cluster_name]))
        g.add((situannotate[annotation_id], situannotate.annotationWithEvocationContext, Literal(evocation_context, datatype=XSD.string)))

        # triples for each lexical entry
        g.add((situannotate[cluster_name], rdf.type, situannotate.LexicalEntry))
        g.add((situannotate[cluster_name], situannotate.typedByConcept, conceptnet[cluster_name]))
        g.add((situannotate[cluster_name], rdfs.label, Literal(cluster_name, datatype=XSD.string)))

        # triples for image in relation to annotation
        g.add((situannotate[image_instance], situannotate.isAnnotatedWithLexicalEntry, situannotate[cluster_name]))
        g.add((situannotate[image_instance], situannotate.hasImageLabelTypedBy, conceptnet[cluster_name]))

# Serialize the RDF graph to Turtle format
turtle_data = g.serialize(format="turtle")

# Print the Turtle data
print(turtle_data)

# Save the Turtle RDF data to a file
with open("output/real_images_acve_kg.ttl", "w") as outfile:  # Open in regular text mode (not binary mode)
    outfile.write(turtle_data)

