import json
from rdflib import Graph, Namespace, Literal, URIRef, XSD
from rdflib.namespace import XSD

# Mapping of codes to (image) annotation types
annotation_codes_classes = [("as", "ArtStyle"), ("act", "Action"), ("age","Age"), ("color", "Color"), ("em", "Emotion"), ("ic", "ImageCaption"), ("hp", "HumanPresence"), ("od", "Object")]
annotation_codes_jsonnames = [("as", "art_style"), ("act", "action_label"), ("age","age_tier"), ("color", "webcolor_name"), ("em", "emotion"), ("ic", "image_description"), ("hp", "human_presence"), ("od", "detected_object")]
annotation_codes_roles = [("as", "detected_art_style"), ("act", "detected_action"), ("age","detected_age"), ("color", "detected_color"), ("em", "detected_emotion"), ("ic", "detected_image_caption"), ("hp", "detected_human_presence"), ("od", "detected_object")]

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
with open('input/toy-image-data.json', 'r') as json_file:
    # Load the JSON data into a Python dictionary
    data = json.load(json_file)

    for image_id, details in data.items():
        image_instance = str(dataset + "_" + image_id)
        g.add((situannotate[image_instance], rdf.type, situannotate.Image))

        for annotation_code, value in details.items():
            annotation_class = next((tup[1] for tup in annotation_codes_classes if annotation_code == tup[0]), None)
            annotation_class = str(annotation_class) + "Annotation"
            annotation_jsonname = next((tup[1] for tup in annotation_codes_jsonnames if annotation_code == tup[0]),
                                       None)
            annotation_role = next((tup[1] for tup in annotation_codes_roles if annotation_code == tup[0]), None)
            if annotation_code in ["act", "age", "as", "em", "hp"]:
                for situation_name, sit_details in value.items():
                    annotation_id = image_id + "_" + situation_name
                    lexical_label = sit_details[annotation_jsonname]
                    lexical_entry = str("le_" + lexical_label)
                    typing_concept = sit_details["conceptnet_concept"].split(":")[1]
                    annotation_id = str(image_id + "_" + situation_name )
                    annotation_strength = sit_details["annotation_strength"]

                    # declare triple between the image and the annotation situation
                    g.add((situannotate[image_instance], situannotate.isInvolvedInAnnotationSituation, situannotate[situation_name]))

                    # triples for each annotation
                    g.add((situannotate[annotation_id], rdf.type, situannotate[annotation_class]))

                    g.add((situannotate[annotation_id], situannotate.isAnnotationInvolvedInSituation, situannotate[situation_name]))
                    g.add((situannotate[annotation_id], situannotate.isClassifiedBy, situannotate[annotation_role]))
                    g.add((situannotate[annotation_id], situannotate.aboutAnnotatedEntity, situannotate[image_instance]))
                    g.add((situannotate[annotation_id], situannotate.hasAnnotationStrength, Literal(annotation_strength, datatype=XSD.decimal)))
                    g.add((situannotate[annotation_id], situannotate.typedByConcept, conceptnet[typing_concept]))
                    g.add((situannotate[annotation_id], situannotate.annotationWithLexicalEntry, situannotate[lexical_entry]))

                    # triples for each lexical entry
                    g.add((situannotate[lexical_entry], rdf.type, situannotate.LexicalEntry))
                    g.add((situannotate[lexical_entry], situannotate.typedByConcept, conceptnet[typing_concept]))
                    g.add((situannotate[lexical_entry], rdfs.label, Literal(lexical_label, datatype=XSD.string)))

                    # triples for image in relation to annotation
                    g.add((situannotate[image_instance], situannotate.isAnnotatedWithLexicalEntry, situannotate[lexical_entry]))
                    g.add((situannotate[image_instance], situannotate.hasImageLabelTypedBy, conceptnet[typing_concept]))

            elif annotation_code == "od":
                for situation_name, value in value.items():
                    detected_objects_list = value["detected_objects"]
                    count = 1
                    for detected_object in detected_objects_list:
                        annotation_id = image_id + "_" + situation_name + "_" + str(count)
                        count += 1
                        lexical_label = detected_object["detected_object"]
                        lexical_entry = str("le_" + lexical_label)
                        typing_concept = lexical_label
                        annotation_strength = detected_object["annotation_strength"]

                        # declare triple between the image and the annotation situation
                        g.add((situannotate[image_instance], situannotate.isInvolvedInAnnotationSituation, situannotate[situation_name]))

                        # triples for each annotation
                        g.add((situannotate[annotation_id], rdf.type, situannotate[annotation_class]))

                        g.add((situannotate[annotation_id], situannotate.isAnnotationInvolvedInSituation, situannotate[situation_name]))
                        g.add((situannotate[annotation_id], situannotate.isClassifiedBy, situannotate[annotation_role]))
                        g.add((situannotate[annotation_id], situannotate.aboutAnnotatedEntity, situannotate[image_instance]))
                        g.add((situannotate[annotation_id], situannotate.hasAnnotationStrength, Literal(annotation_strength, datatype=XSD.decimal)))
                        g.add((situannotate[annotation_id], situannotate.typedByConcept, conceptnet[typing_concept]))
                        g.add((situannotate[annotation_id], situannotate.annotationWithLexicalEntry, situannotate[lexical_entry]))

                        # triples for each lexical entry
                        g.add((situannotate[lexical_entry], rdf.type, situannotate.LexicalEntry))
                        g.add((situannotate[lexical_entry], situannotate.typedByConcept, conceptnet[typing_concept]))
                        g.add((situannotate[lexical_entry], rdfs.label, Literal(lexical_label, datatype=XSD.string)))

                        # triples for image in relation to annotation
                        g.add((situannotate[image_instance], situannotate.isAnnotatedWithLexicalEntry, situannotate[lexical_entry]))
                        g.add((situannotate[image_instance], situannotate.hasImageLabelTypedBy, conceptnet[typing_concept]))

            elif annotation_code == "color":
                for situation_name, detected_colors_list in value.items():
                    count = 1
                    for detected_color in detected_colors_list:
                        annotation_id = image_id + "_" + situation_name + "_" + str(count)
                        count += 1
                        lexical_label = detected_color["webcolor_name"]
                        lexical_entry = str("le_" + lexical_label)
                        typing_concept = detected_color["conceptnet_concept"].split(":")[1]
                        rgb_coordinate_red = detected_color["rgb"][0]
                        rgb_coordinate_green = detected_color["rgb"][1]
                        rgb_coordinate_blue = detected_color["rgb"][2]

                        # declare triple between the image and the annotation situation
                        g.add((situannotate[image_instance], situannotate.isInvolvedInAnnotationSituation, situannotate[situation_name]))

                        # triples for each annotation
                        g.add((situannotate[annotation_id], rdf.type, situannotate[annotation_class]))

                        g.add((situannotate[annotation_id], situannotate.isAnnotationInvolvedInSituation, situannotate[situation_name]))
                        g.add((situannotate[annotation_id], situannotate.isClassifiedBy, situannotate[annotation_role]))
                        g.add((situannotate[annotation_id], situannotate.aboutAnnotatedEntity, situannotate[image_instance]))
                        g.add((situannotate[annotation_id], situannotate.typedByConcept, conceptnet[typing_concept]))
                        g.add((situannotate[annotation_id], situannotate.annotationWithLexicalEntry, situannotate[lexical_entry]))

                        g.add((situannotate[annotation_id], situannotate.rgbCoordinateRed, Literal(rgb_coordinate_red, datatype=XSD.nonNegativeInteger)))
                        g.add((situannotate[annotation_id], situannotate.rgbCoordinateGreen, Literal(rgb_coordinate_green, datatype=XSD.nonNegativeInteger)))
                        g.add((situannotate[annotation_id], situannotate.rgbCoordinateBlue, Literal(rgb_coordinate_blue, datatype=XSD.nonNegativeInteger)))

                        # triples for each lexical entry
                        g.add((situannotate[lexical_entry], rdf.type, situannotate.LexicalEntry))
                        g.add((situannotate[lexical_entry], situannotate.typedByConcept, conceptnet[typing_concept]))
                        g.add((situannotate[lexical_entry], rdfs.label, Literal(lexical_label, datatype=XSD.string)))

                        # triples for image in relation to annotation
                        g.add((situannotate[image_instance], situannotate.isAnnotatedWithLexicalEntry, situannotate[lexical_entry]))
                        g.add((situannotate[image_instance], situannotate.hasImageLabelTypedBy, conceptnet[typing_concept]))

            elif annotation_code == "ic":
                for situation_name, caption_deets in value.items():
                    annotation_id = image_id + "_" + situation_name
                    image_caption = caption_deets["image_description"]

                    # declare triple between the image and the annotation situation
                    g.add((situannotate[image_instance], situannotate.isInvolvedInAnnotationSituation, situannotate[situation_name]))

                    # triples for each annotation
                    g.add((situannotate[annotation_id], rdf.type, situannotate[annotation_class]))
                    g.add((situannotate[annotation_id], rdf.type, situannotate[annotation_class]))
                    g.add((situannotate[annotation_id], situannotate.isAnnotationInvolvedInSituation, situannotate[situation_name]))
                    g.add((situannotate[annotation_id], situannotate.isClassifiedBy, situannotate[annotation_role]))
                    g.add((situannotate[annotation_id], situannotate.aboutAnnotatedEntity, situannotate[image_instance]))

                    g.add((situannotate[annotation_id], rdfs.comment, Literal(image_caption, datatype=XSD.string)))
                    g.add((situannotate[image_instance], rdfs.comment, Literal(image_caption, datatype=XSD.string)))

# Serialize the RDF graph to Turtle format
turtle_data = g.serialize(format="turtle")

# Print the Turtle data
print(turtle_data)

# Save the Turtle RDF data to a file
with open("output/images_kg.ttl", "w") as outfile:  # Open in regular text mode (not binary mode)
    outfile.write(turtle_data)

