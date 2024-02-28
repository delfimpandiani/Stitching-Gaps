import json
from rdflib import Graph, Namespace, Literal, URIRef, XSD
from rdflib.namespace import XSD
import os
def make_img_frame_kg():
    # Define namespaces for your prefixes
    base = "https://w3id.org/situannotate#"
    rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    xsd = Namespace("http://www.w3.org/2001/XMLSchema#")
    situannotate = Namespace("https://w3id.org/situannotate#")
    conceptnet = Namespace("http://etna.istc.cnr.it/framester2/conceptnet/5.7.0/c/en/")
    wn30instances = Namespace("http://etna.istc.cnr.it/framester2/wn/wn30/instances/")
    frame = Namespace("https://w3id.org/framester/data/framestercore/")
    be = Namespace("http://www.ontologydesignpatterns.org/ont/emotions/BasicEmotions.owl")
    mft = Namespace("https://w3id.org/spice/SON/HaidtValues#")
    folk = Namespace("http://www.ontologydesignpatterns.org/ont/values/FolkValues.owl#")
    bhv = Namespace("https://w3id.org/spice/SON/SchwartzValues#")


    # Create an RDF graph
    g = Graph()

    dataset = "ARTstract"

    import csv

    # Input CSV file path
    input_file_path = "frame_output.csv"

    # Open the CSV file for reading
    with open(input_file_path, "r", newline="") as csvfile:
        csvreader = csv.reader(csvfile)
        # Iterate through each row in the CSV file
        next(csvreader)

        for row in csvreader:
            # Access the contents of each column by index
            image_id = row[0]
            image_description = row[1]
            frames = row[2].split(", ")
            emotions = row[3].split(", ")
            values = row[4].split(", ")
            triggers = row[5].split(", ")

            image_instance = str(dataset + "_" + image_id)
            g.add((situannotate[image_instance], rdf.type, situannotate.Image))
            # print(frames)
            # print(emotions)
            # print(values)
            # print(triggers)

            for trigger_synset in triggers:
                trigger_synset = trigger_synset.replace("wn30instances:", "")
                g.add((situannotate[image_instance], situannotate.hasImageCaptionTypedBy, wn30instances[trigger_synset]))

            for evoked_frame in frames:
                evoked_frame = evoked_frame.replace("frame:", "")
                g.add((situannotate[image_instance], situannotate.hasImageCaptionTypedByFrame, frame[evoked_frame]))

            for evoked_em in emotions:
                evoked_em = evoked_em.replace("be:", "")
                g.add((situannotate[image_instance], situannotate.hasImageCaptionTypedByFrame, be[evoked_em]))

            for evoked_value in values:
                if "bhv" in evoked_value:
                    evoked_value = evoked_value.replace("bhv:", "")
                    g.add((situannotate[image_instance], situannotate.hasImageCaptionTypedByFrame, bhv[evoked_value]))
                if "mft" in evoked_value:
                    evoked_value = evoked_value.replace("mft:", "")
                    g.add((situannotate[image_instance], situannotate.hasImageCaptionTypedByFrame, mft[evoked_value]))
                if "folk" in evoked_value:
                    evoked_value = evoked_value.replace("folk:", "")
                    g.add((situannotate[image_instance], situannotate.hasImageCaptionTypedByFrame, folk[evoked_value]))

    # # Serialize the RDF graph to Turtle format
    turtle_data = g.serialize(format="turtle")

    # Print the Turtle data
    print(turtle_data)

    # Save the Turtle RDF data to a file
    with open("image_frames_kg.ttl", "w") as outfile:  # Open in regular text mode (not binary mode)
        outfile.write(turtle_data)


def find_replace_and_save(input_file):
    find_str = "ns1"
    replace_str = "situannotate"

    with open(input_file, "r") as input_f:
        data = input_f.read()
        updated_data = data.replace(find_str, replace_str)

    with open(input_file, "w") as output_f:
        output_f.write(updated_data)
    return

def merge_two_kgs(kgs_paths):

    # Create an empty RDFLib Graph to hold the merged data
    merged_graph = Graph()

    # Dictionary to store the number of triples in each updated graph
    num_triples_dict = {}
    # Perform find and replace for each file pair
    for kg_path in kgs_paths:
        find_replace_and_save(kg_path)
        g = Graph()
        g.parse(kg_path, format="turtle")
        merged_graph += g
        num_triples_dict[kg_path] = len(g)

    # Serialize the merged data to Turtle format
    merged_ttl = merged_graph.serialize(format="turtle").encode("utf-8")

    # Save the merged data to a new Turtle file
    merged_filename = "ARTstract_knowledge_graph.ttl"
    with open(merged_filename, "wb") as output_file:
        output_file.write(merged_ttl)

    # Print the number of triples in each updated graph and the final merged graph
    for input_file, num_triples in num_triples_dict.items():
        print(f"Number of triples in {input_file}: {num_triples}")

    print(f"Number of triples in the final merged graph: {len(merged_graph)}")
    print("Merged knowledge graphs and saved to merged_knowledge_graph.ttl")
    return


kgs_paths = ["merged_knowledge_graph.ttl", "image_frames_kg.ttl"]
merge_two_kgs(kgs_paths)