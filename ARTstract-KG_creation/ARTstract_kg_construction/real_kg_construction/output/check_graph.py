from rdflib import Graph
import os

# Define the find and replace strings
find_str = "ns1"
replace_str = "situannotate"

# Function to perform find and replace in a Turtle file and save it
def find_replace_and_save(input_file, output_file):
    with open(input_file, "r") as input_f:
        data = input_f.read()
        updated_data = data.replace(find_str, replace_str)

    with open(output_file, "w") as output_f:
        output_f.write(updated_data)

# List of input and output file pairs
file_pairs = [
    ("real_acve_situations_kg.ttl", "real_acve_situations_kg_updated.ttl"),
    ("real_images_acve_kg.ttl", "real_images_acve_kg_updated.ttl"),
    ("real_images_kg.ttl", "real_images_kg_updated.ttl"),
    ("real_situations_kg.ttl", "real_situations_kg_updated.ttl")
]

# Perform find and replace for each file pair
for input_file, output_file in file_pairs:
    find_replace_and_save(input_file, output_file)


# Create an empty RDFLib Graph to hold the merged data
merged_graph = Graph()

# Dictionary to store the number of triples in each updated graph
num_triples_dict = {}

# Parse the updated Turtle files and merge them into the merged_graph
for input_file, output_file in file_pairs:
    g = Graph()
    g.parse(output_file, format="turtle")
    merged_graph += g
    num_triples_dict[input_file] = len(g)

# Serialize the merged data to Turtle format
merged_ttl = merged_graph.serialize(format="turtle").encode("utf-8")

# Save the merged data to a new Turtle file
merged_filename = "merged_knowledge_graph.ttl"
with open(merged_filename, "wb") as output_file:
    output_file.write(merged_ttl)

# Clean up: Remove the updated Turtle files (optional)
for _, output_file in file_pairs:
    os.remove(output_file)

# Print the number of triples in each updated graph and the final merged graph
for input_file, num_triples in num_triples_dict.items():
    print(f"Number of triples in {input_file}: {num_triples}")

print(f"Number of triples in the final merged graph: {len(merged_graph)}")
print("Merged knowledge graphs and saved to merged_knowledge_graph.ttl")
