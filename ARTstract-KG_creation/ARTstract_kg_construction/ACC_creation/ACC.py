import json
import os

# Function to read and parse a JSON file
def read_json_file(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

# Directory where your JSON files are located
directory = "../stats/top_relevant_jsons"

# Initialize a dictionary to store the combined data
combined_data = {}

# Iterate over the JSON files
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        file_name_without_extension = os.path.splitext(filename)[0]
        json_data = read_json_file(file_path)
        for key, value in json_data.items():
            if key not in combined_data:
                combined_data[key] = {}
            combined_data[key][file_name_without_extension] = value

# Specify the output file where you want to save the combined data
output_file = "combined_data.json"

# Write the combined data to a new JSON file
with open(output_file, "w") as json_file:
    json.dump(combined_data, json_file, indent=4)

print(f"Combined data has been written to {output_file}")
