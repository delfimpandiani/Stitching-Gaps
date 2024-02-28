import os
import json

# Get a list of all JSON files in the current directory
json_files = [f for f in os.listdir(".") if f.endswith(".json")]

# Function to merge two dictionaries
def merge_dicts(dict1, dict2):
    merged_dict = {**dict1, **dict2}
    return merged_dict

# Initialize an empty dictionary to hold the merged data
merged_data = {}

# Loop through each JSON file and merge its contents
for json_file in json_files:
    with open(json_file, "r") as file:
        data = json.load(file)

    for key in data.keys():
        if key in merged_data:
            merged_data[key] = merge_dicts(merged_data[key], data[key])
        else:
            merged_data[key] = data[key]

# Save the merged data as a new JSON file
with open("merged_data.json", "w") as outfile:
    json.dump(merged_data, outfile, indent=4)
