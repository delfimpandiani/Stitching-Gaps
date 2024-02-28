import os
import json

# Get a list of all JSON files in the "output" directory
json_files = ["ARTstract_evocation_data.json", "ARTstract_perceptual_data.json"]

# Initialize an empty dictionary to hold the merged data
merged_data = {}

# Function to merge two dictionaries
def merge_dicts(dict1, dict2):
    merged_dict = {**dict1, **dict2}
    return merged_dict

# Loop through each JSON file and merge its contents
for json_file in json_files:
    with open(json_file, "r") as file:
        data = json.load(file)
        print(json_file, "has length", len(data))

    # Collect the keys that appear in the current data
    current_keys = set(data.keys())

    # If this is the first JSON file, set it as the initial merged_data
    if not merged_data:
        merged_data = data.copy()
    else:
        # Remove keys from merged_data that don't appear in the current data
        keys_to_remove = [key for key in merged_data.keys() if key not in current_keys]
        for key in keys_to_remove:
            merged_data.pop(key, None)

        # Update the values of keys that appear in both dictionaries
        for key in current_keys.intersection(merged_data.keys()):
            merged_data[key] = merge_dicts(merged_data[key], data[key])

# Save the merged data as a new JSON file
with open("../merged_ARTstract.json", "w") as outfile:
    json.dump(merged_data, outfile, indent=4)

print("merged data has length", len(merged_data))
