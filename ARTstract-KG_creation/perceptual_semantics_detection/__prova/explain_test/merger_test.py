import os
import json

# Get a list of all JSON files in the current directory
import os
import json

# Get a list of all JSON files in the "output" directory

# Get the directory where the script is located
script_directory = os.path.dirname(__file__)

# List JSON files in the script's directory
json_files = [os.path.join(script_directory, f) for f in os.listdir(script_directory) if f.endswith(".json")]
# Rest of your code remains the same...

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
        print(json_file, "has length", len(data))

    for key in data.keys():
        if key in merged_data:
            merged_data[key] = merge_dicts(merged_data[key], data[key])
        else:
            merged_data[key] = data[key]

# Save the merged data as a new JSON file
with open("merged_test_data.json", "w") as outfile:
    json.dump(merged_data, outfile, indent=4)


print("merged data has length", len(merged_data))
