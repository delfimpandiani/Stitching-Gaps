# Create a new dictionary with updated keys using the ID mapping
import json


with open("img-oriented-dict.json", "r") as file:
    main_dict = json.load(file)


with open("img_id_mapping.json", "r") as file:
    id_dict = json.load(file)

updated_dict = {}
for old_key, value in main_dict.items():
    if old_key in id_dict:
        new_key = id_dict[old_key]
        updated_dict[new_key] = value


with open("../evocation_perceptual_merger/ARTstract_evocation_data.json", "w") as file:
    json.dump(updated_dict, file, indent=2)
