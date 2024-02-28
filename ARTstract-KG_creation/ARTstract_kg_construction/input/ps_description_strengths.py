import json

# Load your JSON data
with open('merged_test_data.json', 'r') as json_file:
    data = json.load(json_file)

# Function to round annotation strength to two decimal places
def round_annotation_strength(annotation_strength):
    try:
        return round(float(annotation_strength), 2)
    except (ValueError, TypeError):
        return 'N/A'

# Function to generate the description
def generate_description_with_strength(image_data):

    art_style_data = image_data['as']['ARTstract_as_2023_06_26']
    art_style = art_style_data['art_style']
    art_style_strength = round_annotation_strength(art_style_data.get('annotation_strength', 'N/A'))

    emotion_data = image_data['em']['ARTstract_em_2023_06_26']
    emotion = emotion_data['emotion']
    emotion_strength = round_annotation_strength(emotion_data.get('annotation_strength', 'N/A'))

    action_data = image_data['act']['ARTstract_act_2023_06_28']
    action = action_data['action_label']
    action_strength = round_annotation_strength(action_data.get('annotation_strength', 'N/A'))

    color_data = image_data['color']['ARTstract_color_2023_06_26'][:5]
    colors = [color['webcolor_name'] for color in color_data]
    color_text = ', '.join(colors)

    object_data = image_data['od']['ARTstract_od_2023_06_28']['detected_objects']
    objects = [
        f"{obj['detected_object']} (Strength: {round_annotation_strength(obj.get('annotation_strength', 'N/A'))})" for
        obj in object_data]
    object_text = ', '.join(objects)

    human_presence_data = image_data['hp']['ARTstract_hp_2023_06_26']
    human_presence = human_presence_data['human_presence']
    human_presence_strength = round_annotation_strength(human_presence_data.get('annotation_strength', 'N/A'))

    age_data = image_data['age']['ARTstract_age_2023_06_26']
    age_tier = age_data['age_tier']
    age_strength = round_annotation_strength(age_data.get('annotation_strength', 'N/A'))

    caption_data = image_data['ic']['ARTstract_ic_2023_06_28']
    caption = caption_data['image_description']

    description = f"This image  shows a {art_style} art style, evokes {emotion} emotion, depicts {action} as main action, the top five colors are: {color_text}, and the following objects were detected: {object_text}. It has a human presence: {human_presence}. It was automatically captioned as: '{caption}'. Depicts the following age tier: {age_tier}.)."
    description_with_strength = f"This image shows a {art_style} art style (Strength: {art_style_strength}), evokes {emotion} emotion (Strength: {emotion_strength}), depicts {action} (Strength: {action_strength}), the top five colors are: {color_text}, and the following objects were detected: {object_text}. It has a human presence: {human_presence} (Strength: {human_presence_strength}).  It was automatically captioned as: '{caption}'. Depicts the following age tier: {age_tier} (Strength: {age_strength})."

    return caption, description, description_with_strength


# Create a dictionary to store descriptions
image_descriptions = {}

# Loop through each image data
for image_id, image_data in data.items():
    caption, description, description_with_strength = generate_description_with_strength(image_data)
    image_descriptions[image_id] = {}
    image_descriptions[image_id]["caption"] = caption
    image_descriptions[image_id]["description"] = description
    image_descriptions[image_id]["description_w_strength"] = description_with_strength

# Write the dictionary to a JSON file
with open('test_image_descriptions.json', 'w') as output_file:
    json.dump(image_descriptions, output_file, indent=4)

print("Image descriptions saved to 'image_descriptions.json'.")
