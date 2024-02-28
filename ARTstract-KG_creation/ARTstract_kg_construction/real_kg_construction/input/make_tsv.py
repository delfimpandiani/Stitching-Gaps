import json
import csv

# Load JSON data from a file
with open('real-img-data.json', 'r') as json_file:
    data = json.load(json_file)

# Create a CSV file for writing
with open('output.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write the header row
    writer.writerow(['image id', 'caption'])

    # Iterate through the JSON data and extract information
    for image_id, image_data in data.items():
        if 'ic' in image_data:
            ic_data = image_data['ic']
            for ic_key, ic_value in ic_data.items():
                if 'image_description' in ic_value:
                    caption = ic_value['image_description']
                    writer.writerow([image_id, caption])



import json

# Load JSON data from a file
with open('real-img-data.json', 'r') as json_file:
    data = json.load(json_file)

# Create a TSV file for writing
with open('output.tsv', 'w') as tsv_file:
    # Write the header row
    tsv_file.write('image id\tcaption\n')

    # Iterate through the JSON data and extract information
    for image_id, image_data in data.items():
        if 'ic' in image_data:
            ic_data = image_data['ic']
            for ic_key, ic_value in ic_data.items():
                if 'image_description' in ic_value:
                    caption = ic_value['image_description']
                    tsv_file.write(f'{image_id}\t{caption}\n')
