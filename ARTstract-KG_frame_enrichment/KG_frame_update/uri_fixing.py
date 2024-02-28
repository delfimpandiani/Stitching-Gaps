import csv

# Function to replace the URLs
def find_replace_url(text):
    return text.replace("https://w3id.org/framester/wn/wn30/instances/", "wn30instances:")

# Input and output file paths
input_file_path = "frame_output.csv"
output_file_path = "frame_output_updated.csv"

# Open the input CSV file and create the output CSV file
with open(input_file_path, "r", newline="") as input_file, open(output_file_path, "w", newline="") as output_file:
    # Create CSV reader and writer with comma delimiter
    csv_reader = csv.reader(input_file)
    csv_writer = csv.writer(output_file)

    # Write the header
    header = next(csv_reader)
    csv_writer.writerow(header)

    # Process and write the rows
    for row in csv_reader:
        # Replace URLs in the specified columns (adjust column indices as needed)
        row[4] = find_replace_url(row[4])
        row[5] = find_replace_url(row[5])
        csv_writer.writerow(row)

print("Processing complete. Modified content saved to", output_file_path)
