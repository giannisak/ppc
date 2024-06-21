import os
import pandas as pd
import chardet

# Directory containing the proposals
base_dir = r"/home/llm-server/jak/finetuning"

# Function to detect file encoding
def detect_encoding(filepath):
    with open(filepath, 'rb') as file:
        raw_data = file.read(20000)  # Read the first 10,000 bytes
        result = chardet.detect(raw_data)
        return result['encoding']

# Function to read file with detected encoding and handle errors
def read_file(filepath, encoding):
    try:
        with open(filepath, 'r', encoding=encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        print(f"Error reading {filepath} with encoding {encoding}, trying 'utf-8' with 'ignore' errors")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()

# List to hold the data
data = []

# Iterate over each proposal folder in the base directory
for proposal_folder in os.listdir(base_dir):
    proposal_path = os.path.join(base_dir, proposal_folder)
    if os.path.isdir(proposal_path):
        # Detect encoding for the Abstract.txt
        abstract_path = os.path.join(proposal_path, 'Abstract.txt')
        if os.path.exists(abstract_path):
            abstract_encoding = detect_encoding(abstract_path)
            abstract_text = read_file(abstract_path, abstract_encoding)

            # Iterate over each subsection file in the proposal folder
            for filename in os.listdir(proposal_path):
                if filename.endswith(".txt") and filename != 'Abstract.txt':
                    section_name = filename.replace('.txt', '')
                    filepath = os.path.join(proposal_path, filename)
                    file_encoding = detect_encoding(filepath)
                    text = read_file(filepath, file_encoding)
                    instruction = "You are a proposal writer. According to the following abstract, you will write the text for the requested subsection:"
                    input_text = f"Abstract: {abstract_text} Subsection: {section_name}"
                    data.append({"instruction": instruction, "input": input_text, "output": text})

# Convert to a DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
output_csv_path = os.path.join(base_dir, "proposal_sections_with_abstract.csv")
df.to_csv(output_csv_path, index=False)

print(f"CSV file has been created at: {output_csv_path}")
