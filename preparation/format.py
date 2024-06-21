import pandas as pd
import re

# Path to your CSV file
csv_file_path = '/home/llm-server/jak/finetuning/proposal_subsections.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Function to extract instruction, input, and output from the text
def extract_components(text):
    instruction_match = re.search(r'### System: (.*?)### User:', text, re.DOTALL)
    input_match = re.search(r'### User: (.*?)### Assistant:', text, re.DOTALL)
    output_match = re.search(r'### Assistant: (.*)', text, re.DOTALL)
    
    instruction = instruction_match.group(1).strip() if instruction_match else ""
    input_text = input_match.group(1).strip() if input_match else ""
    output_text = output_match.group(1).strip() if output_match else ""
    
    return instruction, input_text, output_text

# Apply the extraction function to each row
df[['instruction', 'input', 'output']] = df['text'].apply(lambda x: pd.Series(extract_components(x)))

# Save the DataFrame with the extracted columns to a new CSV file
extracted_csv_path = '/home/llm-server/jak/finetuning/cook_subsections.csv'
df[['instruction', 'input', 'output']].to_csv(extracted_csv_path, index=False)

# Optionally, display the first few rows to verify
print(df[['instruction', 'input', 'output']].head())
