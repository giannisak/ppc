import pandas as pd

# Read the CSV file
df = pd.read_csv('proposal_sections_with_abstract.csv')

# Function to format each row
def format_row(row):
    template = """
### System: {instruction}
### User: {input}
### Assistant: {output}
"""
    formatted = template.format(instruction=row['instruction'], input=row['input'], output=row['output'])
    return formatted.strip()

# Apply the function to each row
df['text'] = df.apply(format_row, axis=1)

# Select only the 'text' column for the new CSV
formatted_df = df[['text']]

# Write the output to a new CSV file
formatted_df.to_csv('proposal_subsections.csv', index=False)

print("Formatting completed and output saved to 'proposal_subsections.csv'.")
