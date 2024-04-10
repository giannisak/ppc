import re
import fitz
import os

def split_pdf_into_subsections(pdf_file_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_file_path)
    
    # Extract text from the document
    full_text = ''
    for page in pdf_document:
        full_text += page.get_text()
    
    # Close the PDF document
    pdf_document.close()

    # Find all matches of the title pattern
    matches = re.finditer(r'(\d+\.\d+\.\d+ [^\n]+)', full_text)
    
    # Initialize start position for slicing text
    start_pos = 0
    
    # Initialize variable to store subsections
    subsections = []
    
    # Iterate through matches to split text into subsections
    for match in matches:
        # Get start and end positions of the current match
        title_start = match.start()
        #title_end = match.end()
        
        # Append the subsection (title + content) to the list
        subsections.append(full_text[start_pos:title_start].strip())
        
        # Update start position for the next iteration
        start_pos = title_start
    
    # Append the remaining content as the last subsection
    if start_pos < len(full_text):
        subsections.append(full_text[start_pos:].strip())

    return subsections

# Example usage
pdf_file_path = 'C:/Users/Giannis/Documents/ΔΕΗ/prop 2023/RAIDO/Proposal-SEP-210941314.pdf'

# Call the function to split PDF into subsections
subsections = split_pdf_into_subsections(pdf_file_path)

# Extract the directory path of the PDF file
pdf_directory = os.path.dirname(pdf_file_path)

# Output folder to save text files
output_folder = os.path.join(pdf_directory, 'proposal_subsections')

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through each subsection and save it to a separate text file
for idx, subsection in enumerate(subsections):
    # Extract the title from the subsection
    title = subsection.split('\n')[0].strip()  # Get the first line as the title
    
    # Replace characters that are not suitable for file names
    title = re.sub(r'[^\w\-_\. ]', '_', title)
    
    file_path = os.path.join(output_folder, f"{title}.txt")
    
    # Write the subsection content to the text file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(subsection)

    print(f"Subsection '{title}' saved.")