import re
import fitz
import os

def remove_header_footer(text):
    # Remove header pattern
    # header_pattern = r".*\n.*Topic ID:.*\n.*\n"
    # text = re.sub(header_pattern, "", text)

    # Remove footer pattern
    # footer_pattern = r"Call:.*\n.*\n.*\n.*\n"
    # text = re.sub(footer_pattern, "", text)

    # Remove footer pattern
    header_pattern = r"Call:.*?Re\-GENERATe\] – \[\d+\].\n"
    text = re.sub(header_pattern, "", text, flags=re.DOTALL)

    return text.strip()

def split_pdf_into_subsections(pdf_file_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_file_path)
    
    # Extract text from the document
    full_text = ''
    for page in pdf_document:
        full_text += page.get_text()
    
    # Close the PDF document
    pdf_document.close()

    # Remove header and footer
    full_text = remove_header_footer(full_text)

    # Find all matches of the title pattern
    matches = re.finditer(r'(\d+\.\d+\.\d+.*)', full_text)
    # matches = re.finditer(r'(\d+\.\d+\.\d+.*\n.*)', full_text)

    # Initialize start position for slicing text
    start_pos = 0
    
    # Initialize variables to store titles and subsections
    titles = ['Administrative forms']
    subsections = []
    
    # Iterate through matches to split text into subsections
    for match in matches:
        # Get start and end positions of the current match
        title_start = match.start()
        title_end = match.end()
        
        # Append the subsection's content to the list
        subsections.append(full_text[start_pos:title_start].strip())

        # Extract title
        titles.append(full_text[title_start:title_end].strip())
        
        # Update start position for the next iteration
        start_pos = title_end
    
    # Append the remaining content as the last subsection
    if start_pos < len(full_text):
        subsections.append(full_text[start_pos:].strip())

    return titles, subsections

# Example usage
pdf_file_path = 'C:/Users/Giannis/Documents/ΔΕΗ/prop 2023/Smart-Nets/Proposal-SEP-210931336.pdf'

# Call the function to split PDF into subsections
titles, subsections = split_pdf_into_subsections(pdf_file_path)

# Extract the directory path of the PDF file
pdf_directory = os.path.dirname(pdf_file_path)

# Output folder to save text files
output_folder = os.path.join(pdf_directory, 'proposal_subsections')

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through each subsection and save it to a separate text file
for idx, (title, subsection) in enumerate(zip(titles, subsections)):
    # Replace characters that are not suitable for file names
    # file_name = re.sub(r'[\\/:*?"<>|\n]', ' ', title)
    # file_name = file_name[:50]
    file_name = re.sub(r'[\\/:*?"<>|]', ' ', title)
    file_path = os.path.join(output_folder, f"{file_name}.txt")
    
    # Write the subsection content to the text file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(subsection)

    print(f"Subsection '{title}' saved.")
