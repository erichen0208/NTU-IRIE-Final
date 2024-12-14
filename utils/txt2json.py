import os
import re
import json

def process_file(file_path, file_name):
    """
    Process a single file to extract provisions and content, skipping provisions with content "(刪除)".

    Args:
        file_path (str): Path to the txt file.
        file_name (str): Name of the txt file (used for constructing provisions).

    Returns:
        list: List of dictionaries with 'provision' and 'content'.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Extract file base name without extension
    base_name = os.path.splitext(file_name)[0]

    # Pattern to match provisions (e.g., 第 2 條 or 第 5-1 條)
    provision_pattern = re.compile(r"第\s?(\d+)(?:-(\d+))?\s?條")

    entries = []
    current_provision = None
    current_content = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if the line is a provision
        match = provision_pattern.match(line)
        if match:
            # If there's an ongoing provision, save it (unless the content is "(刪除)")
            if current_provision and current_content:
                content_str = "\n".join(current_content).strip()
                if content_str != "（刪除）":  # Skip provisions marked as "(刪除)"
                    entries.append({
                        "provision": current_provision,
                        "content": content_str
                    })

            # Start a new provision
            main_number = match.group(1)
            sub_number = match.group(2)
            provision_suffix = f"第{main_number}條"
            if sub_number:
                provision_suffix = f"第{main_number}條之{sub_number}"
            current_provision = f"{base_name}{provision_suffix}"
            current_content = []
        else:
            # Add line to the current content
            current_content.append(line)

    # Add the last provision if exists and not "(刪除)"
    if current_provision and current_content:
        content_str = "\n".join(current_content).strip()
        if content_str != "（刪除）":
            entries.append({
                "provision": current_provision,
                "content": content_str
            })

    return entries

def process_directory(input_dir, output_file):
    """
    Process all txt files in the directory and save as JSONL.

    Args:
        input_dir (str): Directory containing txt files.
        output_file (str): Path to the output JSONL file.
    """
    all_entries = []

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith('.txt'):
            file_path = os.path.join(input_dir, file_name)
            entries = process_file(file_path, file_name)
            all_entries.extend(entries)

    # Write all entries to a JSONL file
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for entry in all_entries:
            jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    input_directory = './data/law'
    output_jsonl = './data/law.jsonl'
    process_directory(input_directory, output_jsonl)
    print(f"Processed files from {input_directory} into {output_jsonl}")
