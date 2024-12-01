import os
import zipfile
from striprtf.striprtf import rtf_to_text
import chardet

def detect_and_decode(file_content):
    """
    Detect and decode file content with multiple encoding attempts
    
    Args:
        file_content (bytes): Raw file content to decode
    
    Returns:
        str: Decoded file content
    """
    # Prioritized list of encodings
    encodings_to_try = [
        'utf-8-sig',  # UTF-8 with BOM
        'big5',       # Traditional Chinese 
        'gb2312',     # Simplified Chinese
        'gbk',        # Extended Chinese encoding
        'utf-16',     # Unicode 
        'utf-8',      # Standard UTF-8
        'cp950',      # Traditional Chinese Windows encoding
        'ascii'       # Fallback to ASCII
    ]
    
    # First, try chardet detection
    try:
        detected = chardet.detect(file_content)
        suggested_encoding = detected['encoding']
        
        # Try the suggested encoding first
        if suggested_encoding:
            try:
                return file_content.decode(suggested_encoding, errors='replace')
            except:
                pass
    except:
        pass
    
    # Try fallback encodings
    for encoding in encodings_to_try:
        try:
            decoded = file_content.decode(encoding, errors='replace')
            # Additional validation to ensure meaningful content
            if decoded and len(decoded.strip()) > 10:
                return decoded
        except:
            continue
    
    # Absolute fallback
    return file_content.decode('utf-8', errors='ignore')

def extract_rtf_content(file_path):
    """
    Extract readable content from RTF files using striprtf library
    
    Args:
        file_path (str): Path to the RTF file to extract content from
    
    Returns:
        str: Extracted readable text content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as rtf_file:
            rtf_content = rtf_file.read()  # Now read as a text (not binary)
            text = rtf_to_text(rtf_content)  # Pass the decoded string to rtf_to_text
            return text
    except Exception as e:
        print(f"Error parsing RTF file {file_path}: {e}")
        return ""

def unzip_files(input_dir='./data/law_zipped', output_dir='./data/law'):
    """
    Extract RTF files from zip files in input directory to output directory
    
    Args:
        input_dir (str): Directory containing zipped files
        output_dir (str): Directory to extract RTF files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Counters for tracking
    total_zip_files = 0
    processed_zip_files = 0
    total_rtf_files = 0
    processed_rtf_files = 0
    
    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        # Check if it's a zip file
        if not filename.lower().endswith('.zip'):
            continue
        
        total_zip_files += 1
        zip_path = os.path.join(input_dir, filename)
        
        try:
            # Open the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files in the zip
                file_list = zip_ref.namelist()
                
                # Process each file
                for file in file_list:
                    # Check if it's an RTF file
                    if not file.lower().endswith('.rtf'):
                        continue
                    
                    total_rtf_files += 1
                    
                    try:
                        # Read the file content from the zip
                        file_content = zip_ref.read(file)
                        
                        # Decode content (if required)
                        decoded_content = detect_and_decode(file_content)
                        
                        # Write the decoded content into a temporary file
                        temp_rtf_path = 'temp.rtf'
                        with open(temp_rtf_path, 'w', encoding='utf-8') as temp_rtf:
                            temp_rtf.write(decoded_content)  # Write as string, not bytes
                        
                        # Extract RTF content
                        rtf_text = extract_rtf_content(temp_rtf_path)
                        
                        # Use the zip file's name (without extension) for the .txt file name
                        output_filename = os.path.splitext(filename)[0] + '.txt'  # Use the zip file's name
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Write the extracted content (RTF text) to the corresponding .txt file
                        with open(output_path, 'w', encoding='utf-8') as output_file:
                            output_file.write(rtf_text)
                        
                        processed_rtf_files += 1
                    
                    except Exception as file_error:
                        print(f"Error processing file {file}: {file_error}")
                
                processed_zip_files += 1
        
        except Exception as zip_error:
            print(f"Error processing zip file {filename}: {zip_error}")
    
    # Print summary
    print(f"Processed {processed_zip_files} out of {total_zip_files} zip files")
    print(f"Extracted {processed_rtf_files} out of {total_rtf_files} RTF files")


if __name__ == '__main__':
    unzip_files()
