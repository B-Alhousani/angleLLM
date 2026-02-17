import os
import json
import shutil
import tempfile
import argparse

# --- CONFIGURATION ---
NEW_PROMPT = "How many meters is this point from the camera?"

def process_file(file_path):
    """
    Reads a JSONL, updates the prompt, writes to a temp file, 
    and then replaces the original file.
    """
    print(f"[*] Processing: {file_path}")
    
    # Create a temp file to write the updated data
    fd, temp_path = tempfile.mkstemp()
    
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as temp_file:
            with open(file_path, 'r', encoding='utf-8') as original_file:
                for line in original_file:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        # OVERRIDE PROMPT
                        data['prompt'] = NEW_PROMPT
                        
                        # Write back to temp file
                        temp_file.write(json.dumps(data) + '\n')
                    except json.JSONDecodeError:
                        print(f"[!] Warning: Skipping invalid JSON line in {file_path}")
                        continue
        
        # Replace the original file with the updated temp file
        shutil.move(temp_path, file_path)
        
    except Exception as e:
        print(f"[!] Error processing {file_path}: {e}")
        # Clean up temp file if something went wrong
        if os.path.exists(temp_path):
            os.remove(temp_path)

def main():
    parser = argparse.ArgumentParser(description="Recursively override prompts in all JSONL files.")
    parser.add_argument("parent_directory", help="The root folder to search for JSONL files")
    
    args = parser.parse_args()
    root_dir = args.parent_directory

    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a valid directory.")
        return

    print(f"--- Starting Update in {root_dir} ---")
    print(f"--- New Prompt: '{NEW_PROMPT}' ---")

    count = 0
    # Walk through the directory tree
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jsonl"):
                full_path = os.path.join(root, file)
                process_file(full_path)
                count += 1

    print(f"--- Completed. Updated {count} files. ---")

if __name__ == "__main__":
    main()