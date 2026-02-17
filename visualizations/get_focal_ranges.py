import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---
# The root directory you provided
ROOT_DIR = "/mnt/dgx_lab/datasets/depthLMoriginal"

# Define the focal length ranges (Bins)
# Ranges are [min, max). e.g., 400-800 includes 400 but not 800.
BINS = [0, 400, 800, 1200, 2000, 5000, 10000]
LABELS = [
    'Fish-eye (<400)', 
    'Wide (400-800)', 
    'Standard (800-1200)', 
    'Telephoto (1200-2000)', 
    'Super-Tele (2000-5000)',
    'Extreme (>5000)'
]

def get_bin(focal_length):
    for i in range(len(BINS) - 1):
        if BINS[i] <= focal_length < BINS[i+1]:
            return LABELS[i]
    return "Unknown"

def process_jsonl(filepath):
    """Reads a JSONL and returns a list of focal lengths."""
    focal_lengths = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    # Intrinsics format is usually [fx, fy, cx, cy, w, h]
                    # We take index 0 as fx
                    if "intrinsics" in data and len(data["intrinsics"]) >= 1:
                        fx = float(data["intrinsics"][0])
                        focal_lengths.append(fx)
                except (json.JSONDecodeError, IndexError, ValueError):
                    continue
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        
    return focal_lengths

def main():
    print(f"Scanning directory: {ROOT_DIR} ...")
    
    # 1. Find all JSONL files recursively
    # We use os.walk to go through every subfolder (nu, nyu, sun, etc.)
    jsonl_files = []
    for dirpath, _, filenames in os.walk(ROOT_DIR):
        for f in filenames:
            if f.endswith(".jsonl"):
                jsonl_files.append(os.path.join(dirpath, f))
    
    if not jsonl_files:
        print("No .jsonl files found!")
        return

    print(f"Found {len(jsonl_files)} JSONL files.")
    
    # 2. Extract Data
    all_data = []
    
    for file_path in tqdm(jsonl_files, desc="Processing Files"):
        # Get a readable name (e.g., 'nu/nuscenes_mini.jsonl')
        rel_path = os.path.relpath(file_path, ROOT_DIR)
        dataset_name = rel_path.split('/')[0] # approximate dataset name from folder
        
        focals = process_jsonl(file_path)
        
        # Categorize every single image
        for fx in focals:
            all_data.append({
                'Dataset': dataset_name,
                'File': os.path.basename(file_path),
                'Focal_Length': fx,
                'Range': get_bin(fx)
            })

    # 3. Create DataFrame
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("No intrinsic data found in files.")
        return

    # 4. Generate Summary Table (Pivot)
    # Rows: Dataset Name, Columns: Focal Ranges
    print("\n--- Training Data Distribution ---")
    summary = pd.crosstab(index=df['Dataset'], columns=df['Range'])
    
    # Ensure all columns exist even if count is 0
    for label in LABELS:
        if label not in summary.columns:
            summary[label] = 0
            
    # Reorder columns logically
    summary = summary[LABELS]
    
    # Add Total column
    summary['Total Images'] = summary.sum(axis=1)
    
    print(summary)
    
    # Save CSV
    output_csv = os.path.join(ROOT_DIR, "training_data_distribution.csv")
    summary.to_csv(output_csv)
    print(f"\nSummary saved to: {output_csv}")

    # 5. Generate Plot
    plt.figure(figsize=(14, 7))
    
    # Create a stacked bar chart
    summary_no_total = summary.drop(columns=['Total Images'])
    summary_no_total.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 6))
    
    plt.title("Distribution of Focal Lengths per Dataset")
    plt.xlabel("Dataset Folder")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.legend(title="Focal Range")
    plt.tight_layout()
    
    output_img = os.path.join(ROOT_DIR, "training_data_distribution.png")
    plt.savefig(output_img)
    print(f"Chart saved to: {output_img}")
    
    # 6. Global Stats
    print("\n--- Global Statistics ---")
    print(f"Total Images Scanned: {len(df)}")
    print(f"Min Focal Length: {df['Focal_Length'].min():.2f}")
    print(f"Max Focal Length: {df['Focal_Length'].max():.2f}")
    print(f"Mean Focal Length: {df['Focal_Length'].mean():.2f}")

if __name__ == "__main__":
    main()
