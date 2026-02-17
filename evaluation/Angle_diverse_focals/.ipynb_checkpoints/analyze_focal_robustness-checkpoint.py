import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np

# --- CONFIGURATION ---
BASE_DIR = "/mnt/dgx_lab/evaluation/Angle_diverse_focals/"
JSONL_PATH = "/mnt/dgx_lab/datasets/ibims/DepthLM_Official/examples/ibims1/ibims1_val.jsonl" 

# The list of subfolders to look for
SCALES_FOLDERS = ["0.2", "0.5", "0.8", "1.5", "2.0", "3.0"]

def load_intrinsics_map(json_path):
    """
    Reads the JSONL and creates a dictionary: 
    {'rgb/image_name.png': original_focal_length}
    """
    print(f"Loading original intrinsics from {json_path}...")
    intrinsics_map = {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                
                # Extract Filename
                fname = data.get("image")
                if not fname and "images" in data and len(data["images"]) > 0:
                    fname = data["images"][0]
                
                # Extract Focal Length (fx is usually index 0)
                # Structure seen in your data: "intrinsics": [559.62, 558.14, ...]
                if fname and "intrinsics" in data and len(data["intrinsics"]) >= 1:
                    original_fx = float(data["intrinsics"][0])
                    intrinsics_map[str(fname)] = original_fx
                    
            except Exception as e:
                continue
                
    print(f"Loaded intrinsics for {len(intrinsics_map)} scenes.")
    return intrinsics_map

def load_data(base_dir, folders, intrinsics_map):
    all_dfs = []
    
    # 1. Load subfolders
    for scale in folders:
        csv_path = os.path.join(base_dir, scale, "results.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {scale} (not found)")
            continue
            
        print(f"Loading results for scale {scale}...")
        df = pd.read_csv(csv_path)
        df['Scale_Multiplier'] = float(scale)
        all_dfs.append(df)

    # 2. Load Baseline (1.0) if it exists in root
    root_csv_path = os.path.join(base_dir, "results.csv")
    if os.path.exists(root_csv_path):
        print("Loading Baseline (1.0)...")
        df_root = pd.read_csv(root_csv_path)
        df_root['Scale_Multiplier'] = 1.0
        all_dfs.append(df_root)

    if not all_dfs:
        raise ValueError("No results found! Check your paths.")
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # --- CALCULATE ACTUAL FOCAL LENGTH ---
    # Logic: Real_FL = Original_FL * Scale_Multiplier
    def get_real_fl(row):
        fname = str(row['Parent_File'])
        
        # Try exact match
        base_fl = intrinsics_map.get(fname)
        
        # Fallback: sometimes CSV paths have ./ prefix or different relative paths
        if base_fl is None:
             base_fl = intrinsics_map.get(os.path.basename(fname))

        if base_fl is None:
            return np.nan 
            
        return base_fl * row['Scale_Multiplier']

    full_df['Actual_Focal_Length'] = full_df.apply(get_real_fl, axis=1)
    
    # Filter out missing data
    original_len = len(full_df)
    full_df = full_df.dropna(subset=['Actual_Focal_Length'])
    print(f"Matched {len(full_df)}/{original_len} samples with focal lengths.")
        
    return process_metrics(full_df)

def process_metrics(df):
    # Clean string errors just in case
    df = df[pd.to_numeric(df['Ground_Truth'], errors='coerce').notnull()]
    df = df[pd.to_numeric(df['Prediction'], errors='coerce').notnull()]
    
    df['Ground_Truth'] = df['Ground_Truth'].astype(float)
    df['Prediction'] = df['Prediction'].astype(float)
    
    # Metrics
    # Standard Depth Estimation metric: max(gt/pred, pred/gt) < 1.25
    df['Ratio'] = np.maximum(df['Prediction'] / df['Ground_Truth'], df['Ground_Truth'] / df['Prediction'])
    df['Delta1'] = (df['Ratio'] < 1.25).astype(int)
    
    # Absolute Relative Error: |gt - pred| / gt
    df['AbsRel'] = np.abs(df['Prediction'] - df['Ground_Truth']) / df['Ground_Truth']
    
    return df

def analyze_ranges(df):
    # --- 1. Define Focal Length Bins (in Pixels) ---
    # Based on your data (approx 550px base), ranges might look like:
    # 0.2x -> ~110px (Very Wide)
    # 0.5x -> ~275px (Wide)
    # 1.0x -> ~550px (Standard)
    # 2.0x -> ~1100px (Tele)
    # 3.0x -> ~1650px (Super Tele)
    
    bins = [0, 200, 400, 800, 1500, 10000]
    labels = ['Ultra-Wide (<200px)', 'Wide (200-400px)', 'Standard (400-800px)', 'Tele (800-1500px)', 'Super-Tele (>1500px)']
    
    df['Lens_Type'] = pd.cut(df['Actual_Focal_Length'], bins=bins, labels=labels)

    # --- 2. Performance Summary Table ---
    print("\n--- Performance by Actual Focal Length Range ---")
    summary = df.groupby('Lens_Type', observed=False)[['Delta1', 'AbsRel']].agg(
        Accuracy=('Delta1', 'mean'),
        AbsRel_Error=('AbsRel', 'mean'),
        Count=('Delta1', 'count')
    ).reset_index()
    
    print(summary)
    summary.to_csv(os.path.join(BASE_DIR, "focal_range_performance.csv"))

    # --- 3. Visualization: Accuracy Bar Chart ---
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x='Lens_Type', y='Accuracy', palette='viridis', hue='Lens_Type', legend=False)
    plt.axhline(y=df['Delta1'].mean(), color='red', linestyle='--', label='Dataset Average')
    
    plt.title("AngleLM Accuracy across Focal Length Ranges")
    plt.ylabel("Accuracy (Delta < 1.25)")
    plt.xlabel("Focal Length Category (Pixels)")
    plt.xticks(rotation=15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "focal_range_bar_chart.png"))
    plt.close()

    # --- 4. Visualization: Continuous Scatter Plot ---
    # This shows the "flatness" of your model's performance
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='Actual_Focal_Length', y='Ratio', hue='Scale_Multiplier', 
                    palette='coolwarm', alpha=0.6, s=15)
    
    plt.axhline(y=1.25, color='green', linestyle='-', linewidth=2, label='Success Threshold')
    plt.ylim(0.5, 3.0) # Limit Y axis to zoom in on relevant errors
    plt.xscale('log')  # Log scale usually better for focal lengths
    
    plt.title("Robustness: Error Ratio vs. Actual Focal Length (Log Scale)")
    plt.xlabel("Actual Focal Length (Pixels)")
    plt.ylabel("Error Ratio (Closer to 1.0 is better)")
    plt.legend(title="Zoom Multiplier")
    plt.grid(True, which="both", ls="--", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "focal_continuous_scatter.png"))
    plt.close()

if __name__ == "__main__":
    intrinsics = load_intrinsics_map(JSONL_PATH)
    try:
        full_data = load_data(BASE_DIR, SCALES_FOLDERS, intrinsics)
        analyze_ranges(full_data)
        print("\nAnalysis complete. Files saved:")
        print(f"1. {os.path.join(BASE_DIR, 'focal_range_performance.csv')}")
        print(f"2. {os.path.join(BASE_DIR, 'focal_range_bar_chart.png')}")
        print(f"3. {os.path.join(BASE_DIR, 'focal_continuous_scatter.png')}")
    except Exception as e:
        print(f"An error occurred: {e}")