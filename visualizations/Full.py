import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# ==========================================
# CONFIGURATION
# ==========================================

# File Paths (As specified)
PATH_FULL    = 'DepthLM_aggregated.csv'
PATH_ANGLE   = 'evaluation/Angleour_eval/aggregated.csv'
PATH_NOCROP  = 'evaluation/DepthLm_eval/depthlm_aggregated.csv'

# Labels for the Legend/Plots
LABEL_FULL   = 'DepthLM_Full_approach'
LABEL_ANGLE  = 'AngleLM'
LABEL_NOCROP = 'DepthLM_No_croping'

# Output Filenames
OUT_SCENE_ACC    = '1_scene_accuracy_comparison.png'
OUT_SCENE_DIFF   = '2_scene_difference_from_anglelm.png'
OUT_CAT_ACC      = '3_category_accuracy.png'
OUT_CAT_RADAR    = '4_category_radar.png'

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_base_category(scene_name):
    """
    Extracts the category from the scene name.
    Example: 'corridor_01' -> 'corridor'
    """
    match = re.match(r"([a-zA-Z]+)", str(scene_name))
    if match:
        return match.group(1)
    return scene_name

def load_data(path, label):
    """
    Loads the CSV and standardizes the columns.
    Expected input columns: 'Scene Name', 'Accuracy (%)', 'Total Points', 'Correct Count'
    """
    print(f"Loading {label} from: {path}")
    if not os.path.exists(path):
        print(f"Error: File not found -> {path}")
        return None
    
    df = pd.read_csv(path)
    # Ensure relevant columns exist
    req_cols = ['Scene Name', 'Accuracy (%)', 'Total Points', 'Correct Count']
    for c in req_cols:
        if c not in df.columns:
            print(f"Warning: Column '{c}' missing in {label}. Found: {df.columns}")
            return None
            
    # Return a clean subset
    return df

# ==========================================
# MAIN SCRIPT
# ==========================================

def main():
    # 1. LOAD DATA
    df_full   = load_data(PATH_FULL, LABEL_FULL)
    df_angle  = load_data(PATH_ANGLE, LABEL_ANGLE)
    df_nocrop = load_data(PATH_NOCROP, LABEL_NOCROP)

    if df_full is None or df_angle is None or df_nocrop is None:
        print("Stopping due to missing files.")
        return

    # 2. PREPARE SCENE-LEVEL DATA
    # Rename columns to avoid collisions during merge
    f = df_full[['Scene Name', 'Accuracy (%)']].rename(columns={'Accuracy (%)': LABEL_FULL})
    a = df_angle[['Scene Name', 'Accuracy (%)']].rename(columns={'Accuracy (%)': LABEL_ANGLE})
    n = df_nocrop[['Scene Name', 'Accuracy (%)']].rename(columns={'Accuracy (%)': LABEL_NOCROP})

    # Merge all 3 DataFrames
    merged = pd.merge(f, a, on='Scene Name', how='inner')
    merged = pd.merge(merged, n, on='Scene Name', how='inner')

    # Filter out 'OVERALL' row
    merged = merged[merged['Scene Name'] != 'OVERALL'].copy()
    merged.sort_values('Scene Name', inplace=True)

    # ---------------------------------------------------------
    # VIZ 1: PER-SCENE ACCURACY COMPARISON (Side-by-Side Bars)
    # ---------------------------------------------------------
    print(f"Generating {OUT_SCENE_ACC}...")
    
    scenes = merged['Scene Name']
    y_pos = np.arange(len(scenes))
    
    # Increase figure height to accommodate 3 bars per scene
    fig_height = max(10, len(scenes) * 0.35) 
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    height = 0.25  # Thinner bars to fit 3
    
    # Plot 3 bars centered on the tick
    ax.barh(y_pos + height, merged[LABEL_FULL],   height, label=LABEL_FULL,   color='#2ca02c') # Green
    ax.barh(y_pos,          merged[LABEL_NOCROP], height, label=LABEL_NOCROP, color='#ff7f0e') # Orange
    ax.barh(y_pos - height, merged[LABEL_ANGLE],  height, label=LABEL_ANGLE,  color='#1f77b4') # Blue
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenes, fontsize=9)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Scene Accuracy Comparison', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 105)
    
    plt.tight_layout()
    plt.savefig(OUT_SCENE_ACC)
    plt.close()

    # ---------------------------------------------------------
    # VIZ 2: PER-SCENE DIFFERENCE (Relative to AngleLM)
    # ---------------------------------------------------------
    print(f"Generating {OUT_SCENE_DIFF}...")
    
    # Calculate Deltas
    diff_full = merged[LABEL_FULL] - merged[LABEL_ANGLE]
    diff_nocrop = merged[LABEL_NOCROP] - merged[LABEL_ANGLE]
    
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    # Plotting 2 difference bars
    # Green/Red for Full, Orange/Red for NoCrop? 
    # To keep it clean, we use consistent colors for the models.
    
    ax.barh(y_pos + height/2, diff_full,   height, label=f'{LABEL_FULL} vs {LABEL_ANGLE}', color='#2ca02c', alpha=0.8)
    ax.barh(y_pos - height/2, diff_nocrop, height, label=f'{LABEL_NOCROP} vs {LABEL_ANGLE}', color='#ff7f0e', alpha=0.8)
    
    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenes, fontsize=9)
    ax.set_xlabel(f'Accuracy Difference (vs {LABEL_ANGLE})', fontsize=12)
    ax.set_title(f'Performance Gain relative to {LABEL_ANGLE}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUT_SCENE_DIFF)
    plt.close()

    # ---------------------------------------------------------
    # PREPARE CATEGORY DATA
    # ---------------------------------------------------------
    # Helper to aggregate by category
    def aggregate_category(df_in, label_name):
        df = df_in.copy()
        df['Category'] = df['Scene Name'].apply(get_base_category)
        # Sum counts to get weighted accuracy
        grouped = df.groupby('Category')[['Total Points', 'Correct Count']].sum()
        grouped[label_name] = (grouped['Correct Count'] / grouped['Total Points']) * 100
        return grouped[[label_name]]

    cat_full = aggregate_category(df_full, LABEL_FULL)
    cat_angle = aggregate_category(df_angle, LABEL_ANGLE)
    cat_nocrop = aggregate_category(df_nocrop, LABEL_NOCROP)
    
    # Merge Categories
    cat_merged = pd.merge(cat_full, cat_angle, on='Category', how='inner')
    cat_merged = pd.merge(cat_merged, cat_nocrop, on='Category', how='inner')
    
    if 'OVERALL' in cat_merged.index:
        cat_merged.drop('OVERALL', inplace=True)
    
    categories = cat_merged.index.tolist()
    x_cat = np.arange(len(categories))

    # ---------------------------------------------------------
    # VIZ 3: CATEGORY COMPARISON (Grouped Bar)
    # ---------------------------------------------------------
    print(f"Generating {OUT_CAT_ACC}...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    width = 0.25
    
    rects1 = ax.bar(x_cat - width, cat_merged[LABEL_ANGLE],  width, label=LABEL_ANGLE,  color='#1f77b4')
    rects2 = ax.bar(x_cat,         cat_merged[LABEL_NOCROP], width, label=LABEL_NOCROP, color='#ff7f0e')
    rects3 = ax.bar(x_cat + width, cat_merged[LABEL_FULL],   width, label=LABEL_FULL,   color='#2ca02c')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Aggregated Accuracy by Category', fontsize=14)
    ax.set_xticks(x_cat)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=11)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUT_CAT_ACC)
    plt.close()

    # ---------------------------------------------------------
    # VIZ 4: CATEGORY RADAR CHART
    # ---------------------------------------------------------
    print(f"Generating {OUT_CAT_RADAR}...")
    
    # Close the loop for radar chart
    labels = categories
    stats_f = cat_merged[LABEL_FULL].tolist(); stats_f += stats_f[:1]
    stats_n = cat_merged[LABEL_NOCROP].tolist(); stats_n += stats_n[:1]
    stats_a = cat_merged[LABEL_ANGLE].tolist(); stats_a += stats_a[:1]
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    
    # Plot AngleLM
    ax.plot(angles, stats_a, color='#1f77b4', linewidth=2, label=LABEL_ANGLE)
    ax.fill(angles, stats_a, color='#1f77b4', alpha=0.1)
    
    # Plot NoCrop
    ax.plot(angles, stats_n, color='#ff7f0e', linewidth=2, label=LABEL_NOCROP)
    ax.fill(angles, stats_n, color='#ff7f0e', alpha=0.1)
    
    # Plot Full Approach
    ax.plot(angles, stats_f, color='#2ca02c', linewidth=2, label=LABEL_FULL)
    ax.fill(angles, stats_f, color='#2ca02c', alpha=0.1)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title('Category Performance Profile', y=1.05, fontsize=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(OUT_CAT_RADAR)
    plt.close()
    
    print("\nDone! All 4 visualizations generated.")

if __name__ == "__main__":
    main()