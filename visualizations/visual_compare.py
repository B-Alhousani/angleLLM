import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# --- Configuration ---
FILE_ANGLE_LM = 'evaluation/Angleour_eval/aggregated.csv'
FILE_DEPTH_LM = 'evaluation/DepthLm_eval/depthlm_aggregated.csv'

OUTPUT_BAR = 'visualizations/new_visuals/category_comparison_bar_FUll.png'
OUTPUT_RADAR = 'category_comparison_radar_Full_.png'

def get_base_category(scene_name):
    """
    Extracts 'corridor' from 'corridor_01' or 'corridor_02'.
    """
    match = re.match(r"([a-zA-Z]+)", str(scene_name))
    if match:
        return match.group(1)
    return scene_name

def load_and_group(filepath, label):
    """
    Loads a CSV, groups by base category, and calculates weighted accuracy.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)
    
    # Create a 'Category' column
    df['Category'] = df['Scene Name'].apply(get_base_category)
    
    # Group by Category and sum the raw counts (to get true weighted accuracy)
    grouped = df.groupby('Category')[['Total Points', 'Correct Count']].sum()
    
    # Calculate Accuracy
    col_name = f'{label}_Accuracy'
    grouped[col_name] = (grouped['Correct Count'] / grouped['Total Points']) * 100
    
    return grouped[[col_name]]

def main():
    print("Processing Data...")
    
    # 1. Load and Group Data
    angle_grouped = load_and_group(FILE_ANGLE_LM, 'AngleLM')
    depth_grouped = load_and_group(FILE_DEPTH_LM, 'DepthLM')
    
    if angle_grouped is None or depth_grouped is None:
        return

    # 2. Merge Data
    merged_df = pd.merge(angle_grouped, depth_grouped, on='Category', how='inner')
    
    # Remove 'OVERALL' if it got grouped in (it usually doesn't match regex, but good safety)
    if 'OVERALL' in merged_df.index:
        merged_df = merged_df.drop('OVERALL')
        
    merged_df = merged_df.sort_index()
    
    # Prepare data for plotting
    categories = merged_df.index.tolist()
    angle_acc = merged_df['AngleLM_Accuracy'].tolist()
    depth_acc = merged_df['DepthLM_Accuracy'].tolist()

    print(f"Found {len(categories)} categories: {categories}")

    # --- PLOT 1: Grouped Bar Chart ---
    print("Generating Bar Chart...")
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, angle_acc, width, label='VAIN', color='skyblue', alpha=0.9)
    ax.bar(x + width/2, depth_acc, width, label='DepthLM', color='salmon', alpha=0.9)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Aggregated Accuracy by Scene Category', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_BAR)
    plt.close()
    print(f"Saved: {OUTPUT_BAR}")

    # --- PLOT 2: Radar Chart ---
    print("Generating Radar Chart...")
    
    # Radar charts require closing the loop (repeating the first value at the end)
    stats_angle = angle_acc + [angle_acc[0]]
    stats_depth = depth_acc + [depth_acc[0]]
    labels = categories
    
    # Compute angles
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += [angles[0]] # Close the loop
    
    fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot AngleLM
    ax2.plot(angles, stats_angle, color='skyblue', linewidth=2, label='VAIN')
    ax2.fill(angles, stats_angle, color='skyblue', alpha=0.25)
    
    # Plot DepthLM
    ax2.plot(angles, stats_depth, color='salmon', linewidth=2, label='DepthLM')
    ax2.fill(angles, stats_depth, color='salmon', alpha=0.25)
    
    ax2.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_RADAR)
    plt.close()
    print(f"Saved: {OUTPUT_RADAR}")

if __name__ == "__main__":
    main()