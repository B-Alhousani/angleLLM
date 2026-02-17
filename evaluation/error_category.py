import pandas as pd
import numpy as np
import glob
import os

def analyze_depth_performance(file_pattern="*.csv"):
    """
    Analyzes depth estimation CSV files.
    Groups results into distance ranges and calculates accuracy and average error.
    """
    # Find all CSV files matching the pattern
    csv_files = glob.glob(file_pattern)
    
    if not csv_files:
        print("No CSV files found matching the pattern.")
        return

    # Define the bins and labels for the distance ranges
    # Ranges: 1-3, 3-5, 5-10, 10-15, 15-20, 20+
    bins = [1, 3, 5, 10, 15, 20, float('inf')]
    labels = ['1-3', '3-5', '5-10', '10-15', '15-20', '20+']

    for file_path in csv_files:
        print(f"\n{'='*60}")
        print(f"File: {os.path.basename(file_path)}")
        print(f"{'='*60}")

        try:
            # Load the CSV
            df = pd.read_csv(file_path)
            
            # Ensure required columns are present
            required_cols = ['ground_truth', 'prediction', 'is_correct']
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {file_path}: Required columns {required_cols} not found.")
                continue

            # 1. Calculate the Absolute Error for each row (e.g., |15 - 1| = 14)
            df['abs_error'] = (df['ground_truth'] - df['prediction']).abs()

            # 2. Categorize each row into a distance range based on ground_truth
            # right=False means [1, 3), [3, 5), etc.
            df['range'] = pd.cut(df['ground_truth'], bins=bins, labels=labels, right=False)

            # 3. Aggregate metrics by range
            # observed=False ensures all labels (even empty ones) appear in the table
            analysis = df.groupby('range', observed=False).agg(
                Sample_Count=('ground_truth', 'count'),
                Accuracy=('is_correct', 'mean'),
                Average_Error=('abs_error', 'mean')
            )

            # 4. Formatting for display
            # Convert accuracy to percentage string
            analysis['Accuracy'] = (analysis['Accuracy'] * 100).round(2).astype(str) + '%'
            # Round the Average Error for better readability
            analysis['Average_Error'] = analysis['Average_Error'].round(3)

            # Display the table
            print(analysis)

            # Calculate and print overall file stats
            overall_acc = (df['is_correct'].mean() * 100).round(2)
            overall_avg_err = df['abs_error'].mean().round(3)
            print(f"\nOverall Statistics for this file:")
            print(f"  - Total Accuracy: {overall_acc}%")
            print(f"  - Total Average Error: {overall_avg_err}")

        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

if __name__ == "__main__":
    # You can change the path below to point to your specific CSVs
    # e.g., analyze_depth_performance("results/*.csv")
    analyze_depth_performance("/mnt/dgx_lab/evaluation/Angleour_eval/evaluation_results.csv")
    analyze_depth_performance("/mnt/dgx_lab/evaluation/Angleour_eval/discriptive_evaluation_results.csv")
    analyze_depth_performance("/mnt/dgx_lab/evaluation/DepthLm_eval/evaluation_results.csv")