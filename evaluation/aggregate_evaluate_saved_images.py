import csv
import argparse
import os
from collections import defaultdict

def main(args):
    print(f"Reading: {args.input_csv}")
    
    # Dictionary to store stats: { "scene_name": {total: 0, correct: 0} }
    scene_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    try:
        with open(args.input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Verify columns exist
            if "Parent_File" not in reader.fieldnames and "Filename" not in reader.fieldnames:
                print("Error: CSV must have 'Parent_File' or 'Filename' column.")
                return

            for row in reader:
                # 1. Get Filename (Support both naming conventions)
                filename = row.get("Parent_File") or row.get("Filename")
                if not filename:
                    continue

                # 2. Clean Scene Name (remove extension .jpg/.png)
                # "kitchen_01.jpg" -> "kitchen_01"
                scene_name = os.path.splitext(os.path.basename(filename))[0]
                
                # 3. Check Correctness
                # The previous script saved '1' for pass and '0' for fail in 'Delta1_Met'
                try:
                    is_correct = int(row.get("Delta1_Met", 0))
                except ValueError:
                    is_correct = 0
                
                # 4. Update Stats
                scene_stats[scene_name]["total"] += 1
                if is_correct == 1:
                    scene_stats[scene_name]["correct"] += 1

    except FileNotFoundError:
        print(f"Error: The file {args.input_csv} was not found.")
        return

    # --- WRITE OUTPUT ---
    print(f"Writing summary to: {args.output_csv}")
    
    headers = ["Scene Name", "Total Samples", "Correct", "Incorrect", "Accuracy (%)"]
    
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Sort alphabetically
        sorted_scenes = sorted(scene_stats.keys())
        
        for scene in sorted_scenes:
            stats = scene_stats[scene]
            total = stats["total"]
            correct = stats["correct"]
            incorrect = total - correct
            
            # Calculate Percentage
            accuracy = (correct / total * 100) if total > 0 else 0.0
            
            writer.writerow([
                scene,
                total,
                correct,
                incorrect,
                f"{accuracy:.2f}"
            ])
            
    print("Done.")
    print(f"Aggregated {len(sorted_scenes)} unique scenes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to the results.csv from the previous step")
    parser.add_argument("--output_csv", default="scene_accuracy_summary.csv", help="Where to save the new report")
    args = parser.parse_args()
    main(args)
