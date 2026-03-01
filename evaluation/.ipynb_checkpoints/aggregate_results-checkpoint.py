import csv
import re
import argparse
import os
from collections import defaultdict

def extract_scene_name(image_path):
    """
    Extracts the base scene name from the file path.
    Example: /path/to/corridor_01_p000_true.jpg -> corridor_01
    """
    # 1. Get the filename only (remove directory)
    filename = os.path.basename(image_path)
    
    # 2. Regex to find the pattern "_p" followed by digits
    # We capture everything BEFORE the "_pXXX" part.
    match = re.search(r"(.+)_p\d+", filename)
    
    if match:
        return match.group(1)
    else:
        # Fallback: if it doesn't match the pattern, return the whole filename
        return filename

def main(args):
    print(f"Reading: {args.input_csv}")
    
    # Dictionary to store stats: { "scene_name": {total: 0, correct: 0, incorrect: 0} }
    scene_stats = defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0})
    
    try:
        with open(args.input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                image_path = row.get("image", "")
                if not image_path:
                    continue

                # Get the scene name (grouping key)
                scene = extract_scene_name(image_path)
                
                # Check correctness
                # We check both the string flag and the score number for robustness
                is_correct_str = row.get("is_correct", "False").lower()
                delta_score = float(row.get("delta1_score", 0.0))
                
                # Consider it correct if the flag is true OR score is 1.0
                is_correct = (is_correct_str == "true") or (delta_score == 1.0)
                
                scene_stats[scene]["total"] += 1
                if is_correct:
                    scene_stats[scene]["correct"] += 1
                else:
                    scene_stats[scene]["incorrect"] += 1

    except FileNotFoundError:
        print(f"Error: The file {args.input_csv} was not found.")
        return

    # Write the aggregated results
    print(f"Writing: {args.output_csv}")
    
    headers = ["Scene Name", "Total Points", "Correct Count", "Incorrect Count", "Accuracy (%)"]
    
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Sort alphabetically by scene name for cleaner output
        sorted_scenes = sorted(scene_stats.keys())
        
        for scene in sorted_scenes:
            stats = scene_stats[scene]
            total = stats["total"]
            correct = stats["correct"]
            incorrect = stats["incorrect"]
            
            accuracy = (correct / total * 100) if total > 0 else 0.0
            
            writer.writerow([
                scene,
                total,
                correct,
                incorrect,
                f"{accuracy:.2f}"
            ])
            
    print("Done! Summary:")
    print(f" - Processed {len(sorted_scenes)} unique scenes.")
    print(f" - Saved to {args.output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True, help="Path to the evaluation results CSV")
    p.add_argument("--output_csv", default="scene_accuracy_report.csv", help="Path for the new summary CSV")
    args = p.parse_args()
    main(args)