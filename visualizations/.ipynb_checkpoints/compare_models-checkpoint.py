import csv
import argparse
import sys
import os
import statistics

def load_csv(filepath):
    """
    Reads a CSV and returns a dict: { 'scene_name': accuracy_float }
    """
    data = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Basic validation of headers
            if "Scene Name" not in reader.fieldnames or "Accuracy (%)" not in reader.fieldnames:
                print(f"Warning: File {filepath} might have incorrect headers. Expected 'Scene Name' and 'Accuracy (%)'")
                return None

            for row in reader:
                scene = row["Scene Name"]
                try:
                    acc = float(row["Accuracy (%)"])
                except ValueError:
                    acc = 0.0
                data[scene] = acc
    except FileNotFoundError:
        print(f"Error: Could not find file {filepath}")
        sys.exit(1)
    return data

def print_section_header(title):
    print("\n" + "="*90)
    print(f" {title}")
    print("="*90)

def main(args):
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    model_data = {} # { 'ModelName': {'scene': 90.0, ...} }
    all_scenes_sets = []

    if not args.files:
        print("Error: No files provided.")
        sys.exit(1)

    print(f"Processing {len(args.files)} files...")

    for filepath in args.files:
        # Use filename without extension as the Model Name
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        data = load_csv(filepath)
        
        if data:
            model_data[model_name] = data
            all_scenes_sets.append(set(data.keys()))
        else:
            print(f"Skipping invalid file: {filepath}")

    if not model_data:
        print("No valid data loaded.")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. Find Common Scenes (Intersection)
    # ---------------------------------------------------------
    # We only compare scenes that exist in ALL models to ensure fair ranking
    common_scenes = set.intersection(*all_scenes_sets)
    sorted_scenes = sorted(list(common_scenes))
    
    if not common_scenes:
        print("Error: No common scenes found across the provided files.")
        return

    print(f"Found {len(common_scenes)} common scenes across {len(model_data)} models.")

    # ---------------------------------------------------------
    # 3. Global Leaderboard
    # ---------------------------------------------------------
    print_section_header("GLOBAL LEADERBOARD")
    
    stats = []
    for model, scenes in model_data.items():
        # Filter only common scenes
        scores = [scenes[s] for s in common_scenes]
        mean_acc = statistics.mean(scores)
        try:
            stdev = statistics.stdev(scores)
        except statistics.StatisticsError:
            stdev = 0.0 # Occurs if only 1 scene
        
        stats.append((model, mean_acc, stdev))

    # Sort by Mean Accuracy descending
    stats.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Rank':<5} | {'Model Name':<30} | {'Avg Acc':<10} | {'Consistency (StdDev)':<20}")
    print("-" * 90)
    for i, (model, avg, dev) in enumerate(stats, 1):
        # Lower StdDev means more consistent performance
        print(f"{i:<5} | {model:<30} | {avg:.2f}%     | {dev:.2f}")

    # ---------------------------------------------------------
    # 4. Detailed Comparative Analysis per Model
    # ---------------------------------------------------------
    
    # Pre-calculate scene stats for fast comparison
    # scene_stats[scene] = { 'avg': float, 'max': float, 'best_model': str, 'vals': dict }
    scene_stats = {}
    for scene in common_scenes:
        vals = [(m, model_data[m][scene]) for m in model_data]
        val_only = [v for k, v in vals]
        
        scene_stats[scene] = {
            'avg': statistics.mean(val_only),
            'max': max(val_only),
            'vals': {k: v for k, v in vals}
        }

    for model in model_data:
        print_section_header(f"ANALYSIS: {model}")
        
        wins = []       # Scored higher than ALL other models
        better = []     # Scored higher than the AVERAGE of others
        same = []       # Approx same as average
        worse = []      # Worse than average

        for scene in sorted_scenes:
            my_score = model_data[model][scene]
            
            # Get scores of everyone else
            others_scores = [v for k, v in scene_stats[scene]['vals'].items() if k != model]
            
            if not others_scores: # Should not happen if >1 model
                continue

            avg_others = statistics.mean(others_scores)
            max_others = max(others_scores)

            diff_avg = my_score - avg_others
            
            # Logic for categorization
            # We use 0.001 epsilon for float comparison
            if my_score > max_others + 0.001:
                wins.append((scene, my_score, max_others))
            elif my_score > avg_others + 0.001: 
                better.append((scene, my_score, avg_others))
            elif my_score < avg_others - 0.001:
                worse.append((scene, my_score, avg_others))
            else:
                same.append((scene, my_score))

        # --- Report Output ---

        # 1. Unique Wins (Crown Jewels)
        print(f"\n[+] DOMINANT WINS ({len(wins)} scenes)")
        print(f"    (Scenes where {model} beat EVERY other model)")
        print("-" * 90)
        if wins:
            print(f"    {'Scene Name':<45} | {'My Score':<10} | {'Next Best':<10} | {'Margin':<10}")
            # Sort by margin of victory
            wins.sort(key=lambda x: x[1]-x[2], reverse=True) 
            for s, my, opp in wins:
                print(f"    {s:<45} | {my:.2f}%     | {opp:.2f}%     | +{(my-opp):.2f}%")
        else:
            print("    None.")

        # 2. Above Average
        print(f"\n[+] BETTER THAN AVERAGE ({len(better)} scenes)")
        print(f"    (Scenes where {model} beat the group average)")
        print("-" * 90)
        if better:
            # Sort by difference from average
            better.sort(key=lambda x: x[1]-x[2], reverse=True)
            print(f"    {'Scene Name':<45} | {'My Score':<10} | {'Grp Avg':<10} | {'Diff':<10}")
            for s, my, avg in better: 
                print(f"    {s:<45} | {my:.2f}%     | {avg:.2f}%     | +{(my-avg):.2f}%")
        else:
            print("    None.")

        # 3. Worse than Average
        print(f"\n[-] WORSE THAN AVERAGE ({len(worse)} scenes)")
        print("-" * 90)
        if worse:
            # Sort by difference (biggest loss first)
            worse.sort(key=lambda x: x[2]-x[1], reverse=True)
            print(f"    {'Scene Name':<45} | {'My Score':<10} | {'Grp Avg':<10} | {'Diff':<10}")
            for s, my, avg in worse: 
                print(f"    {s:<45} | {my:.2f}%     | {avg:.2f}%     | -{(avg-my):.2f}%")
        else:
            print("    None.")

        # 4. Neutral
        print(f"\n[=] SAME AS AVERAGE ({len(same)} scenes)")
        print("-" * 90)
        if same:
            print(f"    {'Scene Name':<45} | {'Score':<10}")
            for s, score in same:
                print(f"    {s:<45} | {score:.2f}%")
        else:
            print("    None.")

    # ---------------------------------------------------------
    # 5. Hardest Scenes (Global)
    # ---------------------------------------------------------
    print_section_header("GLOBAL DATA INSIGHTS")
    print("Hardest Scenes (Lowest Average Accuracy across all models):")
    
    # Create list of (scene, avg_acc)
    scene_avgs = [(s, d['avg']) for s, d in scene_stats.items()]
    scene_avgs.sort(key=lambda x: x[1]) # Sort ascending (lowest first)
    
    for s, avg in scene_avgs:
        print(f"    {s:<45} : {avg:.2f}% avg acc")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare multiple Model CSVs.")
    p.add_argument("files", nargs='+', help="List of CSV files to compare (e.g., model1.csv model2.csv ...)")
    args = p.parse_args()
    main(args)