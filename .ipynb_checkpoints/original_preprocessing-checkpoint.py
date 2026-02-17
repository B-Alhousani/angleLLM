# from utils.format_data import process_dataset
# # nu
# JSONL_INPUT = "/mnt/dgx_lab/datasets/depthLMoriginal/nu/nuscenes_mini.jsonl"
# IMAGES_DIR = "/mnt/dgx_lab/datasets/depthLMoriginal/nu/processed_images_mini"
# OUTPUT_DIR = "/mnt/dgx_lab/datasets/AngleLM/nu_images"
# process_dataset(JSONL_INPUT, IMAGES_DIR, OUTPUT_DIR)

# # nyu
# JSONL_INPUT = "/mnt/dgx_lab/datasets/depthLMoriginal/nyu/output_jsonl/NYUv2.jsonl"
# IMAGES_DIR = "/mnt/dgx_lab/datasets/depthLMoriginal/nyu/output_image_folder/kv1/NYUdata"
# OUTPUT_DIR = "/mnt/dgx_lab/datasets/AngleLM/nyu_images"
# process_dataset(JSONL_INPUT, IMAGES_DIR, OUTPUT_DIR)

# # sun
# JSONL_INPUT = "/mnt/dgx_lab/datasets/depthLMoriginal/sun/output_jsonl/sunRGBD.jsonl"
# IMAGES_DIR = "/mnt/dgx_lab/datasets/depthLMoriginal/sun/output_image_folder"
# OUTPUT_DIR = "/mnt/dgx_lab/datasets/AngleLM/sun_images"

# process_dataset(JSONL_INPUT, IMAGES_DIR, OUTPUT_DIR)

import multiprocessing
import os
from utils.original_format_data import process_dataset

def run_task(task_config):
    """Wrapper function to unpack the dictionary and run the process."""
    print(f"Starting process for: {task_config['name']}")
    try:
        process_dataset(
            task_config['json'], 
            task_config['images'], 
            task_config['output']
        )
        print(f"Finished process: {task_config['name']}")
    except Exception as e:
        print(f"Error in {task_config['name']}: {e}")

if __name__ == "__main__":
    # Define your datasets
    datasets = [
        #    {
        #     "name": "IbimsOurs",
        #     "json": "/mnt/dgx_lab/datasets/ibims/DepthLM_Official/examples/ibims1/ibims1_val.jsonl",
        #     "images": "/mnt/dgx_lab/datasets/ibims/DepthLM_Official/examples/ibims1",
        #     "output": "/mnt/dgx_lab/datasets/IbimsOG"
        # },
        # {
        #     "name": "nuScenes",
        #     "json": "/mnt/dgx_lab/datasets/depthLMoriginal/nu/nuscenes_mini.jsonl",
        #     "images": "/mnt/dgx_lab/datasets/depthLMoriginal/nu/processed_images_mini",
        #     "output": "/mnt/dgx_lab/datasets/DepthLMApproach/nu"
        # },
        # {
        #     "name": "NYU",
        #     "json": "/mnt/dgx_lab/datasets/depthLMoriginal/nyu/output_jsonl/NYUv2.jsonl",
        #     "images": "/mnt/dgx_lab/datasets/depthLMoriginal/nyu/output_image_folder",
        #     "output": "/mnt/dgx_lab/datasets/DepthLMApproach/nyu"
        # },
        {
            "name": "SUN",
            "json": "/mnt/dgx_lab/datasets/depthLMoriginal/sun/output_jsonl/sunRGBD.jsonl",
            "images": "/mnt/dgx_lab/datasets/depthLMoriginal/sun/output_image_folder",
            "output": "/mnt/dgx_lab/datasets/DepthLMApproach/sun"
        }
     
    ]

    # Create a process for each dataset
    processes = []
    for ds in datasets:
        p = multiprocessing.Process(target=run_task, args=(ds,))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("All parallel dataset processes have completed.")