# Does the original preprocessing of DepthLM on our desired dataset

#125 image
python utils/curate_nuscenes_eval.py \
  --dataroot /mnt/dgx_lab/datasets/nuscenes_mini \
  --out_json_path  /mnt/dgx_lab/datasets/depthLMoriginal/nu/nuscenes_mini.jsonl \
  --out_image_dir /mnt/dgx_lab/datasets/depthLMoriginal/nu/processed_images_mini

# over 10k image

python utils/curate_sunRGBD.py \
--dataroot /mnt/dgx_lab/datasets/sun \
--out_json_path /mnt/dgx_lab/datasets/depthLMoriginal/s/output_jsonl/sunRGBD.jsonl \
--out_image_dir /mnt/dgx_lab/datasets/depthLMoriginal/sun/output_image_folder
## 3. curate data for NYUv2
python utils/curate_NYU.py \
--dataroot /mnt/dgx_lab/datasets/sun \
--out_json_path /mnt/dgx_lab/datasets/depthLMoriginal/nyu/output_jsonl/NYUv2.jsonl \
--out_image_dir /mnt/dgx_lab/datasets/depthLMoriginal/nyu/output_image_folder