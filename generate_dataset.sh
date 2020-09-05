#!/usr/bin/env bash
dataset_output_path=data/overlap_data/megadepth/my_data
dataset_json_path=data/dataset_jsons/megadepth/bigben.json
python -m src.datasets.dataset_generator.compute_normals \
--dataset_json $dataset_json_path \
--output_folder $dataset_output_path
python -m src.datasets.dataset_generator.compute_overlap \
--dataset_json $dataset_json_path \
--normals_folder $dataset_output_path \
--num_sampled_points 5000 \
--num_pairs 25 \
--threshold 0.1