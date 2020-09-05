#!/usr/bin/env bash
python -m src.train \
--name my_model \
--dataset_json data/dataset_jsons/megadepth/bigben.json \
--box_ndim 32 \
--batch_size 32 \
--model resnet50 \
--num_gpus 1 \
--backend dp