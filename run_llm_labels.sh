#!/bin/bash

python src/llm_labels.py \
--file_path=data/NewsEmp2023/WASSA23_essay_level_dev.tsv \
2>&1 | tee logs/llm_label/llm_labels_2023_dev.txt
