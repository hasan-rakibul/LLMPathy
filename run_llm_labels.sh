#!/bin/bash

python src/llm_labels.py \
--file_path=data/NewsEmp2024/trac3_EMP_train.csv \
2>&1 | tee logs/llm_labels_2024_train.txt
