#!/bin/bash

python src/llm_labels.py \
--file_path=data/NewsEmp2022/messages_test_features_ready_for_WS_2022_llm.tsv \
2>&1 | tee logs/llm_labels_2022_test_4.txt
