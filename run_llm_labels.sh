#!/bin/bash

python src/llm_labels.py \
--openai \
--file_path=data/NewsEmp2022/messages_test_features_ready_for_WS_2022.tsv \
2>&1 | tee logs/llm_label/llm_labels_gpt_2022_test.txt
