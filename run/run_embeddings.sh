python3.7 run/run_embeddings.py \
    --triplet-generator "quinn" \
    --stimulus-generators "split_text" \
    --model "resnext" \
    --saycam "S"  \
    --batch-size 32 \
    --output-file "embedding_outputs/saycam_split_text_original_test.csv" \
    --tqdm \
    --print-setting-options