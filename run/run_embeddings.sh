python3.7 run/run_embeddings.py \
    --triplet-generator "quinn" \
    --stimulus-generators "different_shapes" \
    --stimulus-generators "split_text" \
    --stimulus-generators "random_color" \
    --model "resnext" \
    --flipping "s" \
    --batch-size 8 \
    --output-file "embedding_outputs/flippiing_s.csv" \
    --tqdm