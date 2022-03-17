python3.7 run/run_embeddings.py \
    --triplet-generator "quinn" \
    --stimulus-generators "different_shapes" \
    --stimulus-generators "split_text" \
    --stimulus-generators "random_color" \
    --extra-diagonal-margin 5 \
    --model "mobilenet" \
    --model "resnext" \
    --untrained \
    --output-file "embedding_outputs/untrained_models.csv" \
    --tqdm \
    --profile
    