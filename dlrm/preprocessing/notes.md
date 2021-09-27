```

export PROJECT=jk-mlops-dev
export REGION=us-central1
export STORAGE_BUCKET=gs://jk-criteo-bucket


python3 shard_rebalancer.py \
  --input_path "${STORAGE_BUCKET}/criteo_raw/train/*" \
  --output_path "${STORAGE_BUCKET}/criteo_raw_sharded/train/train" \
  --num_output_files 1024 --filetype csv --runner DataflowRunner \
  --project ${PROJECT} --region ${REGION}

```