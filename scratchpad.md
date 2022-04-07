## Scratch pad

```
gcloud alpha compute tpus tpu-vm create jk-tpu-vm-32 \
--zone us-east1-d \
--accelerator-type=v3-32 \
--version=tpu-vm-tf-2.8.0-pod \
--scopes https://www.googleapis.com/auth/cloud-platform
```

```
export TPU_NAME=jk-tpu-vm-32
export ZONE=us-central1-a


gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=all  --command="sudo systemctl stop tpu-runtime" --zone $ZONE
```

```
export ZONE=us-east1
export STORAGE_BUCKET=gs://jk-bucket-east1
export TPU_NAME=jk-tpu-vm-32
export DATA_DIR=gs://cloud-tpu-test-datasets/fake_imagenet
export MODEL_DIR=${STORAGE_BUCKET}/resnet-2x-pod

export TPU_LOAD_LIBRARY=0

python3 tpu_test.py --tpu_name $TPU_NAME
```

```
export CLOUDSDK_PYTHON=/usr/bin/python3
```

```
cd /usr/share/tpu/models/official/vision/image_classification/resnet

python3 resnet_ctl_imagenet_main.py \
  --tpu=${TPU_NAME} \
  --model_dir=${MODEL_DIR} \
  --data_dir=${DATA_DIR} \
  --batch_size=4096 \
  --steps_per_loop=500 \
  --train_epochs=1 \
  --use_synthetic_data=false \
  --dtype=fp32 \
  --enable_eager=true \
  --enable_tensorboard=true \
  --distribution_strategy=tpu \
  --log_steps=50 \
  --single_l2_loss_op=true \
  --use_tf_function=true
```