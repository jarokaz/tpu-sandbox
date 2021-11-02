## Creating a TPU node on a shared VPC


```
export HOST_NETWORK=projects/jk-sharedvpc-hostproject/global/networks/mynetwork
export ACCELERATOR_TYPE=v3-8
export ACCELERATOR_VERSION=2.6.0

gcloud beta compute tpus create jk-tpu-node501 --zone us-central1-a \
--accelerator-type v3-8 --network $HOST_NETWORK --use-service-networking \
--version $ACCELERATOR_VERSION
```
