# GCP Cloud TPU sandbox

## Using TPU nodes

### Creating a TPU execution group

```
gcloud compute tpus execution-groups create \
--name jk-tpu-exec-group \
--zone us-central1-a \
--machine-type n1-standard-8 \
--accelerator-type v3-8 \
--tf-version 2.5.0 
```

### Creating a TPU node

```
gcloud compute tpus create jk-tpu-node-2 \
--accelerator-type v3-8 \
--version 2.5.0
```


### Creating a TPU VM with an external IP address

```
gcloud alpha compute tpus tpu-vm create \
jk-tpu-vm-2 \
--accelerator-type v3-8 \
--version v2-alpha \
```

### Creating a TPU VM with an external IP address

```
gcloud alpha compute tpus tpu-vm create \
jk-tpu-vm-2 \
--accelerator-type v3-8 \
--version v2-alpha \
--internal-ips
```


## Installing software components

- Don't use TF 2.5 image. Start with a Python image and install TF 2.5

```
pip install tensorflow tensorflow-datasets cloud-tpu-client
```





