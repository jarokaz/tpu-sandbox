# GCP Cloud TPU sandbox

## Using TPU nodes

### Creating a TPU execution group

```
gcloud compute tpus execution-group create \
--name jk-tpu-exec-group \
--zone us-central1-a \
--machine-type n1-standard-8 \
--accelerator-type v3-8 \
--tf-version 2.5.0 \
--use-with-notebook
```


```
gcloud compute tpus create 

## Installing software components

- Don't use TF 2.5 image. Start with a Python image and install TF 2.5

```
pip install tensorflow tensorflow-datasets cloud-tpu-client
```





