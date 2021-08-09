# GCP Cloud TPU sandbox

##  TPU Node quickstart

### Set the default GCP project and zone

```
gcloud config set project jk-mlops-dev
gcloud config set compute/zone us-central1-a
```

### Create a TPU node

```
gcloud compute tpus create jk-tpu-node \
--accelerator-type v3-8 \
--version 2.5.0 
```
### Create a user VM

A user VM can use any base image as long as the pre-requistes like TensorFlow are pre-installed or can be installed. E.g. one of GCP standard deep learning images. 

When provisioning a user VM using the `gcloud compute tpus execution-groups create` command an image from the `ml-images` project is used. E.g. when you create an execution group for TensorFlow 2.5 the tf-2-5-0 image family is used. The images from the `ml-images` project have some TPU samples and pre-requisites pre-installed. 

If you want to mirror what the `gcloud compute tpus execution-groups create` does by using separate `gcloud compute tpus create` and `gcloud compute instance create` commands to create a user VM without an external IP address, use the following settings.

```
VM_NAME=jk-user-vm
IMAGE_FAMILY=tf-2-5-0
IMAGE_PROJECT=ml-images
MACHINE_TYPE=n1-standard-8

gcloud compute instances create jk-user-vm \
--machine-type $VM_NAME \
--image-family $IMAGE_FAMILY \
--image-project $IMAGE_PROJECT \
--scopes https://www.googleapis.com/auth/cloud-platform \
--no-address
```

## TPU VM quickstart


### Creating a TPU VM with an external IP address

```
gcloud alpha compute tpus tpu-vm create \
jk-tpu-vm-2 \
--accelerator-type v3-8 \
--version v2-alpha \
```

### Creating a TPU VM without an external IP address

#### Enable Private Google Access
```
gcloud compute networks subnets update default \
--region us-central1 \
--enable-private-ip-google-access
```

```
gcloud compute networks subnets describe default \
--region=us-central1 \
--format="get(privateIpGoogleAccess)"
```

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


# Parking lot

### Creating a TPU execution group

```
gcloud compute tpus execution-groups create \
--name jk-tpu-exec-group \
--zone us-central1-a \
--machine-type n1-standard-8 \
--accelerator-type v3-8 \
--tf-version 2.5.0 
```


