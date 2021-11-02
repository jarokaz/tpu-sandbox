

## List available subnets

```
export HOST_PROJECT=jk-sharedvpc-hostproject
gcloud compute networks subnets list-usable --project $HOST_PROJECT
```


## Creating a TPU node on a shared VPC


```
export HOST_PROJECT=jk-sharedvpc-hostproject
export HOST_NETWORK=projects/$HOST_PROJECT/global/networks/mynetwork
export ACCELERATOR_TYPE=v3-8
export ACCELERATOR_VERSION=2.6.0

gcloud beta compute tpus create jk-tpu-node501 --zone us-central1-a \
--accelerator-type v3-8 --network $HOST_NETWORK --use-service-networking \
--version $ACCELERATOR_VERSION
```


## Creating a TPU VM on a shared VPC


### With an external IP address


```
export HOST_PROJECT=jk-sharedvpc-hostproject
export SUBNET=mysubnet
export NETWORK=mynetwork
export ZONE=us-central1-b
export REGION=us-central1
export HOST_SUBNET=projects/$HOST_PROJECT/regions/$REGION/subnetworks/$SUBNET
export HOST_NETWORK=projects/$HOST_PROJECT/global/networks/$NETWORK
export ACCELERATOR_TYPE=v3-8
export ACCELERATOR_VERSION=v2-alpha

gcloud alpha compute tpus tpu-vm create \
jk-tpu-vm-502 \
--accelerator-type v3-8 \
--version v2-alpha \
--zone $ZONE \
--network $HOST_NETWORK \
--subnetwork $HOST_SUBNET 


```
