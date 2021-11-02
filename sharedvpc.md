

## Grant TPU service account `networkUser` permissions in the host project

```
export HOST_PROJECT=jk-sharedvpc-hostproject
export TPU_SA="user:service-464426275157@gcp-sa-tpu.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding $HOST_PROJECT \
--member  $TPU_SA \
--role "roles/compute.networkUser"
```

## List available subnets

```

gcloud compute networks subnets list-usable --project $HOST_PROJECT
```


## Creating a TPU node on a shared VPC


```
export HOST_PROJECT=jk-sharedvpc-hostproject
export SUBNET=mysubnet
export ZONE=us-central1-b
export REGION=us-central1
export HOST_NETWORK=projects/$HOST_PROJECT/global/networks/mynetwork
export HOST_SUBNET=projects/$HOST_PROJECT/regions/$REGION/subnetworks/$SUBNET
export ACCELERATOR_TYPE=v2-8
export ACCELERATOR_VERSION=2.6.0
export RANGE=10.0.0.0/29

gcloud beta compute tpus create jk-tpu-node501 \
--zone $ZONE \
--accelerator-type $ACCELERATOR_TYPE \
--version $ACCELERATOR_VERSION \
--network $HOST_NETWORK \
--use-service-networking 

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
