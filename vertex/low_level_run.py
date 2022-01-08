# Copyright (c) 2021 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Standard Libraries
import argparse
import logging
import pprint
import time
import datetime
import json

import google.auth
from google.auth.transport.requests import AuthorizedSession


from google.cloud import aiplatform
from google.cloud.aiplatform_v1beta1 import types
from google.cloud.aiplatform_v1beta1.services.job_service import JobServiceClient



def run(args):

    job_name = 'TPU_JOB_{}'.format(time.strftime("%Y%m%d_%H%M%S"))

    job_client = MyJobServiceClient(args.project, args.region)

    logging.info(f'Starting job: {job_name}')

    worker_pool_specs =  [
        {
            "machine_spec": {
                "machine_type": args.machine_type,
                "accelerator_type": args.accelerator_type,
                "accelerator_count": args.accelerator_num,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": args.image,
                #"command": ["nvidia-smi"],
                #"args": [ ],
            },
        }
        ]

    custom_job_spec = {
            'display_name': job_name,
            'job_spec': {
                'worker_pool_specs': worker_pool_specs
            }
        }

    pp = pprint.PrettyPrinter()
    print(pp.pformat(custom_job_spec))



    authed_session = AuthorizedSession(credentials)

    parent = f'projects/{args.project}/locations/{args.region}'

    response = authed_session.get(
       'https://www.googleapis.com/storage/v1/b')

    response = job_client.create_custom_job(parent, custom_job_spec)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--project',
                        type=str,
                        default='jk-mlops-dev',
                        help='Project ID')
    parser.add_argument('--region',
                        type=str,
                        default='us-central1',
                        help='Region')
    parser.add_argument('--gcs_bucket',
                        type=str,
                        default='gs://jk-vertex-us-central1',
                        help='GCS bucket')
    parser.add_argument('--vertex_sa',
                        type=str,
                        default='training-sa@jk-mlops-dev.iam.gserviceaccount.com',
                        help='Vertex SA')
    parser.add_argument('--machine_type',
                        type=str,
                        default='cloud-tpu',
                        help='Machine type')
    parser.add_argument('--accelerator_type',
                        type=str,
                        default='TPU_V3',
                        help='Accelerator type')
    parser.add_argument('--accelerator_num',
                        type=int,
                        default=8,
                        help='Num of GPUs')
    parser.add_argument('--image',
                        type=str,
                        default='gcr.io/jk-mlops-dev/tpu-helloworld',
                        help='Training image name')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')

    logging.info(f"Args: {args}")

    run(args)