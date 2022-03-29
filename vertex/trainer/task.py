# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tensorflow as tf

from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

from absl import logging
from absl import flags
from absl import app


def train_eval():
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))

def _main(argv):
    train_eval()


FLAGS = flags.FLAGS

flags.DEFINE_list('training_data_paths', None, 'Paths to training datasets')
flags.DEFINE_list('validation_data_paths', None, 'Paths to validation datasets') 
flags.DEFINE_integer('tpu_cores', 8, 'A number of TPU cores')
flags.DEFINE_string('tpu_type', 6, 'TPU type: 6 = TPU_V2, 7 = TPU_V3')


if __name__=='__main__':
    #flags.mark_flags_as_required([
    #    'fasta_path',
    #    'database_paths',
    #    'output_dir'
    #])
    app.run(_main)

