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
"""TPU Hello World."""

import os
import tensorflow as tf

from absl import flags
from absl import app
from absl import logging


FLAGS = flags.FLAGS
    
   
def main(argv):

    logging.info(f"Testing TPU: {FLAGS.tpu_name}")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu_name)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    logging.info("All devices: {}".format(tf.config.list_logical_devices('TPU')))


flags.DEFINE_string('tpu_name', 'local', 'TPU Name')

if __name__ == '__main__':
   app.run(main)
