import tensorflow as tf
import time

#tf.debugging.set_log_device_placement(True)


resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(resolver)

#tf.tpu.experimental.shutdown_tpu_system(
#    cluster_resolver=resolver
#)

# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

@tf.function()#jit_compile=True)
def running_example(x, y):
  return tf.reduce_mean(tf.multiply(x**2, 3) + y)


x = tf.random.uniform((16384, 16384))
y = tf.random.uniform((16384, 16384))
 
#with tf.device('/TPU:0'):
#    z = running_example(x, y)
#    print(z)

logdir = 'gs://jk-tensorboard-logs/tpu-profiling'
options = None

options = tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=1, python_tracer_level=0, device_tracer_level=1, delay_ms=None
)

strategy = tf.distribute.TPUStrategy(resolver)

with tf.profiler.experimental.Profile(logdir, options=options):
    z = strategy.run(running_example, args=(x, y))
    print(z)
    time.sleep(2)