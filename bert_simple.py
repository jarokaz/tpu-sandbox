from absl import app, flags, logging
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from typing import Any, Callable, Dict, List, Optional
import os

#model_uri = 'gs://twttr-mlte-dev-workspace/model/bert_en_cased_L-24_H-1024_A-16_3/'
model_uri = 'gs://tfhub-modules/tensorflow/bert_en_cased_L-24_H-1024_A-16/3/uncompressed'
#preprocessor_uri = 'gs://twttr-mlte-dev-workspace/model/bert_en_cased_preprocess_3/'
preprocessor_uri = 'gs://tfhub-modules/tensorflow/bert_en_cased_preprocess/3/uncompressed'


def tpu_initialize(tpu_address):
  cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=tpu_address)
  if tpu_address not in ("", "local"):
    tf.config.experimental_connect_to_cluster(cluster_resolver)
  tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
  return cluster_resolver

def get_tpu_strategy(tpu_address):
  cluster_resolver = tpu_initialize(tpu_address)
  return tf.distribute.TPUStrategy(cluster_resolver)

def get_bert_model():
#    preprocessor = hub.load(handle=preprocessor_uri)
#    # Tokenize
#    tokenizer = hub.KerasLayer(preprocessor.tokenize, trainable=False)    
#    # PACK
#    seq_length = 128  # Your choice here.
#    bert_pack_inputs = hub.KerasLayer(
#      preprocessor.bert_pack_inputs,
#      arguments=dict(seq_length=seq_length)
#    )  # Optional argument.
    
    bert = hub.KerasLayer(
      handle=model_uri, 
      trainable=True,
      name='encoder'
    )
    preprocessing_layer = hub.KerasLayer(
        preprocessor_uri, 
        name='preprocessing')

    #text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    #Tokenize
 #   tokenized_inputs = [tokenizer(segment) for segment in [text_input]]
    #PACK
 #   encoder_inputs = bert_pack_inputs(tokenized_inputs)
    
#    model_output = bert(encoder_inputs)

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    
    encoder_inputs = preprocessing_layer(text_input)
    model_output = bert(encoder_inputs)

    embedding = model_output["pooled_output"]
    logits = tf.keras.layers.Dense(1)(embedding)
    softmax_prob = tf.keras.layers.Softmax()(logits)
    model = tf.keras.models.Model(inputs=text_input, outputs=softmax_prob)
    return model


def main(_): 
  strategy = get_tpu_strategy('jk-tpu-node')
  #strategy = tf.distribute.get_strategy()
  logging.info('Init Done!')
  tx = tf.constant(['abc def','xyz abc apple', 'abc xyz', 'appel pen'], dtype=tf.string)
  ty = tf.constant([1, 1, 0, 0], dtype=tf.int64)
  vx = tf.constant(['apple peer','abc apple', 'xyz', 'vvvv'], dtype=tf.string)
  vy = tf.constant([1, 0, 0, 1], dtype=tf.int64)
    
  #train_ds = tf.data.Dataset.from_tensor_slices((tx, ty)).batch(2)
  #valid_ds = tf.data.Dataset.from_tensor_slices((vx, vy)).batch(2)
  with strategy.scope():
    model = get_bert_model()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
    model.fit(tx, ty, epochs=2, batch_size=2, validation_data=(vx, vy), verbose=True)
    #model.fit(x=train_ds, validation_data=valid_ds)
  model.summary()

if __name__ == '__main__':
  app.run(main)
