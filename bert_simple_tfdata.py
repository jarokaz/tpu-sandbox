from absl import app, flags, logging
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from typing import Any, Callable, Dict, List, Optional
import os

MODEL_URI = 'gs://tfhub-modules/tensorflow/bert_en_cased_L-24_H-1024_A-16/3/uncompressed'
PREPROCESSOR_URI = 'gs://tfhub-modules/tensorflow/bert_en_cased_preprocess/3/uncompressed'


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


def create_bert_preprocess_model(seq_length=128):

  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')

  bert_preprocess = hub.load(PREPROCESSOR_URI)
  tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
  packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                          arguments=dict(seq_length=seq_length),
                          name='packer')                                  
                                     
  tokenized_inputs = [tokenizer(segment) for segment in [text_input]]
  model_inputs = packer(tokenized_inputs)
                                     
  return tf.keras.Model(text_input, model_inputs)


def get_bert_model(seq_len=128):
    
    bert_layer = hub.KerasLayer(
        handle=MODEL_URI, 
        trainable=True,
        name='encoder'
    )
    encoder_inputs = dict(
        input_word_ids = tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32, name="input_word_ids"),
        input_mask = tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32, name="input_mask"),
        input_type_ids = tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32, name="segment_ids"),
    )
    encoder_outputs = bert_layer(encoder_inputs)
    embedding = encoder_outputs["pooled_output"]
    logits = tf.keras.layers.Dense(1)(embedding)
    softmax_prob = tf.keras.layers.Softmax()(logits)
    model = tf.keras.models.Model(inputs=encoder_inputs,
                                  outputs=softmax_prob)
    
    return model


def create_dataset(inputs, labels, batch_size, preprocessor):
  dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(lambda x, y: (preprocessor(x), y))
  return dataset


def main(_): 

  logging.info('Init Done!')


  preprocessor = create_bert_preprocess_model()

  tx = tf.constant(['abc def','xyz abc apple', 'abc xyz', 'appel pen'], dtype=tf.string)
  ty = tf.constant([1, 1, 0, 0], dtype=tf.int64)
  vx = tf.constant(['apple peer','abc apple', 'xyz', 'vvvv'], dtype=tf.string)
  vy = tf.constant([1, 0, 0, 1], dtype=tf.int64)
    
  train_ds = create_dataset(tx, ty, 2, preprocessor)
  valid_ds = create_dataset(vx, vy, 2, preprocessor)
    
  strategy = get_tpu_strategy('jk-tpu-node')
  #strategy = tf.distribute.get_strategy()
  with strategy.scope():


    model = get_bert_model()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
    model.summary()
    

    model.fit(train_ds, validation_data=valid_ds, verbose=True)
  

if __name__ == '__main__':
  app.run(main)
