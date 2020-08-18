# Copyright 2020 The Google Research Authors.
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

"""Pre-trains an ELECTRA model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json

import tensorflow.compat.v1 as tf

import configure_pretraining
from model import modeling
from model import optimization
from pretrain import pretrain_data
from pretrain import pretrain_helpers
from util import training_utils
from util import utils
from util import gen
import sys

class PretrainingModel(object):
  """Transformer pre-training using the replaced-token-detection task."""

  def __init__(self, config: configure_pretraining.PretrainingConfig,
               features, is_training):
    # Set up model config
    self._config = config
    self._bert_config = training_utils.get_bert_config(config)
	
    inputs = pretrain_data.features_to_inputs(features)
    embedding_size = (
			self._bert_config.hidden_size if config.embedding_size is None else config.embedding_size)
    fake_data = self._get_fake_data(inputs)

    # Discriminator
    disc_output = None
    if config.electra_objective:
      discriminator = self._build_transformer(
          fake_data.inputs, is_training, reuse=False,
          embedding_size=embedding_size)
      disc_output = self._get_discriminator_output(
          fake_data.inputs, discriminator, fake_data.is_fake_tokens)
      self.total_loss = disc_output.loss

    # Evaluation
    eval_fn_inputs = {
        "input_ids": inputs.input_ids,
        "masked_lm_ids": inputs.masked_lm_ids,
        "masked_lm_weights": inputs.masked_lm_weights,
        "input_mask":inputs.label 
    }
    if config.electra_objective:
      eval_fn_inputs.update({
          "disc_loss": disc_output.per_example_loss,
          "disc_labels": disc_output.labels,
          "disc_probs": disc_output.probs,
          "disc_preds": disc_output.preds,
      })
    eval_fn_keys = eval_fn_inputs.keys()
    eval_fn_values = [eval_fn_inputs[k] for k in eval_fn_keys]

    def metric_fn(*args):
      """Computes the loss and accuracy of the model."""
      d = {k: arg for k, arg in zip(eval_fn_keys, args)}
      metrics = dict()
      metrics["masked_lm_accuracy"] = tf.metrics.accuracy(
          labels=tf.reshape(d["masked_lm_ids"], [-1]),
          predictions=tf.reshape(d["masked_lm_preds"], [-1]),
          weights=tf.reshape(d["masked_lm_weights"], [-1]))
      metrics["masked_lm_loss"] = tf.metrics.mean(
          values=tf.reshape(d["mlm_loss"], [-1]),
          weights=tf.reshape(d["masked_lm_weights"], [-1]))
      if config.electra_objective:
        metrics["sampled_masked_lm_accuracy"] = tf.metrics.accuracy(
            labels=tf.reshape(d["masked_lm_ids"], [-1]),
            predictions=tf.reshape(d["sampled_tokids"], [-1]),
            weights=tf.reshape(d["masked_lm_weights"], [-1]))
        if config.disc_weight > 0:
          metrics["disc_loss"] = tf.metrics.mean(d["disc_loss"])
          metrics["disc_auc"] = tf.metrics.auc(
              d["disc_labels"] * d["input_mask"],
              d["disc_probs"] * tf.cast(d["input_mask"], tf.float32))
          metrics["disc_accuracy"] = tf.metrics.accuracy(
              labels=d["disc_labels"], predictions=d["disc_preds"],
              weights=d["input_mask"])
          metrics["disc_precision"] = tf.metrics.accuracy(
              labels=d["disc_labels"], predictions=d["disc_preds"],
              weights=d["disc_preds"] * d["input_mask"])
          metrics["disc_recall"] = tf.metrics.accuracy(
              labels=d["disc_labels"], predictions=d["disc_preds"],
              weights=d["disc_labels"] * d["input_mask"])
      return metrics
    self.eval_metrics = (metric_fn, eval_fn_values)

  def _get_discriminator_output(self, inputs, discriminator, labels):
    """Discriminator binary classifier."""
    with tf.variable_scope("discriminator_predictions"):
      hidden = tf.layers.dense(
          discriminator.get_sequence_output(),
          units=self._bert_config.hidden_size,
          activation=modeling.get_activation(self._bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              self._bert_config.initializer_range))
      logits = tf.squeeze(tf.layers.dense(hidden, units=1), -1)
      weights = tf.cast(inputs.input_mask, tf.float32)
      labelsf = tf.cast(labels, tf.float32)
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=logits, labels=labelsf) * weights
      per_example_loss = (tf.reduce_sum(losses, axis=-1) /
                          (1e-6 + tf.reduce_sum(weights, axis=-1)))
      loss = tf.reduce_sum(losses) / (1e-6 + tf.reduce_sum(weights))
      probs = tf.nn.sigmoid(logits)
      preds = tf.cast(tf.round((tf.sign(logits) + 1) / 2), tf.int32)
      DiscOutput = collections.namedtuple(
          "DiscOutput", ["loss", "per_example_loss", "probs", "preds",
                         "labels"])
      return DiscOutput(
          loss=loss, per_example_loss=per_example_loss, probs=probs,
          preds=preds, labels=labels,
      )

  def _get_fake_data(self, inputs):
    """Sample from the generator to create corrupted input."""
    
    fake_data = inputs
    labels = inputs.label 
    FakedData = collections.namedtuple("FakedData", [
        "inputs", "is_fake_tokens"])
    return FakedData(inputs=fake_data, is_fake_tokens=labels)

  def _build_transformer(self, inputs: pretrain_data.Inputs, is_training,
                         bert_config=None, name="electra", reuse=False, **kwargs):
    """Build a transformer encoder network."""
    if bert_config is None:
      bert_config = self._bert_config
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
      return modeling.BertModel(
          bert_config=bert_config,
          is_training=is_training,
          input_ids=inputs.input_ids,
          input_mask=inputs.input_mask,
          token_type_ids=None,#inputs.segment_ids,
          use_one_hot_embeddings=self._config.use_tpu,
          scope=name,
          **kwargs)


def get_generator_config(config: configure_pretraining.PretrainingConfig,
                         bert_config: modeling.BertConfig):
  """Get model config for the generator network."""
  gen_config = modeling.BertConfig.from_dict(bert_config.to_dict())
  gen_config.hidden_size = int(round(
      bert_config.hidden_size * config.generator_hidden_size))
  gen_config.num_hidden_layers = int(round(
      bert_config.num_hidden_layers * config.generator_layers))
  gen_config.intermediate_size = 4 * gen_config.hidden_size
  gen_config.num_attention_heads = max(1, gen_config.hidden_size // 64)
  return gen_config


def model_fn_builder(config: configure_pretraining.PretrainingConfig):
  """Build the model for training."""

  def model_fn(features, labels, mode, params):

    """Build the model for training."""
    model = PretrainingModel(config, features, 
                             mode == tf.estimator.ModeKeys.TRAIN)
    utils.log("Model is built!")
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          model.total_loss, config.learning_rate, config.num_train_steps,
          weight_decay_rate=config.weight_decay_rate,
          use_tpu=config.use_tpu,
          warmup_steps=config.num_warmup_steps,
          lr_decay_power=config.lr_decay_power
      )
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=model.total_loss,
          train_op=train_op,
          training_hooks=[training_utils.ETAHook(
              {} if config.use_tpu else dict(loss=model.total_loss),
              config.num_train_steps, config.iterations_per_loop,
              config.use_tpu)]
      )
    elif mode == tf.estimator.ModeKeys.EVAL:
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=model.total_loss,
          eval_metrics=model.eval_metrics,
          evaluation_hooks=[training_utils.ETAHook(
              {} if config.use_tpu else dict(loss=model.total_loss),
              config.num_eval_steps, config.iterations_per_loop,
              config.use_tpu, is_training=False)])
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported")
    return output_spec

  return model_fn

def train_or_eval(config: configure_pretraining.PretrainingConfig, _random):
  """Run pre-training or evaluate the pre-trained model."""
  if config.do_train == config.do_eval:
    raise ValueError("Exactly one of `do_train` or `do_eval` must be True.")
  if config.debug and config.do_train:
    utils.rmkdir(config.model_dir)
  utils.heading("Config:")
  utils.log_config(config)

  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  tpu_cluster_resolver = None
  if config.use_tpu and config.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        config.tpu_name, zone=config.tpu_zone, project=config.gcp_project)
  tpu_config = tf.estimator.tpu.TPUConfig(
      iterations_per_loop=config.iterations_per_loop,
      num_shards=config.num_tpu_cores,
      tpu_job_name=config.tpu_job_name,
      per_host_input_for_training=is_per_host)
  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=config.model_dir,
      save_checkpoints_steps=config.save_checkpoints_steps,
      tpu_config=tpu_config)
  model_fn = model_fn_builder(config=config)
  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=config.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=config.train_batch_size,
      eval_batch_size=config.eval_batch_size)

  if config.do_train:
    utils.heading("Running training")
    if _random:
	    print("import random")
	    estimator.train(
			 input_fn=pretrain_data.get_input_fn(config, True, filename='random'),
                    max_steps=config.num_train_steps)
    else:
	    print("import wordnet")
	    estimator.train(
			 input_fn=pretrain_data.get_input_fn(config, True, filename='wordnet'),
                    max_steps=config.num_train_steps)


  if config.do_eval:
    utils.heading("Running evaluation")
    if _random:
      result = estimator.evaluate(
        input_fn=pretrain_data.get_input_fn(config, False, filename='random'),
        steps=config.num_eval_steps)
    else:
      result = estimator.evaluate(
        input_fn=pretrain_data.get_input_fn(config, False, filename='wordnet'),
        steps=config.num_eval_steps)
    for key in sorted(result.keys()):
      utils.log("  {:} = {:}".format(key, str(result[key])))
    return result


def train_one_step(config: configure_pretraining.PretrainingConfig):
  """Builds an ELECTRA model an trains it for one step; useful for debugging."""
  train_input_fn = pretrain_data.get_input_fn(config, True, filename="wordnet")
  features = tf.data.make_one_shot_iterator(train_input_fn(dict(
      batch_size=config.train_batch_size))).get_next()
 
  model = PretrainingModel(config, features, True)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    utils.log(sess.run(model.total_loss))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", required=True,
                      help="Location of data files (model weights, etc).")
  parser.add_argument("--model-name", required=True,
                      help="The name of the model being fine-tuned.")
  parser.add_argument("--hparams", default="{}",
                      help="JSON dict of model hyperparameters.")
  parser.add_argument("--random", action="store_true")
  args = parser.parse_args()
  if args.hparams.endswith(".json"):
    hparams = utils.load_json(args.hparams)
  else:
    hparams = json.loads(args.hparams)
  tf.logging.set_verbosity(tf.logging.ERROR)
  train_or_eval(configure_pretraining.PretrainingConfig(
      args.model_name, args.data_dir, **hparams), args.random)
#  train_one_step(configure_pretraining.PretrainingConfig(
#			  args.model_name, args.data_dir, **hparams))

if __name__ == "__main__":
  main()
