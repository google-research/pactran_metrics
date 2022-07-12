# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
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
"""Model that runs a given hub-module."""

from pactran_metrics import trainer
import tensorflow.compat.v1 as tf
from tensorflow_estimator.compat.v1 import estimator as tf_estimator
import tensorflow_hub as hub


def model_fn(features, mode, params):
  """A function for applying hub module that follows Estimator API."""
  if mode not in (tf_estimator.ModeKeys.EVAL, tf_estimator.ModeKeys.TRAIN,
                  tf_estimator.ModeKeys.PREDICT):
    raise ValueError("mode must be in '[TRAIN, EVAL, Prediction]'"
                     "but got unexpected value: %s" % (mode))
  hub_module = params.get("hub_module")
  finetune_layer = params.get("finetune_layer")
  num_classes = params.get("num_classes")
  initial_learning_rate = params.get("initial_learning_rate")
  momentum = params.get("momentum")
  lr_decay_factor = params.get("lr_decay_factor")
  decay_steps = params.get("decay_steps")
  warmup_steps = params.get("warmup_steps")
  use_anil = params.get("use_anil")

  is_training = (mode == tf_estimator.ModeKeys.TRAIN)
  module = hub.Module(hub_module,
                      trainable=is_training,
                      tags={"train"} if is_training else None)

  pre_logits = module(features["image"],
                      signature=params["hub_module_signature"],
                      as_dict=True)[finetune_layer]

  if use_anil:
    pre_logits = tf.stop_gradient(pre_logits)

  num_dim_pre_logits = len(pre_logits.get_shape().as_list())
  if num_dim_pre_logits == 4:
    pre_logits = tf.squeeze(pre_logits, [1, 2])
  elif num_dim_pre_logits != 2:
    raise ValueError("Invalid number of dimensions in the representation "
                     "layer: {}, but only 2 or 4 are allowed".format(
                         num_dim_pre_logits))

  logits = tf.keras.layers.Dense(
      units=num_classes,
      kernel_initializer=tf.zeros_initializer(),
      name="linear_head")(
          pre_logits)

  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=features["label"])
  loss = tf.reduce_mean(loss)

  def accuracy_metric(logits, labels):
    return {"accuracy": tf.metrics.accuracy(
        labels=labels,
        predictions=tf.argmax(logits, axis=-1))}
  eval_metrics = (accuracy_metric, [logits, features["label"]])

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      "logits": logits,
      # Add `softmax_tensor` to the graph.
      # It is used for PREDICT and by the `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
      "features": pre_logits,
      "labels": features["label"]
    }

  if mode == tf_estimator.ModeKeys.PREDICT:
    return tf_estimator.EstimatorSpec(
        mode=mode, predictions=predictions, scaffold=None)

  elif mode == tf_estimator.ModeKeys.EVAL:
    return tf_estimator.EstimatorSpec(
        mode=mode, loss=loss,
        eval_metric_ops=eval_metrics[0](*eval_metrics[1]))
  elif mode == tf_estimator.ModeKeys.TRAIN:
    train_op = trainer.get_group_of_ops(
        loss,
        initial_learning_rate,
        momentum,
        lr_decay_factor,
        decay_steps,
        warmup_steps)
    spec_type = (
        tf_estimator.EstimatorSpec)
    return spec_type(mode=mode, loss=loss, train_op=train_op)
  else:
    raise ValueError("Unsupported mode %s" % mode)
