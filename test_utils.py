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
"""Utilities for testing."""

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub


def create_dummy_hub_module(num_outputs):
  """Creates minimal hub module for testing purposes."""

  def module_fn():
    x = tf.placeholder(dtype=tf.float32, shape=[32, 224, 224, 3])
    h = tf.reduce_mean(x, axis=[1, 2])
    y = tf.layers.dense(h, num_outputs)
    hub.add_signature(inputs=x, outputs={"pre_logits": h, "logits": y})

  return hub.create_module_spec(
      module_fn, tags_and_args=[({"train"}, {}), (set(), {})])


def get_data_params():
  return {
      "dataset_name": "data.caltech101",
      "dataset_train_split_name": "train",
      "dataset_eval_split_name": "test",
      "shuffle_buffer_size": 1000,
      "prefetch": 1000,
      "num_train_examples": None,
      "batch_size": 32,
      "batch_size_eval": 8,
      "data_dir": None,
      "image_size": 224,
      "input_range": [-1.0, 1.0],
  }


def get_optimization_params():
  return {
      "finetune_layer": "pre_logits",
      "initial_learning_rate": 0.01,
      "momentum": 0.9,
      "lr_decay_factor": 0.1,
      "decay_steps": (10, 20, 30),
      "max_steps": 10,
      "warmup_steps": 0,
  }
