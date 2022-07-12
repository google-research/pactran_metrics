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
"""Tests for the model module."""

from absl.testing import absltest
from pactran_metrics import model
from pactran_metrics import test_utils
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_estimator.compat.v1 import estimator as tf_estimator


class ModelTest(absltest.TestCase):

  def test_model_fn(self):
    num_classes = 1000
    params_list = [
        test_utils.get_optimization_params(),
        test_utils.get_data_params(), {
            "hub_module": test_utils.create_dummy_hub_module(num_classes),
            "hub_module_signature": None,
            "num_classes": num_classes
        }
    ]
    params = {}
    for d in params_list:
      for k, v in d.items():
        params[k] = v

    for mode in [tf_estimator.ModeKeys.TRAIN, tf_estimator.ModeKeys.EVAL]:
      tf.reset_default_graph()
      images = tf.constant(np.random.random([32, 224, 224, 3]),
                           dtype=tf.float32)
      labels = tf.constant(np.random.randint(0, 1000, [32]),
                           dtype=tf.int64)
      model.model_fn({"image": images, "label": labels}, mode, params)


if __name__ == "__main__":
  absltest.main()
