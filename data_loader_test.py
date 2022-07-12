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
"""Tests for data_loader."""


from absl.testing import absltest
from pactran_metrics import data_loader
from pactran_metrics import test_utils
import tensorflow.compat.v1 as tf


class DataLoaderTest(absltest.TestCase):

  def test_build_data_pipeline(self):
    data_params = test_utils.get_data_params()
    data_loader.populate_dataset(data_params)
    input_fn = data_loader.build_data_pipeline(
        data_params, mode="eval")
    data = tf.compat.v1.data.make_one_shot_iterator(
        input_fn({"batch_size": 32})).get_next()
    self.assertIsInstance(data["image"], tf.Tensor)
    self.assertIsInstance(data["label"], tf.Tensor)


if __name__ == "__main__":
  absltest.main()
