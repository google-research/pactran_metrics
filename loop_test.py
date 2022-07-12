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
"""Tests for loop.py."""

import tempfile

from absl.testing import absltest
from pactran_metrics import loop
from pactran_metrics import test_utils
import tensorflow.compat.v1 as tf


class LoopTest(absltest.TestCase):
  """Creates a dummy hub module to test the training loop."""

  def test_run_training_loop(self):
    tmp_dir = tempfile.mkdtemp()
    loop.run_training_loop(
        hub_module=test_utils.create_dummy_hub_module(num_outputs=10),
        hub_module_signature=None,
        work_dir=tmp_dir,
        save_checkpoints_steps=10,
        optimization_params=test_utils.get_optimization_params(),
        data_params=test_utils.get_data_params())

    self.assertNotEmpty([f for f in tf.gfile.ListDirectory(tmp_dir)
                         if f.startswith("model.ckpt")])

  def test_run_prediction_loop(self):
    tmp_dir = tempfile.mkdtemp()
    loop.run_prediction_loop(
        hub_module=test_utils.create_dummy_hub_module(num_outputs=10),
        hub_module_signature=None,
        work_dir=tmp_dir,
        save_checkpoints_steps=10,
        optimization_params=test_utils.get_optimization_params(),
        data_params=test_utils.get_data_params())
    self.assertNotEmpty(
        [f for f in tf.gfile.ListDirectory(tmp_dir) if f.startswith("eval")])

if __name__ == "__main__":
  absltest.main()
