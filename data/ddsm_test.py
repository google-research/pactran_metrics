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
"""Tests for ddsm.py."""

from pactran_metrics.data import data_testing_lib
from pactran_metrics.data import ddsm
import tensorflow.compat.v1 as tf


class DDSMTest(data_testing_lib.BaseTfdsDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(DDSMTest, self).setUp(
        data_wrapper=ddsm.DDSMData(config="tfds"),
        num_classes=5,
        expected_num_samples=dict(
            train=49780,
            val=5580,
            test=9770,
            trainval=49780+5580,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        })

if __name__ == "__main__":
  tf.test.main()
