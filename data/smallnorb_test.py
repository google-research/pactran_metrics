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
"""Tests for smallnorb.py."""

from pactran_metrics.data import data_testing_lib
from pactran_metrics.data import smallnorb
import tensorflow.compat.v1 as tf


class SmallnorbTest(data_testing_lib.BaseTfdsDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(SmallnorbTest, self).setUp(
        data_wrapper=smallnorb.SmallnorbData(config="tfds"),
        num_classes=18,
        expected_num_samples=dict(
            train=24300,
            val=24300,
            test=24300,
            trainval=24300,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label_azimuth": (),
        },
        default_label_key="label_azimuth")

if __name__ == "__main__":
  tf.test.main()
