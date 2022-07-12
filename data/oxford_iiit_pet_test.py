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
"""Tests for plant_leaves.py."""

from garcon.pactran_metrics.data import data_testing_lib
from pactran_metrics.data import oxford_iiit_pet
import tensorflow.compat.v1 as tf


# For each of the 5 different data set configuration we need to run all the
# standardized tests in data_testing_lib.BaseDataTest.
class OxfordiiitpetTest(data_testing_lib.BaseTfdsDataTest):
  """See base classes for usage and test descriptions."""

  def setUp(self):
    super(OxfordiiitpetTest, self).setUp(
        data_wrapper=oxford_iiit_pet.OxfordiiitpetData(),
        num_classes=37,
        expected_num_samples=dict(
            train=3680,
            val=3669,
            trainval=3680,
            test=3669,
        ),
        required_tensors_shapes={
            "image": (64, 64, 3),
            "label": (),
        },)

if __name__ == "__main__":
  tf.test.main()
