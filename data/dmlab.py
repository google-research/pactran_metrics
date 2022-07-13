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
"""Implements Dmlab data class."""

from pactran_metrics import registry
from pactran_metrics.data import base
import tensorflow_datasets as tfds


@registry.register("data.dmlab", "class")
class DmlabData(base.ImageTfdsData):
  """Dmlab dataset.

      The Dmlab dataset contains frames observed by the agent acting in the
      DMLab environment, which are annotated by the distance between
      the agent and various objects present in the environment. The goal is to
      is to evaluate the ability of a visual model to reason about distances
      from the visual input in 3D environments. The Dmlab dataset consists of
      360x480 color images in 6 classes. The classes are
      {close, far, very far} x {positive reward, negative reward}
      respectively.
  """

  def __init__(self, data_dir=None):
    dataset_builder = tfds.builder("dmlab:2.0.1", data_dir=data_dir)

    tfds_splits = {
        "train": "train",
        "val": "validation",
        "trainval": "train+validation",
        "test": "test",
        "train800": "train[:800]",
        "val200": "validation[:200]",
        "train800val200": "train[:800]+validation[:200]",
    }

    # Example counts are retrieved from the tensorflow dataset info.
    train_count = dataset_builder.info.splits["train"].num_examples
    val_count = dataset_builder.info.splits["validation"].num_examples
    test_count = dataset_builder.info.splits["test"].num_examples

    # Creates a dict with example counts for each split.
    num_samples_splits = {
        "train": train_count,
        "val": val_count,
        "trainval": train_count + val_count,
        "test": test_count,
        "train800": 800,
        "val200": 200,
        "train800val200": 1000,
    }

    super(DmlabData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        base_preprocess_fn=base.make_get_and_cast_tensors_fn({
            "image": ("image", None),
            "label": ("label", None),
        }),
        num_classes=dataset_builder.info.features["label"].num_classes,
        image_key="image")
