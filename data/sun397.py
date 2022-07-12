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
"""Implements Sun397 data class."""


from pactran_metrics import registry
from garcon.pactran_metrics.data import base
import tensorflow_datasets as tfds

_IMAGE = "image"
_LABEL = "label"


@registry.register("data.sun397", "class")
class Sun397Data(base.ImageTfdsData):
  """Provides Sun397Data data."""

  def __init__(self, config="tfds", data_dir=None):

    if config == "tfds":
      dataset_builder = tfds.builder("sun397/tfds:4.*.*", data_dir=data_dir)
      dataset_builder.download_and_prepare()

      tfds_splits = {
          "train": "train",
          "val": "validation",
          "test": "test",
          "trainval": "train+validation",
          "train800": "train[:800]",
          "val200": "validation[:200]",
          "train800val200": "train[:800]+validation[:200]",
      }
      # Creates a dict with example counts.
      num_samples_splits = {
          "test": dataset_builder.info.splits["test"].num_examples,
          "train": dataset_builder.info.splits["train"].num_examples,
          "val": dataset_builder.info.splits["validation"].num_examples,
          "train800": 800,
          "val200": 200,
          "train800val200": 1000,
      }
      num_samples_splits["trainval"] = (
          num_samples_splits["train"] + num_samples_splits["val"])
    else:
      # TODO(yandongli): Add support for "official" splits.
      raise ValueError("No supported config %r for Sun397Data." % config)

    super(Sun397Data, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # Note: Export only image and label tensors with their original types.
        base_preprocess_fn=base.make_get_tensors_fn([_IMAGE, _LABEL]),
        num_classes=dataset_builder.info.features[_LABEL].num_classes)
