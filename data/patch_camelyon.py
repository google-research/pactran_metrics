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
"""Implements PatchCamelyon data class."""

from pactran_metrics import registry
from garcon.pactran_metrics.data import base
import tensorflow_datasets as tfds


_IMAGE = "image"
_LABEL = "label"


@registry.register("data.patch_camelyon", "class")
class PatchCamelyonData(base.ImageTfdsData):
  """Provides PatchCamelyon data."""

  def __init__(self, data_dir=None):

    dataset_builder = tfds.builder("patch_camelyon:2.*.*", data_dir=data_dir)
    dataset_builder.download_and_prepare()

    # Defines dataset specific train/val/trainval/test splits.
    tfds_splits = {
        "test": "test",
        "train": "train",
        "val": "validation",
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
    super(PatchCamelyonData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # Note: Export only image and label tensors with their original types.
        base_preprocess_fn=base.make_get_tensors_fn([_IMAGE, _LABEL]),
        num_classes=dataset_builder.info.features[_LABEL].num_classes)
