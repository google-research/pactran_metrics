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
"""Implements DDSM data class."""


from pactran_metrics import registry
from pactran_metrics.data import base
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

_IMAGE = "image"
_LABEL = "label"


@registry.register("data.ddsm", "class")
class DDSMData(base.ImageTfdsData):
  """Provides DDSM data.

  The CBIS-DDSM (Curated Breast Imaging Subset of DDSM) is an updated and
  standardized version of the Digital Database for Screening Mammography (DDSM).
  The DDSM is a database of 2,620 scanned film mammography studies. It contains
  normal, benign, and malignant cases with verified pathology information.

  The default config is made of patches extracted from the original mammograms,
  following the description from http://arxiv.org/abs/1708.09427, in order to
  frame the task to solve in a traditional image classification setting.
  """

  def __init__(self, config="tfds", data_dir=None):

    if config == "tfds":
      dataset_builder = tfds.builder(
          "curated_breast_imaging_ddsm/patches", data_dir=data_dir)
      dataset_builder.download_and_prepare()

      tfds_splits = {
          "train": "train",
          "val": "validation",
          "test": "test",
          "trainval": "train+validation",
      }
      # Creates a dict with example counts.
      num_samples_splits = {
          "test":
              dataset_builder.info.splits["test"].num_examples,
          "train":
              dataset_builder.info.splits["train"].num_examples,
          "val":
              dataset_builder.info.splits["validation"].num_examples,
          "trainval":
              dataset_builder.info.splits["train"].num_examples +
              dataset_builder.info.splits["validation"].num_examples,
      }
    else:
      raise ValueError("No supported config %r for DDSMData." % config)
    def preprocess_fn(tensors):
      images = tf.tile(tensors["image"], [1, 1, 3])
      label = tensors["label"]
      return dict(image=images, label=label)
    super(DDSMData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # Note: Export only image and label tensors with their original types.
        base_preprocess_fn=preprocess_fn,
        num_classes=dataset_builder.info.features[_LABEL].num_classes)
