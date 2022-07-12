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
"""Implements oxford_iiit_pet data class."""

from pactran_metrics import registry
from garcon.pactran_metrics.data import base
import tensorflow_datasets as tfds

_IMAGE = "image"
_LABEL = "label"


@registry.register("data.oxford_iiit_pet", "class")
class OxfordiiitpetData(base.ImageTfdsData):
  """oxford_iiit_pet dataset.

  The Oxford-IIIT pet dataset is a 37 category pet image dataset with roughly
  200 images for each class. The images have large variations in scale, pose and
  lighting. All images have an associated ground truth annotation of breed.
  """

  def __init__(self, config="tfds", data_dir=None):
    if config == "tfds":
      dataset_builder = tfds.builder("oxford_iiit_pet:3.2.0", data_dir=data_dir)
      dataset_builder.download_and_prepare()

      tfds_splits = {
          "train": "train",
          "val": "test",
          "test": "test",
          "trainval": "train",
      }
      # Creates a dict with example counts.
      num_samples_splits = {
          "test":
              dataset_builder.info.splits["test"].num_examples,
          "train":
              dataset_builder.info.splits["train"].num_examples,
          "val":
              dataset_builder.info.splits["test"].num_examples,
          "trainval":
              dataset_builder.info.splits["train"].num_examples,
      }
    else:
      raise ValueError("No supported config %r for OxfordiiitpetData." % config)

    super(OxfordiiitpetData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        base_preprocess_fn=base.make_get_and_cast_tensors_fn({
            "image": (_IMAGE, None),
            "label": (_LABEL, None),
        }),
        num_classes=dataset_builder.info.features[_LABEL].num_classes,
        image_key=_IMAGE)
