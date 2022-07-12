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
"""Implements SmallnorbData data class."""


from pactran_metrics import registry
from garcon.pactran_metrics.data import base
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

_IMAGE = "image"
_LABEL = "label_azimuth"


@registry.register("data.smallnorb", "class")
class SmallnorbData(base.ImageTfdsData):
  """Provides Smallnorb data.

  This database is intended for experiments in 3D object recognition from shape.
  It contains images of 50 toys belonging to 5 generic categories: four-legged
  animals, human figures, airplanes, trucks, and cars. The objects were imaged
  by two cameras under 6 lighting conditions, 9 elevations (30 to 70 degrees
  every 5 degrees), and 18 azimuths (0 to 340 every 20 degrees).

  The training set is composed of 5 instances of each category (instances 4, 6,
  7, 8 and 9), and the test set of the remaining 5 instances (instances 0, 1, 2,
  3, and 5).
  """

  def __init__(self, config="tfds", data_dir=None):
    if config == "tfds":
      dataset_builder = tfds.builder("smallnorb:2.0.0", data_dir=data_dir)
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
      raise ValueError("No supported config %r for SmallnorbData." % config)
    def preprocess_fn(tensors):
      images = tf.tile(tensors[_IMAGE], [1, 1, 3])
      label = tensors[_LABEL]
      return dict(image=images, label=label)
    super(SmallnorbData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # Note: Export only image and label tensors with their original types.
        base_preprocess_fn=preprocess_fn,
        num_classes=dataset_builder.info.features[_LABEL].num_classes,
        default_label_key=_LABEL)
