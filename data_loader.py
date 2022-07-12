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
"""Helper function for loading input data."""
import functools

from pactran_metrics import registered_modulers  # pylint: disable=unused-import
from pactran_metrics import registry
import tensorflow.compat.v1 as tf


def populate_dataset(data_params):
  if "dataset" in data_params:
    raise ValueError(
        "this method should not be called if dataset is already populated")
  data_cls = registry.lookup(data_params["dataset_name"])
  data_params["dataset"] = data_cls(data_dir=data_params["data_dir"])


def standardize_image(data,
                      size=224,
                      input_range=(0.0, 1.0)):
  """Normalizes images to input_range.

  Args:
    data: input dict with image and label
    size: optional, the height and width of the standarized image
    input_range: optional, the range of the standarized image should be
    (0,1) or (-1, 1)

  Returns:
    The dict include the standarized image
  """
  image = data["image"]

  image = tf.image.resize(image, [size, size])
  image = tf.cast(image, tf.float32) / 255.0
  image = image * (input_range[1] - input_range[0]) + input_range[0]

  data["image"] = image
  return data


def build_data_pipeline(data_params, mode):
  """Builds data input pipeline."""

  if mode not in ("train", "eval", "predict"):
    raise ValueError(
        "The input pipeline supports two modes: `train`, `eval` or `predict."
        "Provided mode is {}".format(mode))

  data = data_params["dataset"]
  # Estimator's API requires a named parameter "params".
  def input_fn(params):
    epochs = None
    drop_remainder = params.get("drop_remainder", True)
    split_name = (
        data_params.get("dataset_train_split_name")
        if mode == "train" else data_params.get("dataset_eval_split_name"))
    # do not repeat for predict mode
    if mode == "predict":
      epochs = 1
      drop_remainder = False
      split_name = data_params.get("dataset_eval_split_name")

    return data.get_tf_data(
        split_name=split_name,
        batch_size=params.get("batch_size"),
        preprocess_fn=functools.partial(
            standardize_image,
            size=data_params.get("image_size"),
            input_range=data_params.get("input_range"),
        ),
        epochs=epochs,
        drop_remainder=drop_remainder,
        for_eval=(mode == "eval"),
        shuffle_buffer_size=data_params.get("shuffle_buffer_size"),
        prefetch=data_params.get("prefetch"),
        num_train_examples=data_params.get("num_train_examples"))

  return input_fn
