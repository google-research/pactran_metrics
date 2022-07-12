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
"""Config flags."""
from absl import flags


def define_flags():
  """Add training flags."""

  # task transferability flags
  flags.DEFINE_bool("run_adaptation", True, "Run adaptation.")
  flags.DEFINE_bool("run_evaluation", True, "Run evaluation.")
  flags.DEFINE_bool("linear_eval", False, "linear evaluation")
  flags.DEFINE_bool("run_prediction", False, "Run prediction.")
  flags.DEFINE_bool("use_anil", False,
                    "Freeze the layers until the penultimate.")

  # model checkpoint and data
  flags.DEFINE_string("tf_hub_dir",
                      "/tmp/",
                      "directory to store temp model"
                      "of tf hub module")
  flags.DEFINE_string("hub_module", None, "Hub module to evaluate.")
  flags.DEFINE_string("hub_module_signature", None,
                      "Name of the hub module signature.")
  flags.DEFINE_string("work_dir", None, "Working directory for storing"
                                        "checkpoints, summaries, etc.")
  flags.DEFINE_string("data_dir", None,
                      "A directory to download and store data.")
  flags.DEFINE_string("dataset", None, "Dataset name.")
  flags.DEFINE_enum("dataset_train_split_name", "trainval",
                    ["train", "val", "trainval", "test"],
                    "Dataset train split name.")
  flags.DEFINE_enum("dataset_eval_split_name", "test",
                    ["train", "val", "trainval", "test"],
                    "Dataset evaluation split name.")
  flags.DEFINE_integer("shuffle_buffer_size", 10000,
                       "A size of the shuffle buffer.")
  flags.DEFINE_integer("prefetch", 1000,
                       "How many batches to prefetch in the input pipeline.")
  flags.DEFINE_integer("train_examples", None,
                       "How many training examples to use. Defaults to all.")

  # training
  flags.DEFINE_integer("batch_size", None, "Batch size for training.")
  flags.DEFINE_integer(
      "batch_size_eval", None,
      "Batch size for evaluation: for the precise result should "
      "be a multiplier of the total size of the evaluation"
      "split, otherwise the reaminder is dropped.")
  flags.DEFINE_list("input_range", "0.0,1.0",
                    "Two comma-separated float values that represent "
                    "min and max value of the input range.")
  flags.DEFINE_string("finetune_layer", None, "Layer name for fine tunning.")
  flags.DEFINE_float("initial_learning_rate", None, "Initial learning rate.")
  flags.DEFINE_float("momentum", 0.9, "SGD momentum.")
  flags.DEFINE_float("lr_decay_factor", 0.1, "Learning rate decay factor.")
  flags.DEFINE_float("weight_decay", 0.0001, "weight_decay")
  flags.DEFINE_string("decay_steps", None, "Comma-separated list of steps at "
                      "which learning rate decay is performed.")
  flags.DEFINE_integer("max_steps", None, "Total number of SGD updates.")
  flags.DEFINE_integer("warmup_steps", 0,
                       "Number of step for warming up the leanring rate. It is"
                       "warmed up linearly: from 0 to the initial value.")
  flags.DEFINE_integer("save_checkpoint_steps", 500,
                       "Number of steps between consecutive checkpoints.")
  flags.DEFINE_integer("image_size", 224,
                       "Default is 224, GAN based should set to 128 ")

  flags.mark_flag_as_required("hub_module")
  flags.mark_flag_as_required("hub_module_signature")
  flags.mark_flag_as_required("finetune_layer")
  flags.mark_flag_as_required("work_dir")
  flags.mark_flag_as_required("dataset")
  flags.mark_flag_as_required("batch_size")
  flags.mark_flag_as_required("batch_size_eval")
  flags.mark_flag_as_required("initial_learning_rate")
  flags.mark_flag_as_required("decay_steps")
  flags.mark_flag_as_required("max_steps")

  return flags.FLAGS
