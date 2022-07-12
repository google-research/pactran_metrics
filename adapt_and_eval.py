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
"""A script for running hub-module adaptation and evaluation."""

import os

from absl import app

from pactran_metrics import config
from pactran_metrics import loop

FLAGS = config.define_flags()


def get_data_params_from_flags(mode):

  return {
      "dataset_name": "data." + FLAGS.dataset,
      "dataset_train_split_name": FLAGS.dataset_train_split_name,
      "dataset_eval_split_name": FLAGS.dataset_eval_split_name,
      "shuffle_buffer_size": FLAGS.shuffle_buffer_size,
      "prefetch": FLAGS.prefetch,
      "train_examples": FLAGS.train_examples,
      "batch_size": FLAGS.batch_size,
      "batch_size_eval": FLAGS.batch_size_eval,
      "data_for_eval": mode == "adaptation",
      "data_dir": FLAGS.data_dir,
      "image_size": FLAGS.image_size,
      "input_range": [float(v) for v in FLAGS.input_range],
  }


def get_optimization_params_from_flags():
  return {
      "finetune_layer": FLAGS.finetune_layer,
      "initial_learning_rate": FLAGS.initial_learning_rate,
      "momentum": FLAGS.momentum,
      "lr_decay_factor": FLAGS.lr_decay_factor,
      "decay_steps": [int(x) for x in FLAGS.decay_steps.split(",")],
      "max_steps": FLAGS.max_steps,
      "warmup_steps": FLAGS.warmup_steps,
      "use_anil": FLAGS.use_anil,
  }


def main(argv):
  del argv
  os.environ["TFHUB_CACHE_DIR"] = FLAGS.tf_hub_dir
  gan_models = [
      "https://tfhub.dev/vtab/uncond-biggan/1",
      "https://tfhub.dev/vtab/cond-biggan/1",
      "https://tfhub.dev/vtab/wae-mmd/1",
      "https://tfhub.dev/vtab/vae/1",
      "https://tfhub.dev/vtab/wae-ukl/1",
      "https://tfhub.dev/vtab/wae-gan/1"]

  # The image size of GAN-based model is 128
  if FLAGS.hub_module in gan_models:
    FLAGS.image_size = 128
  if FLAGS.run_adaptation:
    loop.run_training_loop(
        hub_module=FLAGS.hub_module,
        hub_module_signature=FLAGS.hub_module_signature,
        work_dir=FLAGS.work_dir,
        save_checkpoints_steps=FLAGS.save_checkpoint_steps,
        optimization_params=get_optimization_params_from_flags(),
        data_params=get_data_params_from_flags("adaptation"))
  if FLAGS.run_evaluation:
    loop.run_evaluation_loop(
        hub_module=FLAGS.hub_module,
        hub_module_signature=FLAGS.hub_module_signature,
        work_dir=FLAGS.work_dir,
        save_checkpoints_steps=FLAGS.save_checkpoint_steps,
        optimization_params=get_optimization_params_from_flags(),
        data_params=get_data_params_from_flags("evaluation"))
  if FLAGS.run_prediction:
    loop.run_prediction_loop(
        hub_module=FLAGS.hub_module,
        hub_module_signature=FLAGS.hub_module_signature,
        work_dir=FLAGS.work_dir,
        save_checkpoints_steps=FLAGS.save_checkpoint_steps,
        optimization_params=get_optimization_params_from_flags(),
        save_format=FLAGS.save_format,
        data_params=get_data_params_from_flags("prediction"))


if __name__ == "__main__":
  app.run(main)
