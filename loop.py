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
"""Main adaptation and evaluation loops."""

import os
import pickle

from absl import flags
from pactran_metrics import data_loader
from pactran_metrics import model
import tensorflow.compat.v1 as tf
from tensorflow_estimator import estimator as tf_estimator

FLAGS = flags.FLAGS


def setup_estimator(
    hub_module,
    hub_module_signature,
    work_dir,
    save_checkpoints_steps,
    optimization_params,
    data_params):
  """Produces Estimator object for a given configuration."""

  # Merge all parameters into single dictionary (for tf.estimator API).
  params_list = [
      optimization_params, data_params, {
          "hub_module": hub_module,
          "hub_module_signature": hub_module_signature,
          "num_classes": data_params["dataset"].get_num_classes(),
      }
  ]
  params = {}
  for d in params_list:
    params.update(d)
  # Creates a estimator from the configutation of an adaptation/evaluation loop.

  config = tf_estimator.RunConfig(
      model_dir=work_dir,
      keep_checkpoint_max=None,
      save_checkpoints_steps=save_checkpoints_steps)

  estimator = tf_estimator.Estimator(
      model_fn=model.model_fn,
      model_dir=work_dir,
      params=params,
      config=config)

  return estimator


def run_training_loop(hub_module,
                      hub_module_signature,
                      work_dir,
                      save_checkpoints_steps,
                      optimization_params,
                      data_params):
  """Runs training loop."""
  data_loader.populate_dataset(data_params)
  estimator = setup_estimator(hub_module,
                              hub_module_signature,
                              work_dir,
                              save_checkpoints_steps,
                              optimization_params,
                              data_params)
  input_fn = data_loader.build_data_pipeline(data_params, mode="train")

  # TPUs require the max number of steps to be specified explicitly.
  estimator.train(input_fn, max_steps=optimization_params["max_steps"])


def run_evaluation_loop(hub_module,
                        hub_module_signature,
                        work_dir,
                        save_checkpoints_steps,
                        optimization_params,
                        data_params):
  """Runs evaluation loop."""
  data_loader.populate_dataset(data_params)
  estimator = setup_estimator(hub_module,
                              hub_module_signature,
                              work_dir,
                              save_checkpoints_steps,
                              optimization_params,
                              data_params)
  input_fn = data_loader.build_data_pipeline(data_params, mode="eval")

  with tf.gfile.Open(os.path.join(work_dir, "result_file.txt"), "w") as f:
    all_checkpoints = set([".".join(f.split(".")[:-1])
                           for f in tf.gfile.ListDirectory(work_dir)
                           if f.startswith("model.ckpt")])
    # Sort checkpoints by the global step.
    all_checkpoints = sorted(all_checkpoints,
                             key=lambda x: int(x.split("-")[-1]))
    # For efficiency reasons we evluate only the last checkpoint
    for ckpt in all_checkpoints[-1:]:
      ckpt = os.path.join(work_dir, ckpt)
      res = estimator.evaluate(
          input_fn,
          steps=(data_params["dataset"].get_num_samples(
              data_params["dataset_eval_split_name"]) //
                 data_params["batch_size_eval"]),
          checkpoint_path=ckpt)
      f.write("Accuracy at step {}: {}\n".format(res["global_step"],
                                                 res["accuracy"]))


def run_prediction_loop(hub_module, hub_module_signature, work_dir,
                        save_checkpoints_steps, optimization_params,
                        data_params):
  """Runs Prediction loop."""
  data_loader.populate_dataset(data_params)
  estimator = setup_estimator(hub_module,
                              hub_module_signature,
                              work_dir,
                              save_checkpoints_steps,
                              optimization_params,
                              data_params)
  input_fn = data_loader.build_data_pipeline(data_params, mode="predict")

  pred_generator = estimator.predict(input_fn)

  mode = "eval"  # train or eval

  total_image = data_params["dataset"].get_num_samples(
      data_params["dataset_" + mode + "_split_name"])
  work_dir = os.path.join(work_dir, mode)
  if not tf.gfile.Exists(work_dir):
    tf.gfile.MakeDirs(work_dir)

  image_nums = 0
  correct = 0
  feature_list = []
  label_list = []
  predict_list = []
  prob_list = []
  logit_list = []

  pkl_file_prefix = "feature_label_logits_" + mode
  tf.logging.info("save feature to pkl")

  for pred_dict in pred_generator:
    feature_list.append(pred_dict["features"])
    label_list.append(pred_dict["labels"])
    predict_list.append(pred_dict["classes"])
    prob_list.append(pred_dict["probabilities"])
    logit_list.append(pred_dict["logits"])
    if pred_dict["labels"] == pred_dict["classes"]:
      correct += 1
    # print('predict: ', pred_dict["classes"])
    # print('label: ', pred_dict["labels"])
    # "probabilities",
    image_nums += 1
    if image_nums >= total_image:
      break
    if image_nums % (20000) == 0:
      tf.logging.info("image_nums %s", image_nums)
      pkl_file = pkl_file_prefix + str(int(image_nums/(20000))) + ".pkl"
      tf.logging.info("save feature to pkl: %s", pkl_file)
      with tf.gfile.Open(os.path.join(work_dir, pkl_file),
                         "wb") as resultfile:
        pickle.dump(
            {
                "feautres": feature_list,
                "labels": label_list,
                "predicts": predict_list,
                "logits": logit_list,
                "probs": prob_list,
            },
            resultfile,
            protocol=pickle.HIGHEST_PROTOCOL)
      feature_list = []
      label_list = []
      predict_list = []
      prob_list = []
      logit_list = []
  pkl_file = pkl_file_prefix + str(int(image_nums/(20000) + 1)) + ".pkl"
  tf.logging.info("save feature to pkl: %s", pkl_file)

  with tf.gfile.Open(os.path.join(work_dir, pkl_file), "wb") as resultfile:
    pickle.dump(
        {
            "feautres": feature_list,
            "labels": label_list,
            "predicts": predict_list,
            "logits": logit_list,
            "probs": prob_list,
        },
        resultfile,
        protocol=pickle.HIGHEST_PROTOCOL)

  accuracy = correct / image_nums
  tf.logging.info("image_nums %s", image_nums)
  tf.logging.info("accuracy %s", accuracy)
  tf.logging.info("total image %s", total_image)
