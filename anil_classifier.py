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
"""The convex solver for the ANIL classifier."""

import os
import pickle

from absl import app
from absl import flags
import numpy as np
import scipy.optimize
import scipy.special
import tensorflow.compat.v1 as tf

flags.DEFINE_string("hub_module", None, "Hub module to evaluate.")
flags.DEFINE_string("dataset", None, "Dataset name.")
flags.DEFINE_string("feature_dir", None, "Directory that stored features.")
flags.DEFINE_string("work_dir", None, "Working directory for storing"
                                        "checkpoints, summaries, etc.")
flags.DEFINE_boolean("is_vqa", False, "Is VQA or not.")

FLAGS = flags.FLAGS


def one_hot(a, size=None):
  if size is None:
    size = a.max()+1
  b = np.zeros((a.size, size))  # size + 1
  b[np.arange(a.size), a] = 1.
  return b[:, :size]


def write_file(output_name, eval_accs, lda_factors):
  best_id = np.argmax(eval_accs)
  with tf.gfile.Open(output_name, "w") as f:
    f.write("Accuracy of ANIL: %f\n" % (eval_accs[best_id]))
    f.write("lda factor: %f\n" % (lda_factors[best_id]))


def run_classifier(features_np_all, label_np_all,
                   lda_factor,
                   features_np_eval, label_np_eval, is_vqa=False):
  """Compute the PAC_Gauss score with diagonal variance."""
  nclasses = label_np_all.max() + 1
  label_np_1hot = one_hot(label_np_all)  # [n, v]

  bs = features_np_all.shape[0]
  d = features_np_all.shape[-1]
  lda = lda_factor * bs

  def predict_ans(theta, features):
    theta = np.reshape(theta, [d + 1, nclasses])
    w = theta[:d, :]
    b = theta[d:, :]
    logits = np.matmul(features, w) + b
    predict = np.argmax(logits, axis=-1)
    return predict

  def compute_vqa_acc(label, predict):
    label_1hot = np.array([one_hot(t, nclasses) for t in label])
    targets = np.sum(label_1hot, axis=1)
    predictions = one_hot(predict, nclasses)
    vqa_acc = np.mean(
        np.minimum(np.sum(predictions * targets, axis=1) / 3.333, 1))
    return vqa_acc

  # optimizing log lik + log prior
  def pac_loss_fn(theta):
    theta = np.reshape(theta, [d + 1, nclasses])

    w = theta[:d, :]
    b = theta[d:, :]
    logits = np.matmul(features_np_all, w) + b

    log_qz = logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    xent = np.sum(np.sum(
        label_np_1hot * (np.log(label_np_1hot + 1e-10) - log_qz), axis=-1)) / bs
    loss = xent + 0.5 * np.sum(np.square(w)) / lda
    return loss

  # gradient of xent + l2
  def pac_grad_fn(theta):
    theta = np.reshape(theta, [d + 1, nclasses])

    w = theta[:d, :]
    b = theta[d:, :]
    logits = np.matmul(features_np_all, w) + b

    # grad_f = scipy.special.softmax(logits, axis=-1)  # [n, k]
    max_logits = logits.max(axis=-1, keepdims=True)
    grad_f = np.exp(logits - max_logits)
    grad_f /= np.sum(grad_f, axis=-1, keepdims=True)

    grad_f -= label_np_1hot
    grad_f /= bs
    grad_w = np.matmul(features_np_all.transpose(), grad_f)  # [d, k]
    grad_w += w / lda

    grad_b = np.sum(grad_f, axis=0, keepdims=True)  # [1, k]
    grad = np.ravel(np.concatenate([grad_w, grad_b], axis=0))
    return grad

  kernel_shape = [d, nclasses]
  theta = np.random.normal(size=kernel_shape) * 0.03
  theta_1d = np.ravel(np.concatenate(
      [theta, np.zeros([1, nclasses])], axis=0))

  theta_1d = scipy.optimize.minimize(
      pac_loss_fn, theta_1d, method="L-BFGS-B",
      jac=pac_grad_fn,
      options=dict(maxiter=50), tol=1e-6).x

  predict_train = predict_ans(theta_1d, features_np_all)
  train_acc = np.mean(label_np_all == predict_train)
  if is_vqa:
    predict_eval = predict_ans(theta_1d, features_np_eval)
    eval_acc = compute_vqa_acc(label_np_eval, predict_eval)
  else:
    predict_eval = predict_ans(theta_1d, features_np_eval)
    eval_acc = np.mean(label_np_eval == predict_eval)

  return train_acc, eval_acc


def main(argv):
  del argv
  np.random.seed(0)

  feature_dir = FLAGS.feature_dir

  model_name = FLAGS.hub_module.split("/")[-2]
  work_dir = os.path.join(FLAGS.work_dir, model_name)
  if not tf.gfile.Exists(work_dir):
    tf.gfile.MakeDirs(work_dir)
  modeldir = os.path.join(feature_dir, model_name)
  datadir = FLAGS.dataset
  dataset_dir = os.path.join(modeldir, datadir)

  train_folder = os.path.join(dataset_dir, "train")
  eval_folder = os.path.join(dataset_dir, "eval")

  features = []
  labels = []
  for pkl_file_train in tf.io.gfile.listdir(train_folder):
    tf.logging.info("load pkl_file_train: %s", pkl_file_train)
    pkl_file_train = os.path.join(train_folder, pkl_file_train)
    with tf.gfile.Open(pkl_file_train, "rb") as resultfile:
      features_train = pickle.load(resultfile)
    features += features_train["feautres"]
    labels += features_train["labels"]

  features_ev = []
  labels_ev = []
  for pkl_file_eval in tf.io.gfile.listdir(eval_folder):
    tf.logging.info("load pkl_file_eval: %s", pkl_file_eval)
    pkl_file_eval = os.path.join(eval_folder, pkl_file_eval)
    with tf.gfile.Open(pkl_file_eval, "rb") as resultfile:
      features_eval = pickle.load(resultfile)
    features_ev += features_eval["feautres"]
    labels_ev += features_eval["labels"]

  features_np_all = np.asarray(features)
  label_np_all = np.asarray(labels)  # [n]
  tf.logging.info("features_np_all shape: %s", features_np_all.shape)
  tf.logging.info("label_np_all shape: %s", label_np_all.shape)

  features_np_eval = np.asarray(features_ev)
  label_np_eval = np.asarray(labels_ev)  # [n, 10]

  eval_accs = []
  lda_factors = [100., 10., 1., 0.1, 0.01]
  for lda_factor in lda_factors:
    train_acc, eval_acc = run_classifier(
        features_np_all, label_np_all, lda_factor,
        features_np_eval, label_np_eval, is_vqa=FLAGS.is_vqa)
    tf.logging.info("train acc: %f (lda: %f)", train_acc, lda_factor)
    tf.logging.info("eval acc: %f (lda: %f)", eval_acc, lda_factor)
    eval_accs.append(eval_acc)

  eval_accs = np.asarray(eval_accs)

  # write files
  write_file(os.path.join(
      work_dir, "eval_accuracy.txt"), eval_accs, lda_factors)


if __name__ == "__main__":
  app.run(main)
