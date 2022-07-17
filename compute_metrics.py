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
"""Compute different transferability metrics."""

import os
import pickle
import random
import time

from absl import app
from absl import flags
import numpy as np
import scipy.optimize
import scipy.special
import sklearn.decomposition
import sklearn.mixture
import sklearn.neighbors
import sklearn.svm
import tensorflow.compat.v1 as tf


flags.DEFINE_string("hub_module", None, "Hub module to evaluate.")
flags.DEFINE_string("dataset", None, "Dataset name.")
flags.DEFINE_string("feature_dir", None, "Directory that stored features.")
flags.DEFINE_string("work_dir", None, "Working directory for storing"
                                        "checkpoints, summaries, etc.")
flags.DEFINE_integer("num_examples", -1, "Num examples per class.")
flags.DEFINE_integer("num_runs", 5, "Num runs for metrics eval.")
flags.DEFINE_integer("num_classes", -1, "Num classes to evaluate.")

FLAGS = flags.FLAGS


def one_hot(a):
  b = np.zeros((a.size, a.max()+1))
  b[np.arange(a.size), a] = 1.
  return b


def write_file(output_name, kvs):
  with tf.gfile.Open(output_name, "w") as f:
    for kv in kvs:
      f.write("%s: %f\n" % (kv[0], kv[1]))


def gmm_estimator(features_np_all, label_np_all):
  """Estimate the GMM posterior assignment."""
  pca_model = sklearn.decomposition.PCA(n_components=0.8)
  pca_model.fit(features_np_all)
  features_lowdim_train = pca_model.transform(features_np_all)

  num_examples = label_np_all.shape[0]
  y_classes = max([min([label_np_all.max() + 1, int(num_examples * 0.2)]),
                   int(num_examples * 0.1)])
  clf = sklearn.mixture.GaussianMixture(n_components=y_classes)
  clf.fit(features_lowdim_train)
  prob_np_all_gmm = clf.predict_proba(features_lowdim_train)
  return prob_np_all_gmm, features_lowdim_train


def calculate_leep(prob_np_all, label_np_all):
  """Calculating the LEEP score."""

  starttime = time.time()
  label_np_all = one_hot(label_np_all)

  # compute p(y|z)
  pz = np.expand_dims(np.mean(prob_np_all, axis=0), axis=-1) + 1e-10  # [v, 1]
  pzy = np.mean(
      np.einsum("BF,BH->BFH", prob_np_all, label_np_all), axis=0)  # [v, v]
  pycz = pzy / pz  # p(y|z) [v, v]

  # compute p(y) = sum_z p(y|z)p(z)
  eep = np.matmul(prob_np_all, pycz) + 1e-10

  # leep = KL(label | eep)
  leep = np.mean(np.sum(
      label_np_all * (np.log(label_np_all + 1e-10) - np.log(eep)), axis=1))
  endtime = time.time()

  return [("leep", leep),
          ("time", endtime - starttime)]


def calculate_nleep(features_np_all, label_np_all):
  """Calculate LEEP with GMM classifier."""

  starttime = time.time()
  prob_np_all_gmm, _ = gmm_estimator(features_np_all, label_np_all)
  nleep_results = calculate_leep(prob_np_all_gmm, label_np_all)
  nleep = nleep_results[0][1]
  endtime = time.time()
  return [("nleep", nleep),
          ("time", endtime - starttime)]


def calculate_hscore(features_np_all, label_np_all):
  """Calculate hscore."""
  starttime = time.time()
  label_np_all = one_hot(label_np_all)
  mean_feature = np.mean(features_np_all, axis=0, keepdims=True)
  features_np_all -= mean_feature  # [n, k]
  d = features_np_all.shape[1]
  covf = np.matmul(
      features_np_all.transpose(), features_np_all) + np.eye(d) * 1e-6

  yf = np.matmul(label_np_all.transpose(), features_np_all)  # [v, k]
  sumy = np.sum(label_np_all, axis=0) + 1e-10  # [v]
  fcy = yf / np.expand_dims(sumy, axis=-1)  # [v, k]
  covfcy = np.matmul(yf.transpose(), fcy)

  hscore = np.trace(np.matmul(np.linalg.pinv(covf), covfcy))
  endtime = time.time()
  return [("nhscore", -hscore),
          ("time", endtime - starttime)]


def calculate_logme(features_np_all, label_np_all, num_iters=20):
  """Compute the LogME."""
  starttime = time.time()
  nclasses = label_np_all.max()+1
  label_np_all = one_hot(label_np_all)  # [n, v]

  n = features_np_all.shape[0]
  d = features_np_all.shape[1]
  ftf = np.matmul(features_np_all.transpose(), features_np_all)

  if num_iters == 0:
    alpha = 1.
    beta = 1.
    a = alpha * np.eye(d) + beta * ftf
    _, sigma_a, _ = np.linalg.svd(a, hermitian=True)
    log_deta = np.sum(np.log(sigma_a))
    m = beta * np.matmul(np.linalg.pinv(a),
                         np.matmul(features_np_all.transpose(), label_np_all))
    logme = -0.5 * (
        beta * np.sum(np.square(
            np.matmul(features_np_all, m) - label_np_all))
        + alpha * (np.sum(np.square(m)))
        + log_deta * nclasses) / n
  else:
    v, sigma, _ = np.linalg.svd(ftf, hermitian=True)  # [d, d]
    alpha = [1.] * nclasses
    beta = [1.] * nclasses
    logme = 0.
    for k in range(nclasses):
      for _ in range(num_iters):
        gamma = np.minimum(
            np.sum(beta[k] * sigma / (alpha[k] + beta[k] * sigma + 1e-10)),
            float(n))
        cov_inv = np.diag(1. /(alpha[k] + beta[k] * sigma + 1e-10))
        # a = alpha[k] * np.eye(d) + beta[k] * ftf
        sigma_a = sigma * beta[k] + alpha[k]
        log_deta = np.sum(np.log(sigma_a))
        m = beta[k] * np.matmul(v, np.matmul(cov_inv, np.matmul(
            v.transpose(), np.matmul(features_np_all.transpose(),
                                     label_np_all[:, k]))))
        alpha[k] = gamma / (np.sum(np.square(m)) + 1e-10)
        beta[k] = (float(n) - gamma) / (np.sum(np.square(
            np.matmul(features_np_all, m) - label_np_all[:, k])) + 1e-10)

      logme += 0.5 * (np.log(beta[k] + 1e-10) +
                      float(d) / float(n) * np.log(alpha[k] + 1e-10) -
                      1. - log_deta / float(n))
  logme /= nclasses
  endtime = time.time()
  return [("logme", logme),
          ("time", endtime - starttime)]


def calculate_nce(prob_np_all, label_np_all):
  """Compute the NCE estimator."""
  starttime = time.time()
  label_np_all = one_hot(label_np_all)  # [n, v]

  # compute p(z) and p(z, y)
  pz = np.expand_dims(np.mean(prob_np_all, axis=0), axis=-1) + 1e-10  # [v, 1]
  pzy = np.mean(
      np.einsum("BF,BH->BFH", prob_np_all, label_np_all),
      axis=0)  # [v, v]
  pycz = pzy / pz  # p(y|z) [v, v]

  nce = np.sum(pzy * np.log(pycz + 1e-10))
  endtime = time.time()
  return [("nce", nce),
          ("time", endtime - starttime)]


def calculate_pac_dir(prob_np_all, label_np_all, alpha):
  """Compute the PACTran-Dirichlet estimator."""
  starttime = time.time()
  label_np_all = one_hot(label_np_all)  # [n, v]
  soft_targets_sum = np.sum(label_np_all, axis=0)  # [v]
  soft_targets_sum = np.expand_dims(soft_targets_sum, axis=1)  # [v, 1]
  a0 = alpha * soft_targets_sum / np.sum(soft_targets_sum) + 1e-10

  # initialize
  qz = prob_np_all  # [n, d]
  log_s = np.log(prob_np_all + 1e-10)  # [n, d]

  for _ in range(10):
    aw = a0 + np.sum(np.einsum("BY,BZ->BYZ", label_np_all, qz), axis=0)
    logits_qz = (log_s +
                 np.matmul(label_np_all, scipy.special.digamma(aw)) -
                 np.reshape(scipy.special.digamma(np.sum(aw, axis=0)), [1, -1]))
    log_qz = logits_qz - scipy.special.logsumexp(
        logits_qz, axis=-1, keepdims=True)
    qz = np.exp(log_qz)

  log_c0 = scipy.special.loggamma(np.sum(a0)) - np.sum(
      scipy.special.loggamma(a0))
  log_c = scipy.special.loggamma(np.sum(aw, axis=0)) - np.sum(
      scipy.special.loggamma(aw), axis=0)

  pac_dir = np.sum(
      log_c0 - log_c - np.sum(qz * (log_qz - log_s), axis=0))
  pac_dir = -pac_dir / label_np_all.size
  endtime = time.time()
  return [("pac_dir_%.1f" % alpha, pac_dir),
          ("time", endtime - starttime)]


def calculate_pac_gamma(prob_np_all, label_np_all, alpha):
  """Compute the PAC-Gamma estimator."""
  starttime = time.time()
  label_np_all = one_hot(label_np_all)  # [n, v]
  soft_targets_sum = np.sum(label_np_all, axis=0)  # [v]
  soft_targets_sum = np.expand_dims(soft_targets_sum, axis=1)  # [v, 1]

  a0 = alpha * soft_targets_sum / np.sum(soft_targets_sum) + 1e-10
  beta = 1.

  # initialize
  qz = prob_np_all  # [n, d]
  s = prob_np_all  # [n, d]
  log_s = np.log(prob_np_all + 1e-10)  # [n, d]
  aw = a0
  bw = beta
  lw = np.sum(s, axis=-1, keepdims=True) * np.sum(aw / bw)  # [n, 1]

  for _ in range(10):
    aw = a0 + np.sum(np.einsum("BY,BZ->BYZ", label_np_all, qz),
                     axis=0)  # [v, d]
    lw = np.matmul(
        s, np.expand_dims(np.sum(aw / bw, axis=0), axis=1))  # [n, 1]
    logits_qz = (
        log_s + np.matmul(label_np_all, scipy.special.digamma(aw) - np.log(bw)))
    log_qz = logits_qz - scipy.special.logsumexp(
        logits_qz, axis=-1, keepdims=True)
    qz = np.exp(log_qz)  # [n, a, d]

  pac_gamma = (
      np.sum(scipy.special.loggamma(a0) - scipy.special.loggamma(aw) +
             aw * np.log(bw) - a0 * np.log(beta)) +
      np.sum(np.sum(qz * (log_qz - log_s), axis=-1) +
             np.log(np.squeeze(lw, axis=-1)) - 1.))
  pac_gamma /= label_np_all.size
  pac_gamma += 1.
  endtime = time.time()
  return [("pac_gamma_%.1f" % alpha, pac_gamma),
          ("time", endtime - starttime)]


def calculate_pac_gauss(features_np_all, label_np_all,
                        lda_factor):
  """Compute the PAC_Gauss score with diagonal variance."""
  starttime = time.time()
  nclasses = label_np_all.max()+1
  label_np_all = one_hot(label_np_all)  # [n, v]
  
  mean_feature = np.mean(features_np_all, axis=0, keepdims=True)
  features_np_all -= mean_feature  # [n,k]

  bs = features_np_all.shape[0]
  kd = features_np_all.shape[-1] * nclasses
  ldas2 = lda_factor * bs  # * features_np_all.shape[-1]
  dinv = 1. / float(features_np_all.shape[-1])

  # optimizing log lik + log prior
  def pac_loss_fn(theta):
    theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

    w = theta[:features_np_all.shape[-1], :]
    b = theta[features_np_all.shape[-1]:, :]
    logits = np.matmul(features_np_all, w) + b

    log_qz = logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    xent = np.sum(np.sum(
        label_np_all * (np.log(label_np_all + 1e-10) - log_qz), axis=-1)) / bs
    loss = xent + 0.5 * np.sum(np.square(w)) / ldas2
    return loss

  # gradient of xent + l2
  def pac_grad_fn(theta):
    theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

    w = theta[:features_np_all.shape[-1], :]
    b = theta[features_np_all.shape[-1]:, :]
    logits = np.matmul(features_np_all, w) + b

    grad_f = scipy.special.softmax(logits, axis=-1)  # [n, k]
    grad_f -= label_np_all
    grad_f /= bs
    grad_w = np.matmul(features_np_all.transpose(), grad_f)  # [d, k]
    grad_w += w / ldas2

    grad_b = np.sum(grad_f, axis=0, keepdims=True)  # [1, k]
    grad = np.ravel(np.concatenate([grad_w, grad_b], axis=0))
    return grad

  # 2nd gradient of theta (elementwise)
  def pac_grad2(theta):
    theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

    w = theta[:features_np_all.shape[-1], :]
    b = theta[features_np_all.shape[-1]:, :]
    logits = np.matmul(features_np_all, w) + b

    prob_logits = scipy.special.softmax(logits, axis=-1)  # [n, k]
    grad2_f = prob_logits - np.square(prob_logits)  # [n, k]
    xx = np.square(features_np_all)  # [n, d]

    grad2_w = np.matmul(xx.transpose(), grad2_f)  # [d, k]
    grad2_w += 1. / ldas2
    grad2_b = np.sum(grad2_f, axis=0, keepdims=True)  # [1, k]
    grad2 = np.ravel(np.concatenate([grad2_w, grad2_b], axis=0))
    return grad2

  kernel_shape = [features_np_all.shape[-1], nclasses]
  theta = np.random.normal(size=kernel_shape) * 0.03
  theta_1d = np.ravel(np.concatenate(
      [theta, np.zeros([1, nclasses])], axis=0))

  theta_1d = scipy.optimize.minimize(
      pac_loss_fn, theta_1d, method="L-BFGS-B",
      jac=pac_grad_fn,
      options=dict(maxiter=100), tol=1e-6).x

  pac_opt = pac_loss_fn(theta_1d)
  endtime_opt = time.time()

  h = pac_grad2(theta_1d)
  sigma2_inv = np.sum(h) * ldas2  / kd + 1e-10
  endtime = time.time()

  if lda_factor == 10.:
    s2s = [1000., 100.]
  elif lda_factor == 1.:
    s2s = [100., 10.]
  elif lda_factor == 0.1:
    s2s = [10., 1.]
    
  returnv = []
  for s2_factor in s2s:
    s2 = s2_factor * dinv
    pac_gauss = pac_opt + 0.5 * kd / ldas2 * s2 * np.log(
        sigma2_inv)
    
    # the first item is the pac_gauss metric
    # the second item is the linear metric (without trH)
    returnv += [("pac_gauss_%.1f" % lda_factor, pac_gauss),
                ("time", endtime - starttime),
                ("pac_opt_%.1f" % lda_factor, pac_opt),
                ("time", endtime_opt - starttime)]
  return returnv, theta_1d


def calculate_linear_valerr(theta_1d, features_np_val, label_np_val):
  """Compute the linear classifier validation error."""
  starttime = time.time()
  d = features_np_val.shape[-1]

  theta = np.reshape(theta_1d, [d + 1, -1])
  w = theta[:d, :]
  b = theta[d:, :]
  logits = np.matmul(features_np_val, w) + b
  predict = np.argmax(logits, axis=-1)
  eval_acc = np.mean(label_np_val == predict)
  endtime = time.time()
  return [("linear_valerr:", 1. - eval_acc),
          ("time", endtime - starttime)]


def return_nclass_data(labels, features, probs, nclasses):
  """Only return the example ids from the top-n classes."""
  nclass_labels = []
  nclass_features = []
  nclass_probs = []
  for i in range(len(labels)):
    if labels[i] < nclasses:
      nclass_labels.append(labels[i])
      nclass_features.append(features[i])
      nclass_probs.append(probs[i])
  return nclass_labels, nclass_features, nclass_probs


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

  probs = []
  features = []
  labels = []

  for pkl_file_train in tf.io.gfile.listdir(train_folder):
    tf.logging.info("load pkl_file_train: %s", pkl_file_train)
    pkl_file_train = os.path.join(train_folder, pkl_file_train)
    with tf.gfile.Open(pkl_file_train, "rb") as resultfile:
      features_train = pickle.load(resultfile)
    if "probs" in features_train:
      probs += features_train["probs"]
    features += features_train["feautres"]
    labels += features_train["labels"]

  if FLAGS.num_classes <= 0 or FLAGS.num_classes > np.asarray(labels).max():
    k = np.asarray(labels).max() + 1
  else:
    k = FLAGS.num_classes
    labels, features, probs = return_nclass_data(
        labels, features, probs, k)
    tf.logging.info("Number of total examples: %d", len(labels))

  leep = []
  pac_dir = []
  pac_gam = []
  pac_ndir = []
  pac_ngam = []
  nleep = []
  hscore = []
  logme = []
  pac_gauss = []
  pac_gauss_half = []
  valerr_half = []

  for run in range(FLAGS.num_runs):
    if len(probs) < FLAGS.num_examples * k:
      num_examples = len(probs)
    else:
      num_examples = FLAGS.num_examples * k

    num_examples = max(min([num_examples, 8000]), 20)

    example_ids = list(range(len(probs)))
    random.seed(run)
    random.shuffle(example_ids)
    example_ids = np.array(example_ids[:num_examples])

    features_np_all = np.asarray(features)[example_ids]
    prob_np_all = np.asarray(probs)[example_ids]
    label_np_all = np.asarray(labels)[example_ids]
    tf.logging.info("features_np_all shape: %s", features_np_all.shape)
    tf.logging.info("prob_np_all shape: %s", prob_np_all.shape)
    tf.logging.info("label_np_all shape: %s", label_np_all.shape)

    prob_np_all_gmm, features_np_all_pca = gmm_estimator(
        features_np_all, label_np_all)
    tf.logging.info("prob_np_all_gmm shape: %s", prob_np_all_gmm.shape)
    tf.logging.info("features_np_all_pca shape: %s",
                    features_np_all_pca.shape)

    # depends on probs
    leep += calculate_leep(prob_np_all, label_np_all)
    pac_dir += calculate_pac_dir(
        prob_np_all, label_np_all, alpha=1.)
    pac_gam += calculate_pac_gamma(
        prob_np_all, label_np_all, alpha=1.)

    # depends on gmms
    pac_ndir += calculate_pac_dir(
        prob_np_all_gmm, label_np_all, alpha=1.)
    pac_ngam += calculate_pac_gamma(
        prob_np_all_gmm, label_np_all, alpha=1.)

    # depends on features
    nleep += calculate_nleep(features_np_all, label_np_all)
    hscore += calculate_hscore(features_np_all, label_np_all)
    logme += calculate_logme(
        features_np_all, label_np_all, num_iters=0)

    for lda_factor in [10., 1.0, 0.1]:
      pg, _ = calculate_pac_gauss(
          features_np_all, label_np_all, lda_factor=lda_factor)
      pac_gauss += pg

      # mini train-val split
      pg, theta_1d = calculate_pac_gauss(
          features_np_all[:num_examples // 2], label_np_all[:num_examples // 2],
          lda_factor=lda_factor)
      pac_gauss_half += pg
      valerr_half += calculate_linear_valerr(
          theta_1d, features_np_all[num_examples // 2:],
          label_np_all[num_examples // 2:])

  # write files
  write_file(os.path.join(
      work_dir, "n%d_leep.txt" % FLAGS.num_examples), leep)
  write_file(os.path.join(
      work_dir, "n%d_pac_dir.txt" % FLAGS.num_examples), pac_dir)
  write_file(os.path.join(
      work_dir, "n%d_pac_gam.txt" % FLAGS.num_examples), pac_gam)
  write_file(os.path.join(
      work_dir, "n%d_pac_ndir.txt" % FLAGS.num_examples), pac_ndir)
  write_file(os.path.join(
      work_dir, "n%d_pac_ngam.txt" % FLAGS.num_examples), pac_ngam)
  write_file(os.path.join(
      work_dir, "n%d_nleep.txt" % FLAGS.num_examples), nleep)
  write_file(os.path.join(
      work_dir, "n%d_hscore.txt" % FLAGS.num_examples), hscore)
  write_file(os.path.join(
      work_dir, "n%d_logme.txt" % FLAGS.num_examples), logme)
  write_file(os.path.join(
      work_dir, "n%d_pac_gauss.txt" % FLAGS.num_examples), pac_gauss)
  write_file(os.path.join(
      work_dir, "n%d_pac_gauss_half.txt" % FLAGS.num_examples), pac_gauss_half)
  write_file(os.path.join(
      work_dir, "n%d_valerr_half.txt" % FLAGS.num_examples), valerr_half)


if __name__ == "__main__":
  app.run(main)
