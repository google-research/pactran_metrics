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
"""Compute the correlation between metric and eval error."""

import csv

from absl import app
import numpy as np
from scipy import stats
import tensorflow.compat.v1 as tf


def main(_):
  model_dir = ''  # put your saved model_dir here
  results_dir = ''  # put your saved results_dir here
  compute_correlation_vtab([2, 5, 10], model_dir, results_dir)


def compute_correlation_vtab(num_examples, model_dir, results_dir):
  """Compute the correlation between metric and eval error."""
  datasets = ['caltech101', 'oxford_flowers102', 'sun397',
              'cifar', 'oxford_iiit_pet', 'smallnorb',
              'patch_camelyon', 'dmlab', 'ddsm']
  models = ['sup-100', 'sup-rotation-100', 'sup-exemplar-100',
            'semi-exemplar-10', 'semi-rotation-10', 'rotation', 'exemplar',
            'relative-patch-location', 'jigsaw', 'uncond-biggan', 'cond-biggan',
            'wae-mmd', 'vae', 'wae-ukl', 'wae-gan', 'feature_vector']
  metrics = ['valerr_half', 'leep', 'nleep', 'hscore', 'logme',
             'pac_dir', 'pac_gam', 'pac_ndir', 'pac_ngam', 'pac_gauss']

  lrs = [0.1, 0.05, 0.01, 0.005]
  iters = [5000, 10000]
  runs = 5

  for data in datasets:
    print(data)

    # read eval err of each model
    errs = []
    for model in models:
      tf.logging.info('Read model eval error: %s %s', data, model)
      acc_runs = [-1.]

      # full model finetune
      for lr in lrs:
        for it in iters:
          filepath = '%s/%s_incepaug_l2decay_float/%s/%s_%s_%s/result_file.txt' % (
              model_dir, model, data, lr, it, 'False')
          if not tf.io.gfile.exists(filepath):
            tf.logging.info('Not Exist: %s', filepath)
          else:
            with tf.io.gfile.GFile(filepath, 'r') as f:
              csv_reader = csv.reader(f, delimiter=':')
              for row in csv_reader:
                acc_runs.append(float(row[-1]))
                break
      # anil
      filepath = '%s/%s/%s/eval_accuracy.txt' % (
          results_dir, data, model)
      with tf.io.gfile.GFile(filepath, 'r') as f:
        csv_reader = csv.reader(f, delimiter=':')
        for row in csv_reader:
          acc_runs.append(float(row[-1]))
          break

      err = 1. - np.array(acc_runs).max()
      errs.append(err)

    errs = np.array(errs)
    # print(errs)

    # read metrics of each model feature
    for n in num_examples:
      for metric in metrics:
        vals = []
        times = []
        for model in models:
          filepath = '%s/%s/%s/n%d_%s.txt' % (
              results_dir, data, model, n, metric)
          if not tf.io.gfile.exists(filepath):
            print('Not Exist: %s' % filepath)
            continue

          with tf.io.gfile.GFile(filepath, 'r') as f:
            csv_reader = csv.reader(f, delimiter=':')
            for i, row in enumerate(csv_reader):
              if i % 2 == 0:
                vals.append(float(row[-1]))
              else:
                times.append(float(row[-1]))

        vals = np.array(vals)
        vals = np.reshape(vals, [len(models), runs, -1])
        tau = np.zeros([runs, vals.shape[2]])
        rho = np.zeros([runs, vals.shape[2]])
        for r in range(runs):
          for m in range(vals.shape[2]):
            tau[r, m], _ = stats.kendalltau(vals[:, r, m], errs)
            rho[r, m], _ = stats.pearsonr(vals[:, r, m], errs)

        time = np.median(np.array(times))
        tau_m = np.mean(tau, axis=0)
        tau_s = np.std(tau, axis=0) / np.sqrt(5.)
        for m in range(tau_m.shape[0]):
          vm = vals[:, :, m]
          print_metric = metric
          if metric == 'pac_gauss':
            if m % 2 == 0:
              # this is the FR value.
              vm = vals[:, :, m] - vals[:, :, m + 1]
            else:
              # the linear metrics are reported in pac_guass result files.
              print_metric = 'linear'

          print('%s %s %d: %.3f +-%.3f (%.1f %.3f %.3f)' %
                (data, print_metric, n, tau_m[m], tau_s[m], time,
                 np.mean(vm), np.std(vm)))

        if metric == 'valerr_half':
          valerr_half = np.min(vals, axis=2)
          tau = np.zeros([runs])
          for r in range(runs):
            tau[r], _ = stats.kendalltau(valerr_half[:, r], errs)
          print('%s valerr_half_best %d: %.3f' %
                (data, n, np.mean(tau)))
          ref = vals[:, :, 0]
        elif metric == 'pac_gauss':
          tauv = np.zeros([runs, vals.shape[2]])
          for r in range(runs):
            for m in range(vals.shape[2]):
              tauv[r, m], _ = stats.kendalltau(vals[:, r, m], valerr_half[:, r])
          tauv_m = np.mean(tauv, axis=0)
          # for m in range(tauv_m.shape[0]):
          #   print('%s %s %d: %.3f (tau with valerr_half)' %
          #         (data, metric, n, tauv_m[m]))
          tau_m_0 = tau_m[0::2][np.argmax(tauv_m[0::2])]
          tau_m_1 = tau_m[1::2][np.argmax(tauv_m[1::2])]
          print('%s pac_gauss_grid %d: %.3f' % (data, n, tau_m_0))
          print('%s linear_grid %d: %.3f' % (data, n, tau_m_1))


if __name__ == '__main__':
  app.run(main)
