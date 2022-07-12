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
"""Tests for the trainer module."""

from absl.testing import absltest
from pactran_metrics import trainer
import tensorflow.compat.v1 as tf


class TrainerTest(absltest.TestCase):

  def test_get_group_of_ops(self):
    dummy_net = tf.Variable(0.0) + 0.0
    group_of_ops = trainer.get_group_of_ops(
        dummy_net,
        initial_learning_rate=0.01,
        momentum=0.9,
        lr_decay_factor=0.1,
        decay_steps=(1000, 2000, 3000),
        warmup_steps=0)
    self.assertIsInstance(group_of_ops, tf.Operation)

  def test_apply_warmup_lr(self):
    global_step = tf.train.get_or_create_global_step()
    decay_steps = (1000, 2000, 3000)
    warmup_steps = 0
    initial_learning_rate = 0.01
    lr_decay_factor = 0.1

    lr = tf.train.piecewise_constant(global_step, decay_steps, [
        initial_learning_rate * (lr_decay_factor**i)
        for i in range(len(decay_steps) + 1)
    ])

    lr = trainer.apply_warmup_lr(global_step, lr, initial_learning_rate,
                                 warmup_steps)
    init = tf.global_variables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(init)
    lr = sess.run(lr)

if __name__ == '__main__':
  absltest.main()
