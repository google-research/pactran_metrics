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
"""Tests for registry."""

from unittest import mock

from absl.testing import absltest

from pactran_metrics import registry


class RegistryTest(absltest.TestCase):

  def setUp(self):
    super(RegistryTest, self).setUp()
    # Mock global registry in each test to keep them isolated and allow for
    # concurrent tests.
    self.addCleanup(mock.patch.stopall)
    self.global_registry = dict()
    self.mocked_method = mock.patch.object(
        registry, "global_registry",
        return_value=self.global_registry).start()

  def test_parse_name(self):
    name, kwargs = registry.parse_name("f")
    self.assertEqual(name, "f")
    self.assertEqual(kwargs, {})

    name, kwargs = registry.parse_name("f()")
    self.assertEqual(name, "f")
    self.assertEqual(kwargs, {})

    name, kwargs = registry.parse_name("func(a=0,b=1,c='s')")
    self.assertEqual(name, "func")
    self.assertEqual(kwargs, {"a": 0, "b": 1, "c": "s"})

    name, kwargs = registry.parse_name("foo.bar.func(a=0,b=(1),c='s')")
    self.assertEqual(name, "foo.bar.func")
    self.assertEqual(kwargs, dict(a=0, b=1, c="s"))

    with self.assertRaises(SyntaxError):
      registry.parse_name("func(0")
    with self.assertRaises(SyntaxError):
      registry.parse_name("func(a=0,,b=0)")
    with self.assertRaises(SyntaxError):
      registry.parse_name("func(a=0,b==1,c='s')")
    with self.assertRaises(ValueError):
      registry.parse_name("func(a=0,b=undefined_name,c='s')")
    with self.assertRaises(ValueError):
      registry.parse_name("func(0)")

  def test_register(self):
    # pylint: disable=unused-variable
    @registry.register("func1", "function")
    def func1():
      pass

    @registry.register("A", "object")
    class A(object):
      pass

    @registry.register("B", "factory")
    class B(object):
      pass

    with self.assertRaises(KeyError):
      @registry.register("A", "factory")
      class A1(object):
        pass
    # pylint: enable=unused-variable

    self.assertLen(registry.global_registry(), 3)

  def test_lookup_function(self):

    @registry.register("func1", "function")
    def func1(arg1, arg2, arg3):  # pylint: disable=unused-variable
      return arg1, arg2, arg3

    self.assertTrue(callable(registry.lookup("func1")))
    self.assertEqual(registry.lookup("func1")(1, 2, 3), (1, 2, 3))
    self.assertEqual(
        registry.lookup("func1(arg3=9)")(1, 2), (1, 2, 9))
    self.assertEqual(
        registry.lookup("func1(arg2=9,arg1=99)")(arg3=3), (99, 9, 3))
    self.assertEqual(
        registry.lookup("func1(arg2=9,arg1=99)")(arg1=1, arg3=3),
        (1, 9, 3))

  def test_lookup_object(self):

    @registry.register("A", "object")
    class A(object):

      def __init__(self, arg1, arg2=2):
        self.arg1 = arg1
        self.arg2 = arg2

      def as_tuple(self):
        return (self.arg1, self.arg2)

    self.assertIsInstance(registry.lookup("A(arg1=25)"), A)
    self.assertEqual(registry.lookup("A(arg1=25)").as_tuple(), (25, 2))
    self.assertEqual(
        registry.lookup("A(arg1=8, arg2=9)").as_tuple(), (8, 9))

  def test_lookup_factory(self):

    @registry.register("A", "factory")
    class A(object):  # pylint: disable=unused-variable

      def __init__(self, arg1, arg2=2):
        self.arg1 = arg1
        self.arg2 = arg2

      def as_tuple(self):
        return (self.arg1, self.arg2)

    self.assertTrue(callable(registry.lookup("A(arg1=25)")))
    self.assertEqual(
        registry.lookup("A(arg1=25)")().as_tuple(), (25, 2))
    self.assertEqual(
        registry.lookup("A")(arg1=25).as_tuple(), (25, 2))
    self.assertEqual(
        registry.lookup("A(arg1=9)")(arg2=8).as_tuple(), (9, 8))
    self.assertEqual(
        registry.lookup("A(arg1=8,arg2=9)")().as_tuple(), (8, 9))
    self.assertEqual(
        registry.lookup("A(arg1=8,arg2=9)")(arg1=7).as_tuple(), (7, 9))

if __name__ == "__main__":
  absltest.main()
