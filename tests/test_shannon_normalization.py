# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Regression tests for Shannon choice normalization and public DP operators."""

import itertools
import unittest

import numpy as np

import diff_knapsack as dk


class ShannonNormalizationTest(unittest.TestCase):

  def test_tied_choices_remain_balanced_at_large_common_offsets(self):
    for offset in (1e16, -1e16, 1e30, -1e30):
      for gamma in (0.25, 1.0, 4.0):
        with self.subTest(offset=offset, gamma=gamma):
          value, probability = dk._omega_value_and_choice(
              offset, offset, gamma, 'shannon'
          )
          self.assertTrue(np.isfinite(value))
          self.assertEqual(probability, 0.5)

  def test_choice_depends_on_the_representable_gap_not_the_common_offset(self):
    for offset in (0.0, 2.0**53, -(2.0**53)):
      for gap in (-8.0, -2.0, 0.0, 2.0, 8.0):
        with self.subTest(offset=offset, gap=gap):
          skip, pick, gamma = offset, offset + gap, 2.0
          maximum = max(skip, pick)
          shifted = np.array([skip - maximum, pick - maximum]) / gamma
          weights = np.exp(shifted)
          expected = weights[1] / weights.sum()
          value, probability = dk._omega_value_and_choice(
              skip, pick, gamma, 'shannon'
          )
          np.testing.assert_allclose(probability, expected, rtol=2e-15)
          self.assertGreaterEqual(probability, 0.0)
          self.assertLessEqual(probability, 1.0)
          np.testing.assert_equal(
              value, maximum + gamma * np.log(weights.sum())
          )

  def test_two_item_top_k_has_balanced_selections_and_nonzero_derivatives(self):
    offsets = np.array([0.0, 1e16, -1e16, 1e30, -1e30])
    theta = np.repeat(offsets[:, None], 2, axis=1)
    weights = np.ones_like(theta, dtype=np.int64)
    z = np.broadcast_to(np.array([1.0, -1.0]), theta.shape).copy()
    originals = [array.copy() for array in (theta, weights, z)]
    for gamma in (0.25, 1.0, 4.0):
      with self.subTest(gamma=gamma):
        _, values, choices = dk.dp_value(
            theta, 1, gamma, weights, constraint='top_k'
        )
        selected, states = dk.dp_layer(values, choices, weights)
        gradient = dk.dp_vjp(z, values, choices, states, gamma, weights)
        np.testing.assert_allclose(selected, np.full(theta.shape, 0.5))
        np.testing.assert_allclose(gradient, z * (0.5 / gamma))
        self.assertTrue(np.isfinite(gradient).all())
        np.testing.assert_allclose(states.sum(axis=2), 1.0)
    for array, original in zip((theta, weights, z), originals):
      np.testing.assert_array_equal(array, original)

  def test_public_sampler_does_not_collapse_a_tied_top_k_distribution(self):
    theta = np.array([[1e30, 1e30], [-1e30, -1e30]])
    weights = np.ones_like(theta, dtype=np.int64)
    _, _, choices = dk.dp_value(theta, 1, 1.0, weights, constraint='top_k')
    original = choices.copy()
    samples = dk.dp_sample(choices, weights, num_samples=8192)
    self.assertEqual(samples.shape, (2, 8192, 2))
    self.assertTrue(np.isin(samples, [0.0, 1.0]).all())
    np.testing.assert_array_equal(samples.sum(axis=2), 1.0)
    # A wide tolerance avoids depending on a particular Numba RNG stream.
    np.testing.assert_allclose(samples.mean(axis=1), 0.5, atol=0.05)
    np.testing.assert_array_equal(choices, original)

  def test_ordinary_shannon_operators_match_exact_subset_enumeration(self):
    theta = np.array([[0.2, -0.3, 1.1, 0.6], [-0.8, 0.4, 0.5, 1.0]])
    z = np.array([[0.7, -1.0, 0.2, 0.4], [1.0, -0.4, 0.8, -0.6]])
    subsets = np.array(list(itertools.product((0.0, 1.0), repeat=4)))
    for constraint, capacity in (('knapsack', 3), ('top_k', 1), ('top_k', 2)):
      with self.subTest(constraint=constraint, capacity=capacity):
        weight_row = [1, 2, 0, 3] if constraint == 'knapsack' else [1, 1, 1, 1]
        weights = np.tile(weight_row, (2, 1)).astype(np.int64)
        totals = subsets @ np.array(weight_row)
        feasible = (
            subsets[totals <= capacity]
            if constraint == 'knapsack'
            else subsets[totals == capacity]
        )
        gamma = 0.7
        objective, values, choices = dk.dp_value(
            theta, capacity, gamma, weights, constraint
        )
        selected, states = dk.dp_layer(values, choices, weights)
        gradient = dk.dp_vjp(z, values, choices, states, gamma, weights)
        for batch in range(2):
          scores = feasible @ theta[batch] / gamma
          maximum = scores.max()
          unnormalized = np.exp(scores - maximum)
          probabilities = unnormalized / unnormalized.sum()
          mean = probabilities @ feasible
          covariance = (feasible.T * probabilities) @ feasible - np.outer(
              mean, mean
          )
          np.testing.assert_allclose(
              objective[batch],
              gamma * (maximum + np.log(unnormalized.sum())),
              rtol=1e-13,
          )
          np.testing.assert_allclose(
              selected[batch], mean, rtol=1e-13, atol=1e-14
          )
          np.testing.assert_allclose(
              gradient[batch],
              covariance @ z[batch] / gamma,
              rtol=1e-12,
              atol=1e-14,
          )

  def test_impossible_branches_and_other_regularizers_are_unchanged(self):
    for regularizer in ('shannon', 'gini', 'tsallis'):
      for skip, pick, expected in ((3.0, -np.inf, 0.0), (-np.inf, 3.0, 1.0)):
        with self.subTest(regularizer=regularizer, skip=skip, pick=pick):
          value, probability = dk._omega_value_and_choice(
              skip, pick, 1.0, regularizer
          )
          self.assertEqual(value, 3.0)
          self.assertEqual(probability, expected)
      self.assertEqual(
          dk._omega_value_and_choice(-np.inf, -np.inf, 1.0, regularizer),
          (-np.inf, 0.0),
      )
      if regularizer != 'shannon':
        self.assertEqual(
            dk._omega_value_and_choice(1e30, 1e30, 1.0, regularizer)[1], 0.5
        )


if __name__ == '__main__':
  unittest.main()
