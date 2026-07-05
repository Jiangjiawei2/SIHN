import unittest

import numpy as np

from sihn.utils.masks import (
    cartesian_random_mask,
    equispaced_mask,
    estimate_acceleration,
    radial_mask,
)


class MaskGenerationTest(unittest.TestCase):
    def test_cartesian_random_mask_keeps_center_and_matches_acceleration(self):
        mask = cartesian_random_mask((64, 64), acceleration=4, center_fraction=0.08, seed=2024)

        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(mask.dtype, np.float32)
        self.assertTrue(np.all(mask[:, 30:34] == 1.0))
        self.assertLess(abs(estimate_acceleration(mask) - 4.0), 0.5)

    def test_equispaced_mask_is_deterministic(self):
        first = equispaced_mask((64, 64), acceleration=8, center_fraction=0.06)
        second = equispaced_mask((64, 64), acceleration=8, center_fraction=0.06)

        np.testing.assert_array_equal(first, second)
        self.assertLess(abs(estimate_acceleration(first) - 8.0), 1.0)

    def test_radial_mask_has_center_coverage(self):
        mask = radial_mask((64, 64), acceleration=6, seed=7)

        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(mask[32, 32], 1.0)
        self.assertGreater(mask.mean(), 0.0)
        self.assertLess(mask.mean(), 0.5)


if __name__ == "__main__":
    unittest.main()
