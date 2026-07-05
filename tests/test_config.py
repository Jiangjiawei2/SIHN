import unittest
from pathlib import Path

from sihn.config import load_config


class ConfigTest(unittest.TestCase):
    def test_fastmri_config_contains_reproducibility_fields(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "fastmri_singlecoil.yaml")

        self.assertEqual(cfg["experiment"]["seed"], 2024)
        self.assertEqual(cfg["model"]["stages"], 5)
        self.assertIn("train", cfg)
        self.assertIn("mask", cfg)

    def test_ixi_config_matches_manuscript_dataset(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "ixi_singlecoil.yaml")

        self.assertEqual(cfg["data"]["dataset"], "ixi")
        self.assertEqual(cfg["data"]["image_size"], [256, 256])
        self.assertEqual(cfg["experiment"]["seed"], 2024)


if __name__ == "__main__":
    unittest.main()
