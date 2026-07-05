import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for model shape tests")
class SIHNShapeTest(unittest.TestCase):
    def test_singlecoil_forward_shape(self):
        import torch

        from sihn.models.sihn import SIHN

        model = SIHN(in_channels=2, out_channels=2, base_channels=8, stages=5, semantic_nodes=16)
        kspace = torch.randn(1, 64, 64, dtype=torch.complex64)
        mask = torch.ones(1, 64, 64)

        out = model(kspace, mask)

        self.assertEqual(tuple(out.shape), (1, 2, 64, 64))

    def test_multicoil_forward_shape(self):
        import torch

        from sihn.models.sihn import SIHN

        model = SIHN(in_channels=2, out_channels=2, base_channels=8, stages=5, semantic_nodes=16)
        kspace = torch.randn(1, 4, 64, 64, dtype=torch.complex64)
        mask = torch.ones(1, 1, 64, 64)
        sensitivity = torch.randn(1, 4, 64, 64, dtype=torch.complex64)

        out = model(kspace, mask, sensitivity)

        self.assertEqual(tuple(out.shape), (1, 2, 64, 64))


if __name__ == "__main__":
    unittest.main()
