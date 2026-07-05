"""SIHN package.

The top-level package avoids importing PyTorch eagerly so utility modules such
as configuration and mask generation remain usable in lightweight environments.
"""

__all__ = ["SIHN"]


def __getattr__(name):
    if name == "SIHN":
        from sihn.models.sihn import SIHN

        return SIHN
    raise AttributeError(name)
