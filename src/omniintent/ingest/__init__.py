"""
omniintent.ingest package
=========================
Convenience re-exports so callers can do:

    from omniintent.ingest import Quest3Dataset, load_quest3
"""

from .quest3_ingest import load as load_quest3  # noqa: F401
from .quest3_dataset import Quest3Dataset  # noqa: F401

__all__ = ["Quest3Dataset", "load_quest3"]
