"""matheo - Matheo is a python package with mathematical algorithms for use in earth observation data and tools."""

__all__ = []
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("matheo")
except PackageNotFoundError:
    __version__ = "unknown"
