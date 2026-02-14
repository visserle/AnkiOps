from importlib.metadata import version

from ankiops.cli import main

__version__ = version("ankiops")
__all__ = ["main", "__version__"]
