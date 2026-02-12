from importlib.metadata import version

from deckops.cli import main

__version__ = version("deckops")
__all__ = ["main", "__version__"]
