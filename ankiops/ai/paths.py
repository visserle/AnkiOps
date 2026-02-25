"""AI asset path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ankiops.config import AI_DIR, AI_PROMPTS_DIR


@dataclass(frozen=True)
class AIPaths:
    """Resolved filesystem locations for collection-scoped AI assets."""

    root: Path
    models: Path
    prompts: Path

    @classmethod
    def from_collection_dir(cls, collection_dir: Path) -> AIPaths:
        """Build AI asset paths from a collection directory."""
        root = collection_dir / AI_DIR
        return cls(
            root=root,
            models=root / "models.yaml",
            prompts=root / AI_PROMPTS_DIR,
        )
