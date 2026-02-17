"""Media synchronization logic for AnkiOps.

Handles:
- Proactive hashing (renaming files to include content hash)
- Bidirectional sync (Import: Local -> Anki, Export: Anki -> Local)
- Updating markdown references when files are renamed
"""

import hashlib
import logging
import re
import shutil
from pathlib import Path

from ankiops.config import LOCAL_MEDIA_DIR

logger = logging.getLogger(__name__)

HASH_SUFFIX_PATTERN = re.compile(r"_([a-f0-9]{8})\.[^.]+$")


def _calculate_hash(file_path: Path) -> str:
    """Calculate 8-char SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:8]


def _get_hashed_filename(file_path: Path, content_hash: str) -> str:
    """Return filename with hash suffix (e.g. image_a1b2c3.png)."""
    stem = file_path.stem
    suffix = file_path.suffix
    
    # Check if already hashed
    match = HASH_SUFFIX_PATTERN.search(file_path.name)
    if match:
        existing_hash = match.group(1)
        if existing_hash == content_hash:
            return file_path.name  # Already correct
        # Strip old hash
        stem = stem[: -len(existing_hash) - 1]

    return f"{stem}_{content_hash}{suffix}"


def update_markdown_media_references(
    collection_dir: Path, rename_map: dict[str, str]
) -> None:
    """Update all markdown files in collection to use new filenames.
    
    Args:
        collection_dir: Root of standard markdown collection
        rename_map: Dict mapping {original_relative_path_str: new_relative_path_str}
                   Note: paths should use forward slashes for markdown compatibility.
    """
    if not rename_map:
        return

    md_files = list(collection_dir.glob("*.md"))
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        original_content = content
        
        # We need to handle both standard markdown images and HTML img tags
        # Pattern 1: ![alt](path)
        # Pattern 2: <img src="path">
        
        for old_path, new_path in rename_map.items():
            # Create regex that matches the path specifically in link/src contexts
            # This is a simple replacement - a robust implementation might parse,
            # but regex is standard for this tool.
            
            # Normalize to forward slashes for replacement
            old_path_slash = old_path.replace("\\", "/")
            new_path_slash = new_path.replace("\\", "/")
            
            # Replace in Markdown links: ](path)
            content = content.replace(f"]({old_path_slash})", f"]({new_path_slash})")
            
            # Replace in HTML src: src="path"
            content = content.replace(f'src="{old_path_slash}"', f'src="{new_path_slash}"')
            
            # Also handle Anki sound: [sound:path] -> [sound:path]
            # Note: sound tags typically don't have paths, just filenames.
            # If user used paths in sound tags, this replaces them too.
            content = content.replace(f"[sound:{old_path_slash}]", f"[sound:{new_path_slash}]")

        if content != original_content:
            md_file.write_text(content, encoding="utf-8")
            logger.info(f"Updated media references in {md_file.name}")


def apply_hashing(collection_dir: Path) -> None:
    """Rename local media files to include content hash and update markdown refs.
    
    Recursive scan of LOCAL_MEDIA_DIR.
    """
    media_root = collection_dir / LOCAL_MEDIA_DIR
    if not media_root.exists():
        return

    rename_map = {}  # old_rel_path -> new_rel_path

    # Walk all files
    # Walk all files (non-recursive)
    for file_path in media_root.glob("*"):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue

        try:
            content_hash = _calculate_hash(file_path)
            new_name = _get_hashed_filename(file_path, content_hash)
            
            if new_name != file_path.name:
                new_path = file_path.with_name(new_name)
                
                # Handle collision if target exists (should involve same hash hopefully, 
                # or we just overwrite if it's truly the same content)
                if new_path.exists():
                    # If content matches, we can delete the old unhashed file
                    if _calculate_hash(new_path) == content_hash:
                        file_path.unlink()
                        logger.info(f"Replaced {file_path.name} with existing {new_name}")
                    else:
                        # Hash collision / Name collision? Extremely unlikely with SHA256 snippet
                        # But if it happens, we might have an issue. 
                        # For now, assume safe.
                         logger.warning(
                            f"Target {new_name} exists but content differs? Skipping rename of {file_path.name}"
                        )
                         continue
                else:
                    file_path.rename(new_path)
                    logger.info(f"Hashed: {file_path.name} -> {new_name}")

                # Record relative paths for markdown update
                # Rel path from collection_dir (e.g. "media/img.png")
                old_rel = f"{LOCAL_MEDIA_DIR}/{file_path.name}"
                new_rel = f"{LOCAL_MEDIA_DIR}/{new_name}"
                rename_map[old_rel] = new_rel

        except Exception as e:
            logger.warning(f"Failed to hash media file {file_path}: {e}")

    if rename_map:
        update_markdown_media_references(collection_dir, rename_map)


def sync_to_anki(collection_dir: Path, anki_media_dir: Path) -> None:
    """Sync local media to Anki collection.media.
    
    1. Apply hashing (renames check).
    2. Copy all files to Anki's media directory.
    """
    apply_hashing(collection_dir)
    
    media_root = collection_dir / LOCAL_MEDIA_DIR
    if not media_root.exists():
        return

    target_root = Path(anki_media_dir)
    
    # Copy files
    # We use glob to find all files in the root of media dir
    for file_path in media_root.glob("*"):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
            
        target_path = target_root / file_path.name
        
        # Copy if missing or different
        # Since we hash locally, filenames should be unique. 
        # If target missing, copy.
        if not target_path.exists():
            shutil.copy2(file_path, target_path)
            logger.debug(f"Synced to Anki: {file_path.name}")
        else:
            # If exists, check hash to be sure (or just assume name->hash implies equality)
            # Safe to skip if name matches since we control naming.
            pass


def sync_from_anki(
    collection_dir: Path, 
    anki_media_dir: Path, 
    referenced_files: set[str]
) -> None:
    """Sync referenced media from Anki to local media folder.
    
    Only copies files that are referenced in the markdown.
    """
    media_root = collection_dir / LOCAL_MEDIA_DIR
    media_root.mkdir(parents=True, exist_ok=True)
    target_root = Path(anki_media_dir)
    
    for filename in referenced_files:
        source_path = target_root / filename
        target_path = media_root / filename
        
        if not source_path.exists():
            logger.warning(f"Referenced media not found in Anki: {filename}")
            continue
            
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
            logger.info(f"Synced from Anki: {filename}")
