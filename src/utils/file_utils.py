"""
File and path utility functions.

Common operations for working with files and directories
in the electoral roll processing pipeline.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator, Optional, List, Tuple

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def safe_stem(path: Path) -> str:
    """
    Get filesystem-safe stem from path.
    
    Replaces special characters with underscores to ensure
    the result can be used as a directory/file name.
    
    Args:
        path: Path to extract stem from
    
    Returns:
        Sanitized filename stem
    """
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in path.stem
    )


def derive_page_id(image_path: str) -> str:
    """
    Derive page ID from image filename.
    
    Handles various naming patterns:
    - page-003.png -> page-003
    - page-003-img-01.png -> page-003
    - page_003_img_01.jpg -> page-003
    
    Args:
        image_path: Path to image file
    
    Returns:
        Page ID string
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    # Remove image suffix patterns
    page_id = re.sub(r"(?:[-_]img[-_]?\d+)$", "", base, flags=re.IGNORECASE)
    return page_id


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        The same path (for chaining)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_pdfs(input_dir: Path) -> Iterator[Path]:
    """
    Iterate over PDF files in a directory.
    
    Args:
        input_dir: Directory to search
    
    Yields:
        Paths to PDF files (sorted alphabetically)
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return
    
    for path in sorted(input_dir.glob("*.pdf")):
        if path.is_file():
            yield path


def iter_extracted_folders(extracted_dir: Path) -> Iterator[Path]:
    """
    Iterate over extracted folder structures.
    
    Looks for folders with the expected structure:
    extracted/<name>/images/
    
    Args:
        extracted_dir: Root extracted directory
    
    Yields:
        Paths to extracted folders (that contain images/ subdirectory)
    """
    extracted_dir = Path(extracted_dir)
    if not extracted_dir.exists():
        return
    
    for folder in sorted(extracted_dir.iterdir()):
        if not folder.is_dir():
            continue
        
        images_dir = folder / "images"
        if images_dir.exists() and images_dir.is_dir():
            yield folder


def iter_images(images_dir: Path) -> Iterator[Path]:
    """
    Iterate over image files in a directory.
    
    Args:
        images_dir: Directory containing images
    
    Yields:
        Paths to image files (sorted alphabetically)
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return
    
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def list_images(images_dir: Path) -> List[Path]:
    """
    List all image files in a directory.
    
    Args:
        images_dir: Directory containing images
    
    Returns:
        List of image paths (sorted)
    """
    return list(iter_images(images_dir))


def get_page_dirs(crops_dir: Path) -> List[Path]:
    """
    Get sorted list of page directories under crops/.
    
    Args:
        crops_dir: Crops directory path
    
    Returns:
        Sorted list of page directories
    """
    crops_dir = Path(crops_dir)
    if not crops_dir.exists():
        return []
    
    return sorted([
        p for p in crops_dir.iterdir()
        if p.is_dir()
    ])


def resolve_images_dir(page_dir: Path) -> Path:
    """
    Resolve the actual images directory for a page.
    
    Some extractions store images directly under page_dir,
    others under page_dir/images.
    
    Args:
        page_dir: Page directory path
    
    Returns:
        Resolved images directory path
    """
    page_dir = Path(page_dir)
    images_subdir = page_dir / "images"
    
    if images_subdir.exists() and images_subdir.is_dir():
        return images_subdir
    return page_dir


def get_relative_path(path: Path, base: Path) -> str:
    """
    Get path relative to base directory.
    
    Args:
        path: Full path
        base: Base directory
    
    Returns:
        Relative path string
    """
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def count_files_by_extension(directory: Path) -> dict[str, int]:
    """
    Count files by extension in a directory.
    
    Args:
        directory: Directory to scan
    
    Returns:
        Dictionary mapping extension to count
    """
    directory = Path(directory)
    if not directory.exists():
        return {}
    
    counts: dict[str, int] = {}
    for path in directory.rglob("*"):
        if path.is_file():
            ext = path.suffix.lower() or "(no extension)"
            counts[ext] = counts.get(ext, 0) + 1
    
    return counts
