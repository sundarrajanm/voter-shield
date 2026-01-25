"""
Field Cropper processor.

Crops specific data field regions from voter images and stitches them
back into a single compact image to reduce file size.

Fields extracted:
- serial_no (top left area)
- epic_no (top right area)
- name (first text line)
- relation (father/mother/husband name line)
- house_no
- age
- gender
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from .base import BaseProcessor, ProcessingContext
from ..logger import get_logger

logger = get_logger("field_cropper")


# Field region definitions as fractions of image dimensions
# Format: (x1_frac, y1_frac, x2_frac, y2_frac)
#
# Voter card layout (typical Tamil Nadu electoral roll):
# +------------------------------------------+
# |          |       EPIC Number             |  <- EPIC_ROI region
# +----------+-------------------------------+
# |          |  Name                         |
# |  PHOTO   |  Relation                     |  <- OTHER_FIELDS_ROI region
# |   BOX    |  House No | Age | Gender      |
# |          |                               |
# +----------+-------------------------------+
#
# Strategy:
# 1. Extract the EPIC region using EPIC_ROI
# 2. Extract the other fields region using OTHER_FIELDS_ROI
# 3. Stitch: EPIC on top, other fields below

# EPIC number region (x1, y1, x2, y2) as fractions
# Located in the top-right area of the voter card
EPIC_ROI = (0.662, 0.064, 0.980, 0.194)

# Other fields region (Name, Relation, House No, Age, Gender)
# Located in the center-left area, below the EPIC row
OTHER_FIELDS_ROI = (0.028, 0.225, 0.735, 0.909)

# Output image settings
STITCH_SPACING = 2  # Pixels between stitched regions
STITCH_BG_COLOR = 255  # White background


@dataclass
class FieldCrop:
    """Information about a cropped field region."""
    field_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    image: Optional[np.ndarray] = None
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class FieldCropResult:
    """Result of field cropping for a single voter image."""
    image_name: str
    original_size_bytes: int
    cropped_size_bytes: int
    compression_ratio: float
    fields_extracted: int
    output_path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class PageFieldCropResult:
    """Result of field cropping for a page."""
    page_id: str
    images_processed: int
    total_original_bytes: int
    total_cropped_bytes: int
    overall_compression: float
    elapsed_seconds: float
    results: List[FieldCropResult] = field(default_factory=list)


@dataclass
class FieldCropSummary:
    """Summary of field cropping for a document."""
    pdf_name: str
    total_pages: int
    total_images: int
    total_original_mb: float
    total_cropped_mb: float
    overall_compression: float
    elapsed_seconds: float


class FieldCropper(BaseProcessor):
    """
    Crop specific data field regions from voter images.
    
    Takes cropped voter box images and extracts only the essential
    text field regions (excluding photo and whitespace), then
    stitches them into a compact single image.
    
    This significantly reduces storage requirements while preserving
    all the necessary text data for OCR.
    """
    
    name = "FieldCropper"
    
    def __init__(
        self,
        context: ProcessingContext,
        output_suffix: str = "_compact",
        keep_original: bool = True,
        stitch_layout: str = "vertical",  # "vertical" or "horizontal"
    ):
        """
        Initialize field cropper.
        
        Args:
            context: Processing context
            output_suffix: Suffix to add to output filenames
            keep_original: Whether to keep original images
            stitch_layout: How to stitch fields - "vertical" (stacked) or "horizontal" (side by side)
        """
        super().__init__(context)
        self.output_suffix = output_suffix
        self.keep_original = keep_original
        self.stitch_layout = stitch_layout
        self.page_results: List[PageFieldCropResult] = []
        self.summary: Optional[FieldCropSummary] = None
    
    def validate(self) -> bool:
        """Validate prerequisites."""
        if not self.context.crops_dir:
            self.log_error("Crops directory not set")
            return False
        
        if not self.context.crops_dir.exists():
            self.log_error(f"Crops directory not found: {self.context.crops_dir}")
            return False
        
        return True
    
    def process(self) -> bool:
        """
        Process all cropped voter images.
        
        Returns:
            True if processing succeeded
        """
        import time
        
        crops_dir = self.context.crops_dir
        
        # Find page directories
        page_dirs = [
            d for d in crops_dir.iterdir()
            if d.is_dir() and d.name.startswith("page-")
        ]
        page_dirs.sort(key=lambda p: p.name)
        
        if not page_dirs:
            self.log_warning(f"No page directories found in {crops_dir}")
            return False
        
        self.log_info(f"Found {len(page_dirs)} page(s) with crops")
        
        # Create base output directory: /extracted/compact-crop/<pdf_name>/
        compact_base_dir = self.context.config.extracted_dir / "compact-crop" / self.context.pdf_name
        compact_base_dir.mkdir(parents=True, exist_ok=True)
        
        run_start = time.perf_counter()
        total_images = 0
        total_original_bytes = 0
        total_cropped_bytes = 0
        
        for page_dir in page_dirs:
            page_result = self._process_page(page_dir, compact_base_dir)
            self.page_results.append(page_result)
            total_images += page_result.images_processed
            total_original_bytes += page_result.total_original_bytes
            total_cropped_bytes += page_result.total_cropped_bytes
        
        elapsed = time.perf_counter() - run_start
        
        # Calculate summary
        overall_compression = 1.0
        if total_original_bytes > 0:
            overall_compression = total_cropped_bytes / total_original_bytes
        
        self.summary = FieldCropSummary(
            pdf_name=self.context.pdf_name,
            total_pages=len(page_dirs),
            total_images=total_images,
            total_original_mb=total_original_bytes / (1024 * 1024),
            total_cropped_mb=total_cropped_bytes / (1024 * 1024),
            overall_compression=overall_compression,
            elapsed_seconds=elapsed,
        )
        
        self.log_info(
            f"Field cropping complete",
            pages=len(page_dirs),
            images=total_images,
            original_mb=f"{self.summary.total_original_mb:.2f}",
            cropped_mb=f"{self.summary.total_cropped_mb:.2f}",
            compression=f"{overall_compression:.1%}",
            time=f"{elapsed:.2f}s"
        )
        
        return True
    
    def _process_page(self, page_dir: Path, compact_base_dir: Path) -> PageFieldCropResult:
        """
        Process all images in a page directory.
        
        Args:
            page_dir: Path to page directory containing crop images
            compact_base_dir: Base directory for compact output (/extracted/compact-crop/<pdf_name>/)
            
        Returns:
            PageFieldCropResult
        """
        import time
        
        page_start = time.perf_counter()
        page_id = page_dir.name
        
        # List image files
        exts = {".png", ".jpg", ".jpeg"}
        images = [
            p for p in page_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]
        images.sort(key=lambda p: p.name)
        
        if not images:
            return PageFieldCropResult(
                page_id=page_id,
                images_processed=0,
                total_original_bytes=0,
                total_cropped_bytes=0,
                overall_compression=1.0,
                elapsed_seconds=0.0,
            )
        
        # Create output directory: /extracted/compact-crop/<pdf_name>/<page_name>/
        compact_dir = compact_base_dir / page_id
        compact_dir.mkdir(parents=True, exist_ok=True)
        
        results: List[FieldCropResult] = []
        total_original = 0
        total_cropped = 0
        
        # Process images in parallel
        max_workers = min(os.cpu_count() or 4, 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self._process_image, img_path, compact_dir): img_path
                for img_path in images
            }
            
            for future in as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    total_original += result.original_size_bytes
                    total_cropped += result.cropped_size_bytes
                except Exception as e:
                    self.log_error(f"Error processing {img_path.name}: {e}")
                    results.append(FieldCropResult(
                        image_name=img_path.name,
                        original_size_bytes=0,
                        cropped_size_bytes=0,
                        compression_ratio=1.0,
                        fields_extracted=0,
                        error=str(e)
                    ))
        
        elapsed = time.perf_counter() - page_start
        
        overall_compression = 1.0
        if total_original > 0:
            overall_compression = total_cropped / total_original
        
        self.log_debug(
            f"Page {page_id}: processed {len(results)} images",
            compression=f"{overall_compression:.1%}",
            time=f"{elapsed:.2f}s"
        )
        
        return PageFieldCropResult(
            page_id=page_id,
            images_processed=len(results),
            total_original_bytes=total_original,
            total_cropped_bytes=total_cropped,
            overall_compression=overall_compression,
            elapsed_seconds=elapsed,
            results=results,
        )
    
    def _process_image(self, img_path: Path, output_dir: Path) -> FieldCropResult:
        """
        Process a single voter image using ROI-based extraction.
        
        Strategy:
        1. Extract EPIC region using EPIC_ROI coordinates
        2. Extract other fields region using OTHER_FIELDS_ROI coordinates
        3. Stitch: EPIC on top, other fields below
        
        Args:
            img_path: Path to voter image
            output_dir: Directory to save compact image
            
        Returns:
            FieldCropResult
        """
        # Read image
        img = cv2.imdecode(
            np.fromfile(str(img_path), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
        
        if img is None:
            return FieldCropResult(
                image_name=img_path.name,
                original_size_bytes=0,
                cropped_size_bytes=0,
                compression_ratio=1.0,
                fields_extracted=0,
                error="Failed to load image"
            )
        
        original_size = img_path.stat().st_size
        H, W = img.shape[:2]
        
        # Extract EPIC region using ROI coordinates
        epic_x1 = int(W * EPIC_ROI[0])
        epic_y1 = int(H * EPIC_ROI[1])
        epic_x2 = int(W * EPIC_ROI[2])
        epic_y2 = int(H * EPIC_ROI[3])
        epic_region = img[epic_y1:epic_y2, epic_x1:epic_x2]
        
        # Extract other fields region using ROI coordinates
        fields_x1 = int(W * OTHER_FIELDS_ROI[0])
        fields_y1 = int(H * OTHER_FIELDS_ROI[1])
        fields_x2 = int(W * OTHER_FIELDS_ROI[2])
        fields_y2 = int(H * OTHER_FIELDS_ROI[3])
        fields_region = img[fields_y1:fields_y2, fields_x1:fields_x2]
        
        # Validate regions
        if epic_region.size == 0 or fields_region.size == 0:
            return FieldCropResult(
                image_name=img_path.name,
                original_size_bytes=original_size,
                cropped_size_bytes=original_size,
                compression_ratio=1.0,
                fields_extracted=0,
                error="Invalid region extraction"
            )
        
        # Make widths consistent for stacking
        epic_w = epic_region.shape[1]
        fields_w = fields_region.shape[1]
        max_width = max(epic_w, fields_w)
        
        # Pad EPIC region if needed (center it)
        if epic_w < max_width:
            left_pad = (max_width - epic_w) // 2
            right_pad = max_width - epic_w - left_pad
            epic_region = np.hstack([
                np.full((epic_region.shape[0], left_pad), STITCH_BG_COLOR, dtype=np.uint8),
                epic_region,
                np.full((epic_region.shape[0], right_pad), STITCH_BG_COLOR, dtype=np.uint8)
            ])
        
        # Pad fields region if needed (left-align)
        if fields_w < max_width:
            padding = np.full((fields_region.shape[0], max_width - fields_w), STITCH_BG_COLOR, dtype=np.uint8)
            fields_region = np.hstack([fields_region, padding])
        
        # Create spacing row
        spacing = np.full((STITCH_SPACING, max_width), STITCH_BG_COLOR, dtype=np.uint8)
        
        # Stack vertically: EPIC -> spacing -> other fields
        compact_img = np.vstack([epic_region, spacing, fields_region])
        
        # Save compact image
        output_name = img_path.stem + self.output_suffix + ".png"
        output_path = output_dir / output_name
        
        # Encode to measure size
        success, encoded = cv2.imencode('.png', compact_img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if not success:
            return FieldCropResult(
                image_name=img_path.name,
                original_size_bytes=original_size,
                cropped_size_bytes=original_size,
                compression_ratio=1.0,
                fields_extracted=2,
                error="Failed to encode compact image"
            )
        
        cropped_size = len(encoded)
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(encoded)
        
        compression_ratio = cropped_size / original_size if original_size > 0 else 1.0
        
        return FieldCropResult(
            image_name=img_path.name,
            original_size_bytes=original_size,
            cropped_size_bytes=cropped_size,
            compression_ratio=compression_ratio,
            fields_extracted=2,  # EPIC + other fields
            output_path=output_path,
        )
    
    def process_single_image(self, img: np.ndarray) -> np.ndarray:
        """
        Process a single image and return the compact version.
        
        Uses ROI-based extraction:
        1. EPIC region using EPIC_ROI coordinates
        2. Other fields region using OTHER_FIELDS_ROI coordinates
        
        This is useful for inline processing during OCR.
        
        Args:
            img: Input voter image (grayscale or color)
            
        Returns:
            Compact stitched image
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        H, W = gray.shape[:2]
        
        # Extract EPIC region
        epic_x1 = int(W * EPIC_ROI[0])
        epic_y1 = int(H * EPIC_ROI[1])
        epic_x2 = int(W * EPIC_ROI[2])
        epic_y2 = int(H * EPIC_ROI[3])
        epic_region = gray[epic_y1:epic_y2, epic_x1:epic_x2]
        
        # Extract other fields region
        fields_x1 = int(W * OTHER_FIELDS_ROI[0])
        fields_y1 = int(H * OTHER_FIELDS_ROI[1])
        fields_x2 = int(W * OTHER_FIELDS_ROI[2])
        fields_y2 = int(H * OTHER_FIELDS_ROI[3])
        fields_region = gray[fields_y1:fields_y2, fields_x1:fields_x2]
        
        # Validate regions
        if epic_region.size == 0 or fields_region.size == 0:
            return gray
        
        # Make widths consistent
        epic_w = epic_region.shape[1]
        fields_w = fields_region.shape[1]
        max_width = max(epic_w, fields_w)
        
        # Pad EPIC region if needed (center it)
        if epic_w < max_width:
            left_pad = (max_width - epic_w) // 2
            right_pad = max_width - epic_w - left_pad
            epic_region = np.hstack([
                np.full((epic_region.shape[0], left_pad), STITCH_BG_COLOR, dtype=np.uint8),
                epic_region,
                np.full((epic_region.shape[0], right_pad), STITCH_BG_COLOR, dtype=np.uint8)
            ])
        
        # Pad fields region if needed
        if fields_w < max_width:
            padding = np.full((fields_region.shape[0], max_width - fields_w), STITCH_BG_COLOR, dtype=np.uint8)
            fields_region = np.hstack([fields_region, padding])
        
        # Create spacing row
        spacing = np.full((STITCH_SPACING, max_width), STITCH_BG_COLOR, dtype=np.uint8)
        
        # Stack vertically: EPIC -> spacing -> other fields
        return np.vstack([epic_region, spacing, fields_region])


def crop_fields(
    extracted_dir: Path,
    output_suffix: str = "_compact",
    stitch_layout: str = "vertical",
) -> FieldCropSummary:
    """
    Convenience function to crop fields from all voter images.
    
    Args:
        extracted_dir: Path to extracted folder
        output_suffix: Suffix for output files
        stitch_layout: "vertical" or "horizontal"
    
    Returns:
        FieldCropSummary
    """
    from ..config import Config
    
    config = Config()
    context = ProcessingContext(config=config)
    context.setup_paths_from_extracted(extracted_dir)
    
    cropper = FieldCropper(
        context,
        output_suffix=output_suffix,
        stitch_layout=stitch_layout,
    )
    
    if not cropper.run():
        raise RuntimeError("Field cropping failed")
    
    return cropper.summary
