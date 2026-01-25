"""
ID Field Cropper processor.

Crops Serial Number and house_no from voter images for AI processing.

This creates a horizontally stitched image: [Serial Number] [Separator] [House Number]
The AI will use Serial Number as a reference to maintain proper sequence alignment.

Updated ROIs (measured from actual voter images):
SERIAL_NO_ROI = (0.200883, 0.014433, 0.367550, 0.135440)
HOUSE_ROI = (0.306843, 0.425564, 0.491170, 0.538425)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from .base import BaseProcessor, ProcessingContext
from ..logger import get_logger

logger = get_logger("id_field_cropper")


# Field ROI - Serial Number and House Number (for AI processing)
# Format: (x1_frac, y1_frac, x2_frac, y2_frac)
# Measured from actual voter images
SERIAL_NO_ROI = (0.200883, 0.014433, 0.367550, 0.135440)
EPIC_ROI = (0.464680, 0.011625, 0.845475, 0.167043)  # Not used, kept for reference
HOUSE_ROI = (0.306843, 0.425564, 0.491170, 0.538425)
STITCH_SPACING = 15  # Pixels between serial and house regions
STITCH_BG_COLOR = 255  # White background


@dataclass
class IdCropResult:
    """Result of ID field cropping for a single voter image."""
    image_name: str
    output_path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class IdFieldCropperSummary:
    """Summary of ID field cropping."""
    total_images: int = 0
    successful_crops: int = 0
    failed_crops: int = 0


class IdFieldCropper(BaseProcessor):
    """
    Crops the house number field from voter images.
    """
    
    name = "IdFieldCropper"
    
    def __init__(self, context: ProcessingContext):
        super().__init__(context)
        self.summary = IdFieldCropperSummary()

    def validate(self) -> bool:
        """Validate prerequisites."""
        if not self.context.crops_dir:
            self.log_error("Crops directory not set")
            return False
        
        if not self.context.crops_dir.exists():
            self.log_error(f"Crops directory not found: {self.context.crops_dir}")
            return False
            
        if not self.context.id_crops_dir:
            self.log_error("ID Crops directory not set")
            return False
            
        return True

    def process(self) -> bool:
        """Process all cropped voter images."""
        crops_dir = self.context.crops_dir
        
        # Find page directories
        page_dirs = sorted([
            d for d in crops_dir.iterdir()
            if d.is_dir() and d.name.startswith("page-")
        ])
        
        if not page_dirs:
            self.log_warning(f"No page directories found in {crops_dir}")
            return False
        
        self.log_info(f"Found {len(page_dirs)} page(s) with crops")
        
        # Process pages
        total_images = 0
        successful = 0
        failed = 0
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(os.cpu_count() or 4, 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for page_dir in page_dirs:
                # Create output dir for page
                page_output_dir = self.context.id_crops_dir / page_dir.name
                page_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Get images
                images = sorted([
                    p for p in page_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ])
                
                total_images += len(images)
                
                for img_path in images:
                    futures.append(
                        executor.submit(self._process_image, img_path, page_output_dir)
                    )
            
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                if result.error:
                    failed += 1
                else:
                    successful += 1
        
        self.summary.total_images = total_images
        self.summary.successful_crops = successful
        self.summary.failed_crops = failed
        
        self.log_info(
            f"ID Field Cropping complete: {successful}/{total_images} successful, {failed} failed"
        )
        
        return True

    def _process_image(self, img_path: Path, output_dir: Path) -> IdCropResult:
        """
        Process a single voter image.
        
        Extracts Serial Number and House Number ROIs and stitches them horizontally:
        [Serial Number] [Separator] [House Number]
        
        This allows AI to use Serial Number as a reference point for proper sequence alignment.
        """
        try:
            # Read image
            img = cv2.imdecode(
                np.fromfile(str(img_path), dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE
            )
            
            if img is None:
                return IdCropResult(img_path.name, error="Failed to load image")
            
            H, W = img.shape[:2]
            
            # Extract Serial Number ROI
            serial_x1 = int(W * SERIAL_NO_ROI[0])
            serial_y1 = int(H * SERIAL_NO_ROI[1])
            serial_x2 = int(W * SERIAL_NO_ROI[2])
            serial_y2 = int(H * SERIAL_NO_ROI[3])
            
            # Clamp serial coordinates
            serial_x1, serial_y1 = max(0, serial_x1), max(0, serial_y1)
            serial_x2, serial_y2 = min(W, serial_x2), min(H, serial_y2)
            
            serial_crop = img[serial_y1:serial_y2, serial_x1:serial_x2]
            
            # Extract house number ROI
            house_x1 = int(W * HOUSE_ROI[0])
            house_y1 = int(H * HOUSE_ROI[1])
            house_x2 = int(W * HOUSE_ROI[2])
            house_y2 = int(H * HOUSE_ROI[3])
            
            # Clamp house coordinates
            house_x1, house_y1 = max(0, house_x1), max(0, house_y1)
            house_x2, house_y2 = min(W, house_x2), min(H, house_y2)
            
            house_crop = img[house_y1:house_y2, house_x1:house_x2]
            
            # Validate crops
            if serial_crop.size == 0:
                return IdCropResult(img_path.name, error="Invalid Serial Number ROI dimensions")
            if house_crop.size == 0:
                return IdCropResult(img_path.name, error="Invalid house number ROI dimensions")
            
            # Stitch Serial and House horizontally: [Serial] [Separator] [House]
            serial_h, serial_w = serial_crop.shape[:2]
            house_h, house_w = house_crop.shape[:2]
            max_h = max(serial_h, house_h)
            
            # Pad Serial vertically to match height
            if serial_h < max_h:
                top = (max_h - serial_h) // 2
                bottom = max_h - serial_h - top
                serial_crop = np.vstack([
                    np.full((top, serial_w), STITCH_BG_COLOR, dtype=np.uint8),
                    serial_crop,
                    np.full((bottom, serial_w), STITCH_BG_COLOR, dtype=np.uint8)
                ])
            
            # Pad house vertically to match height
            if house_h < max_h:
                top = (max_h - house_h) // 2
                bottom = max_h - house_h - top
                house_crop = np.vstack([
                    np.full((top, house_w), STITCH_BG_COLOR, dtype=np.uint8),
                    house_crop,
                    np.full((bottom, house_w), STITCH_BG_COLOR, dtype=np.uint8)
                ])
            
            # Create horizontal separator/spacer
            spacer = np.full((max_h, STITCH_SPACING), STITCH_BG_COLOR, dtype=np.uint8)
            
            # Stitch together: [Serial] [Spacer] [House]
            stitched = np.hstack([serial_crop, spacer, house_crop])
            
            # Save
            output_path = output_dir / img_path.name
            success, encoded = cv2.imencode(".png", stitched)
            if success:
                encoded.tofile(str(output_path))
                return IdCropResult(img_path.name, output_path=output_path)
            else:
                return IdCropResult(img_path.name, error="Failed to encode image")

        except Exception as e:
            return IdCropResult(img_path.name, error=str(e))

