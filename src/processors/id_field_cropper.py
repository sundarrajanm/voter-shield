"""
ID Field Cropper processor.

Crops house_no from voter images.

ROI:
HOUSE_ROI = (0.303532, 0.410835, 0.728477, 0.559819)
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


# Field ROI - House Number only
# Format: (x1_frac, y1_frac, x2_frac, y2_frac)
HOUSE_ROI = (0.303532, 0.410835, 0.65, 0.559819)


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
        """Process a single voter image."""
        try:
            # Read image
            img = cv2.imdecode(
                np.fromfile(str(img_path), dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE
            )
            
            if img is None:
                return IdCropResult(img_path.name, error="Failed to load image")
            
            H, W = img.shape[:2]
            
            # Extract house number ROI
            x1 = int(W * HOUSE_ROI[0])
            y1 = int(H * HOUSE_ROI[1])
            x2 = int(W * HOUSE_ROI[2])
            y2 = int(H * HOUSE_ROI[3])
            
            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            house_crop = img[y1:y2, x1:x2]
            
            # Validate crop
            if house_crop.size == 0:
                 return IdCropResult(img_path.name, error="Invalid house number ROI dimensions")
            
            # Save
            output_path = output_dir / img_path.name
            success, encoded = cv2.imencode(".png", house_crop)
            if success:
                encoded.tofile(str(output_path))
                return IdCropResult(img_path.name, output_path=output_path)
            else:
                return IdCropResult(img_path.name, error="Failed to encode image")

        except Exception as e:
            return IdCropResult(img_path.name, error=str(e))

