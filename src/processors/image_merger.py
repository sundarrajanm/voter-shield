"""
Image Merger for combining cropped voter images into batches.

Combines cropped voter images into batched merged images (max 10 voters per batch),
with a voter_end.jpg separator between each voter. This allows for faster
OCR processing by loading fewer images while keeping memory usage manageable.

Input: /extracted/<folder>/crops/<page-XXX>/<cropped images>
Output: /extracted/<folder>/merged/<page-XXX>/<batch-001.png, batch-002.png, ...>
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from .base import BaseProcessor, ProcessingContext
from ..config import Config





@dataclass
class MergeSummary:
    """Summary of merge operation."""
    total_pages: int = 0
    total_images_merged: int = 0
    total_batch_files: int = 0
    elapsed_seconds: float = 0.0


class ImageMerger(BaseProcessor):
    """
    Merge cropped voter images into batched page images.
    
    Combines cropped voter images into batches (max 10 per batch) vertically,
    inserting a voter_end.jpg separator between each voter to enable
    batch OCR processing with voter boundary detection.
    """
    
    name = "ImageMerger"
    
    def __init__(
        self,
        context: ProcessingContext,
        voter_end_image: Optional[Path] = None,
        max_voters_per_batch: Optional[int] = None,
    ):
        """
        Initialize image merger.
        
        Args:
            context: Processing context
            voter_end_image: Path to voter_end.jpg separator image
            max_voters_per_batch: Maximum voters per merged batch (default: 10)
        """
        super().__init__(context)
        
        # Set voter_end image path
        if voter_end_image is None:
            self.voter_end_path = self.config.base_dir / "voter_end.jpg"
        else:
            self.voter_end_path = voter_end_image
        
        if max_voters_per_batch is not None:
            self.max_voters_per_batch = max_voters_per_batch
        else:
            self.max_voters_per_batch = self.config.merge.batch_size
        
        # Use standard crops and merged directories
        self.crops_dir = self.context.crops_dir
        self.merged_dir = self.context.extracted_dir / "merged" if self.context.extracted_dir else None
        
        self.summary: Optional[MergeSummary] = None
        
        # Cache voter_end image
        self._voter_end_img: Optional[np.ndarray] = None
    
    def validate(self) -> bool:
        """Validate prerequisites."""
        if not self.merged_dir:
            self.log_error("Merged directory not set (extracted_dir not set)")
            return False
        
        if not self.crops_dir:
            self.log_error("Crops directory not set")
            return False
        
        if not self.crops_dir.exists():
            self.log_error(f"Crops directory not found: {self.crops_dir}")
            return False
        
        if not self.voter_end_path.exists():
            self.log_error(f"voter_end.jpg not found: {self.voter_end_path}")
            return False
        
        return True
    
    def _load_voter_end_image(self) -> np.ndarray:
        """Load and cache voter_end separator image."""
        if self._voter_end_img is None:
            self._voter_end_img = cv2.imdecode(
                np.fromfile(str(self.voter_end_path), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if self._voter_end_img is None:
                raise ValueError(f"Failed to load voter_end image: {self.voter_end_path}")
            self.log_debug(f"Loaded voter_end image: {self._voter_end_img.shape}")
        return self._voter_end_img
    
    def process(self) -> bool:
        """
        Process all cropped images and merge them into batches.
        
        Returns:
            True if processing succeeded
        """
        start_time = time.perf_counter()
        
        page_dirs = self._get_page_dirs(self.crops_dir)
        
        if not page_dirs:
            self.log_warning(f"No page directories found in {self.crops_dir}")
            return False
        
        self.log_info(f"Found {len(page_dirs)} page(s) with crops")
        
        # Load voter_end image
        voter_end_img = self._load_voter_end_image()
        
        total_images = 0
        total_batches = 0
        
        for page_dir in page_dirs:
            page_id = page_dir.name
            
            # Create output directory for this page (same structure as crops)
            page_output_dir = self.merged_dir / page_id
            page_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get images and create batches
            images = self._get_images(page_dir)
            if not images:
                continue
            
            # Split into batches
            batches = self._split_into_batches(images, self.max_voters_per_batch)
            
            for batch_idx, batch_images in enumerate(batches, start=1):
                batch_name = f"batch-{batch_idx:03d}.png"
                merged_path = self._merge_batch(batch_images, voter_end_img, page_output_dir, batch_name)
                if merged_path:
                    total_batches += 1
                    total_images += len(batch_images)
                    self.log_debug(f"Created {page_id}/{batch_name} with {len(batch_images)} voters")
            
            self.log_info(f"Merged {len(images)} images from {page_id} into {len(batches)} batch(es)")
        
        elapsed = time.perf_counter() - start_time
        
        self.summary = MergeSummary(
            total_pages=len(page_dirs),
            total_images_merged=total_images,
            total_batch_files=total_batches,
            elapsed_seconds=elapsed,
        )
        
        self.log_info(
            f"Merge complete: {total_images} images merged into {total_batches} batch files "
            f"in {elapsed:.2f}s"
        )
        
        return True
    
    def _get_page_dirs(self, crops_dir: Path) -> List[Path]:
        """Get sorted page directories."""
        if not crops_dir.exists():
            return []
        return sorted([p for p in crops_dir.iterdir() if p.is_dir()])
    
    def _get_images(self, images_dir: Path) -> List[Path]:
        """Get sorted images from directory."""
        exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
        images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        return sorted(images)
    
    def _split_into_batches(self, images: List[Path], batch_size: int) -> List[List[Path]]:
        """Split list of images into batches of max batch_size."""
        batches = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def _merge_batch(
        self,
        image_paths: List[Path],
        voter_end_img: np.ndarray,
        output_dir: Path,
        batch_name: str,
    ) -> Optional[Path]:
        """
        Merge a batch of images into a single image.
        
        Args:
            image_paths: List of image paths to merge
            voter_end_img: The voter_end separator image
            output_dir: Directory to save merged image
            batch_name: Name of the batch file
        
        Returns:
            Path to merged image if successful
        """
        if not image_paths:
            return None
        
        # Load all images
        loaded_images: List[np.ndarray] = []
        max_width = voter_end_img.shape[1]  # Start with voter_end width
        
        for image_path in image_paths:
            img = cv2.imdecode(
                np.fromfile(str(image_path), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is not None:
                loaded_images.append(img)
                if img.shape[1] > max_width:
                    max_width = img.shape[1]
        
        if not loaded_images:
            self.log_warning(f"No valid images loaded for batch {batch_name}")
            return None
        
        # Resize all images to max_width (pad with white if needed)
        resized_images: List[np.ndarray] = []
        
        for img in loaded_images:
            resized_images.append(self._resize_to_width(img, max_width))
        
        # Resize voter_end to max_width
        voter_end_resized = self._resize_to_width(voter_end_img, max_width)
        
        # Build the merged image: img1 -> voter_end -> img2 -> voter_end -> ...
        merged_parts: List[np.ndarray] = []
        
        for i, img in enumerate(resized_images):
            merged_parts.append(img)
            # Add voter_end after each image (including the last one)
            merged_parts.append(voter_end_resized)
        
        # Stack vertically
        merged_image = np.vstack(merged_parts)
        
        # Save merged image
        output_path = output_dir / batch_name
        
        # Encode and save
        success, encoded = cv2.imencode(".png", merged_image)
        if success:
            encoded.tofile(str(output_path))
            return output_path
        else:
            self.log_error(f"Failed to encode merged batch {batch_name}")
            return None
    
    def _resize_to_width(self, img: np.ndarray, target_width: int) -> np.ndarray:
        """
        Resize image to target width, padding if necessary.
        
        Args:
            img: Input image
            target_width: Target width
        
        Returns:
            Resized/padded image
        """
        h, w = img.shape[:2]
        
        if w == target_width:
            return img
        
        if w < target_width:
            # Pad with white on the right
            pad_width = target_width - w
            padding = np.ones((h, pad_width, 3), dtype=np.uint8) * 255
            return np.hstack([img, padding])
        else:
            # Scale down proportionally
            scale = target_width / w
            new_h = int(h * scale)
            return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
