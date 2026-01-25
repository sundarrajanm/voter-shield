"""
Crop Top Merger for combining crop-top images into batches.

Combines crop-top (header) images into batched merged images,
with a filename label added before each image. This allows for
batch processing of page header information.

Input: /extracted/<folder>/crop-top/<page-XXX.png>
Output: /extracted/<folder>/crop-top-merged/<batch-001.png, batch-002.png, ...>
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .base import BaseProcessor, ProcessingContext
from ..config import Config


# Default font settings for filename labels
LABEL_FONT_SIZE = 24
LABEL_PADDING = 10
LABEL_BG_COLOR = (255, 255, 255)  # White background
LABEL_TEXT_COLOR = (0, 0, 0)  # Black text


@dataclass
class TopMergeSummary:
    """Summary of crop-top merge operation."""
    total_images: int = 0
    total_batch_files: int = 0
    elapsed_seconds: float = 0.0


class CropTopMerger(BaseProcessor):
    """
    Merge crop-top (header) images into batched page images with filenames.
    
    Combines crop-top images into batches (configurable batch size) vertically,
    inserting a filename label before each image to identify the source page.
    """
    
    name = "CropTopMerger"
    
    def __init__(
        self,
        context: ProcessingContext,
        max_images_per_batch: Optional[int] = None,
    ):
        """
        Initialize crop-top merger.
        
        Args:
            context: Processing context
            max_images_per_batch: Maximum images per merged batch (default from config)
        """
        super().__init__(context)
        
        if max_images_per_batch is not None:
            self.max_images_per_batch = max_images_per_batch
        else:
            self.max_images_per_batch = self.config.top_merge.batch_size
        
        # Input and output directories
        self.crop_top_dir = self.context.crop_top_dir
        self.merged_dir = self.context.extracted_dir / "crop-top-merged" if self.context.extracted_dir else None
        
        self.summary: Optional[TopMergeSummary] = None
        
        # Font for labels (will be loaded on first use)
        self._font: Optional[ImageFont.FreeTypeFont] = None
    
    def validate(self) -> bool:
        """Validate prerequisites."""
        if not self.merged_dir:
            self.log_error("Merged directory not set (extracted_dir not set)")
            return False
        
        if not self.crop_top_dir:
            self.log_error("Crop-top directory not set")
            return False
        
        if not self.crop_top_dir.exists():
            self.log_error(f"Crop-top directory not found: {self.crop_top_dir}")
            return False
        
        return True
    
    def _get_font(self) -> ImageFont.FreeTypeFont:
        """Get or load font for labels."""
        if self._font is None:
            try:
                # Try to load a system font
                self._font = ImageFont.truetype("arial.ttf", LABEL_FONT_SIZE)
            except (IOError, OSError):
                try:
                    # Fallback for Linux
                    self._font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", LABEL_FONT_SIZE)
                except (IOError, OSError):
                    # Use default font
                    self._font = ImageFont.load_default()
        return self._font
    
    def _create_filename_label(self, filename: str, target_width: int) -> np.ndarray:
        """
        Create an image with the filename as a label.
        
        Args:
            filename: The filename to display
            target_width: Width to match the target images
            
        Returns:
            Label image as numpy array (BGR)
        """
        font = self._get_font()
        
        # Create a PIL image for the label
        # Calculate text size
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), filename, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create label image with padding
        label_height = text_height + 2 * LABEL_PADDING
        label_img = Image.new("RGB", (target_width, label_height), color=LABEL_BG_COLOR)
        draw = ImageDraw.Draw(label_img)
        
        # Center the text horizontally
        x_pos = (target_width - text_width) // 2
        y_pos = LABEL_PADDING
        
        draw.text((x_pos, y_pos), filename, font=font, fill=LABEL_TEXT_COLOR)
        
        # Convert to numpy (BGR for OpenCV)
        label_np = np.array(label_img)
        label_bgr = cv2.cvtColor(label_np, cv2.COLOR_RGB2BGR)
        
        return label_bgr
    
    def process(self) -> bool:
        """
        Process all crop-top images and merge them into batches.
        
        Returns:
            True if processing succeeded
        """
        start_time = time.perf_counter()
        
        # Get all crop-top images
        images = self._get_images(self.crop_top_dir)
        
        if not images:
            self.log_warning(f"No images found in {self.crop_top_dir}")
            return False
        
        self.log_info(f"Found {len(images)} crop-top image(s)")
        
        # Create output directory
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into batches
        batches = self._split_into_batches(images, self.max_images_per_batch)
        
        total_batches = 0
        total_images = 0
        
        for batch_idx, batch_images in enumerate(batches, start=1):
            batch_name = f"batch-{batch_idx:03d}.png"
            merged_path = self._merge_batch(batch_images, self.merged_dir, batch_name)
            if merged_path:
                total_batches += 1
                total_images += len(batch_images)
                self.log_debug(f"Created {batch_name} with {len(batch_images)} images")
        
        elapsed = time.perf_counter() - start_time
        
        self.summary = TopMergeSummary(
            total_images=total_images,
            total_batch_files=total_batches,
            elapsed_seconds=elapsed,
        )
        
        self.log_info(
            f"Crop-top merge complete: {total_images} images merged into {total_batches} batch files "
            f"in {elapsed:.2f}s"
        )
        
        return True
    
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
        output_dir: Path,
        batch_name: str,
    ) -> Optional[Path]:
        """
        Merge a batch of images into a single image with filename labels.
        
        Args:
            image_paths: List of image paths to merge
            output_dir: Directory to save merged image
            batch_name: Name of the batch file
        
        Returns:
            Path to merged image if successful
        """
        if not image_paths:
            return None
        
        # Load all images and find max width
        loaded_images: List[tuple] = []  # List of (filename, image)
        max_width = 0
        
        for image_path in image_paths:
            img = cv2.imdecode(
                np.fromfile(str(image_path), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is not None:
                loaded_images.append((image_path.name, img))
                if img.shape[1] > max_width:
                    max_width = img.shape[1]
        
        if not loaded_images:
            self.log_warning(f"No valid images loaded for batch {batch_name}")
            return None
        
        # Ensure minimum width for labels
        max_width = max(max_width, 300)
        
        # Build the merged image: label -> img -> label -> img -> ...
        merged_parts: List[np.ndarray] = []
        
        for filename, img in loaded_images:
            # Create and add filename label
            label_img = self._create_filename_label(filename, max_width)
            merged_parts.append(label_img)
            
            # Resize image to max_width if needed
            resized_img = self._resize_to_width(img, max_width)
            merged_parts.append(resized_img)
        
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
