"""
Image Cropper processor.

Detects and crops voter information boxes from page images
using grid-line detection and morphological operations.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Any

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import BaseProcessor, ProcessingContext
from ..exceptions import CropExtractionError
from ..utils.file_utils import derive_page_id


# Canonical image dimensions for consistent detection
CANON_W, CANON_H = 1187, 1679

# Box detection filters
MIN_BOX_AREA_FRAC = 0.006
MAX_BOX_AREA_FRAC = 0.25
MIN_AR, MAX_AR = 0.55, 2.8

# Crop padding
PAD = 3

# Grid-line extraction parameters
HLINE_SCALE = 25
VLINE_SCALE = 25

# Voter info box detection thresholds
VOTER_INK_FRAC_MIN = 0.008  # Restored - works for all voter pages
VOTER_INK_FRAC_MAX = 0.28   # Restored - works for all voter pages
VOTER_MAX_LINE_RATIO = 0.55  # Slightly relaxed from 0.60 - voter pages have ~0.41-0.50
VOTER_MIN_SMALL_COMPONENTS = 12  # Lowered from 15 - some voters with less text have ~13 components
VOTER_MAX_LARGEST_CC_RATIO = 0.25  # Slightly raised from 0.22 - voter pages have max ~0.20
VOTER_MIN_EDGE_FRAC = 0.07  # Key differentiator: voter=0.08+, non-voter=0.03-0.06
POST_OCR_MIN_EDGE_FRAC = 0.03  # Lower threshold for preprocessed images (upscaling reduces edge density)

# Auto-skip thresholds for non-voter pages
AUTO_SKIP_MAX_SMALL_COMPONENTS = 14
AUTO_SKIP_MIN_LINE_RATIO = 0.70
AUTO_SKIP_MIN_LARGEST_CC_RATIO = 0.35
AUTO_SKIP_MAX_EDGE_FRAC = 0.015

# Language-based page skipping
# Non-voter pages (cover, instructions, etc.) at the beginning of documents
TAMIL_SKIP_PAGES = 3   # Tamil documents: skip first 3 pages
ENGLISH_SKIP_PAGES = 2  # English documents: skip first 2 pages
DEFAULT_SKIP_PAGES = 2  # Fallback if language unknown

# Field cropping ROI definitions (x1, y1, x2, y2) as fractions
# Applied after cropping each voter box to create compact images with only data fields
# Serial No region: top-left area
SERIAL_NO_ROI = (0.184327, 0.065617, 0.337748, 0.207349)
# EPIC region: top-right area containing the EPIC number
EPIC_ROI = (0.662, 0.064, 0.980, 0.194)
# Other fields region: center-left area containing Name, Relation, House No, Age, Gender
OTHER_FIELDS_ROI = (0.028, 0.225, 0.735, 0.909)
# Spacing between stitched regions
STITCH_SPACING = 2
STITCH_BG_COLOR = 255  # White background


@dataclass
class CropMetrics:
    """Metrics for classifying crops as voter/non-voter."""
    ink_frac: float
    line_ratio: float
    small_components: int
    largest_cc_ratio: float
    edge_frac: float


@dataclass
class CroppedBox:
    """Information about a cropped voter box."""
    page_id: str
    box_index: int
    x1: int
    y1: int
    x2: int
    y2: int
    output_path: Optional[Path] = None
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class PageCropResult:
    """Result of cropping a single page."""
    page_id: str
    page_path: Path
    total_boxes_detected: int
    crops_saved: int
    skipped_diagram: int
    skipped_post_ocr: int
    elapsed_seconds: float
    crops: List[CroppedBox] = field(default_factory=list)


@dataclass
class CropSummary:
    """Summary of cropping results for a document."""
    pdf_name: str
    total_pages: int
    processed_pages: int
    skipped_pages: int
    unreadable_pages: int
    total_crops: int
    elapsed_seconds: float


class ImageCropper(BaseProcessor):
    """
    Crop voter information boxes from page images.
    
    Uses grid-line detection and morphological operations to:
    1. Detect table cell boundaries
    2. Filter for voter info boxes (vs diagrams/photos)
    3. Apply OCR preprocessing
    4. Save cropped images
    """
    
    name = "ImageCropper"
    
    def __init__(
        self,
        context: ProcessingContext,
        diagram_filter: str = "auto",
    ):
        """
        Initialize cropper.
        
        Args:
            context: Processing context
            diagram_filter: "auto", "on", or "off"
        """
        super().__init__(context)
        self.diagram_filter = diagram_filter
        self.page_results: List[PageCropResult] = []
        self.summary: Optional[CropSummary] = None
        self.skip_pages = 0  # Will be updated in process()
    
    def _get_skip_pages_count(self) -> int:
        """
        Determine how many initial pages to skip based on document language.
        
        Reads language from metadata JSON file.
        Tamil documents: skip first 3 pages (cover, instructions, diagrams)
        English documents: skip first 2 pages
        
        Returns:
            Number of pages to skip
        """
        if not self.context.output_dir:
            return DEFAULT_SKIP_PAGES
        
        # Find metadata JSON file in output directory
        metadata_files = list(self.context.output_dir.glob("*-metadata.json"))
        if not metadata_files:
            self.log_debug("No metadata file found, using default skip pages")
            return DEFAULT_SKIP_PAGES
        
        try:
            metadata_path = metadata_files[0]
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            languages = metadata.get("language_detected", [])
            
            if not languages:
                self.log_debug("No language detected in metadata, using default")
                return DEFAULT_SKIP_PAGES
            
            # Check if Tamil is the primary language
            primary_language = languages[0].lower() if languages else ""
            
            if "tamil" in primary_language:
                self.log_info(f"Tamil document detected, will skip first {TAMIL_SKIP_PAGES} pages")
                return TAMIL_SKIP_PAGES
            elif "english" in primary_language:
                self.log_info(f"English document detected, will skip first {ENGLISH_SKIP_PAGES} pages")
                return ENGLISH_SKIP_PAGES
            else:
                self.log_debug(f"Unknown language '{primary_language}', using default")
                return DEFAULT_SKIP_PAGES
                
        except Exception as e:
            self.log_warning(f"Error reading metadata for language detection: {e}")
            return DEFAULT_SKIP_PAGES
    
    
    def validate(self) -> bool:
        """Validate prerequisites."""
        if not self.context.images_dir:
            self.log_error("Images directory not set")
            return False
        
        if not self.context.images_dir.exists():
            self.log_error(f"Images directory not found: {self.context.images_dir}")
            return False
        
        return True
    
    def _get_expected_voter_count(self) -> Optional[int]:
        """
        Read expected voter count from metadata JSON file.
        
        Returns:
            Expected total voter count or None if not available
        """
        if not self.context.output_dir:
            return None
        
        # Find metadata JSON file
        metadata_files = list(self.context.output_dir.glob("*-metadata.json"))
        if not metadata_files:
            return None
        
        try:
            metadata_path = metadata_files[0]
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            
            # NEW: Try flat structure first
            total = metadata.get("total")
            
            # Fallback: Try nested structure for backward compatibility
            if total is None:
                detailed_summary = metadata.get("detailed_elector_summary", {})
                net_total = detailed_summary.get("net_total", {})
                total = net_total.get("total")
            
            if total is not None and isinstance(total, int):
                self.log_debug(f"Expected voter count from metadata: {total}")
                return total
            else:
                self.log_debug("No valid total voter count in metadata")
                return None
                
        except Exception as e:
            self.log_warning(f"Error reading expected voter count: {e}")
            return None
    
    def _is_tamil_document(self) -> bool:
        """
        Check if document is Tamil based on metadata.
        
        Returns:
            True if Tamil document, False otherwise
        """
        if not self.context.output_dir:
            return False
        
        metadata_files = list(self.context.output_dir.glob("*-metadata.json"))
        if not metadata_files:
            return False
        
        try:
            metadata_path = metadata_files[0]
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            languages = metadata.get("language_detected", [])
            if languages and isinstance(languages, list):
                primary_language = languages[0].lower() if languages else ""
                return "tamil" in primary_language
                
        except Exception as e:
            self.log_debug(f"Error checking language: {e}")
            return False
        
        return False
    
    def _process_pages_batch(self, page_images: List[Path]) -> int:
        """
        Process a batch of page images in parallel.
        
        Args:
            page_images: List of page image paths to process
            
        Returns:
            Total number of crops saved
        """
        if not page_images:
            return 0
        
        total_crops = 0
        max_workers = min(os.cpu_count() or 4, 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self._process_page, p): p 
                for p in page_images
            }
            
            for future in as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    result = future.result()
                    
                    if result is None:
                        continue
                    
                    self.page_results.append(result)
                    total_crops += result.crops_saved
                        
                except Exception as e:
                    self.log_error(f"Error processing {img_path.name}: {e}")
        
        return total_crops
    
    def _extract_fields(self, img: np.ndarray) -> np.ndarray:
        """
        Apply ROI-based field extraction to create a compact image.
        
        Extracts Serial No, EPIC region and other fields region, then stitches them
        together:
        Top Row: [Serial No] [Spacer] [EPIC]
        Bottom Row: [Other Fields]
        
        Args:
            img: Input grayscale voter crop image
            
        Returns:
            Compact stitched image with only data fields
        """
        H, W = img.shape[:2]

        # Extract Serial No region
        serial_x1 = int(W * SERIAL_NO_ROI[0])
        serial_y1 = int(H * SERIAL_NO_ROI[1])
        serial_x2 = int(W * SERIAL_NO_ROI[2])
        serial_y2 = int(H * SERIAL_NO_ROI[3])
        serial_region = img[serial_y1:serial_y2, serial_x1:serial_x2]
        
        # Extract EPIC region
        epic_x1 = int(W * EPIC_ROI[0])
        epic_y1 = int(H * EPIC_ROI[1])
        epic_x2 = int(W * EPIC_ROI[2])
        epic_y2 = int(H * EPIC_ROI[3])
        epic_region = img[epic_y1:epic_y2, epic_x1:epic_x2]
        
        # Extract other fields region
        fields_x1 = int(W * OTHER_FIELDS_ROI[0])
        fields_y1 = int(H * OTHER_FIELDS_ROI[1])
        fields_x2 = int(W * OTHER_FIELDS_ROI[2])
        fields_y2 = int(H * OTHER_FIELDS_ROI[3])
        fields_region = img[fields_y1:fields_y2, fields_x1:fields_x2]
        
        # Validate regions - fallback to original if extraction fails
        if epic_region.size == 0 or fields_region.size == 0 or serial_region.size == 0:
            return img
        
        # --- Build Top Row (Serial + EPIC) ---
        serial_h, serial_w = serial_region.shape[:2]
        epic_h, epic_w = epic_region.shape[:2]
        max_top_h = max(serial_h, epic_h)
        
        # Pad Serial vertically to match height
        if serial_h < max_top_h:
            top = (max_top_h - serial_h) // 2
            bottom = max_top_h - serial_h - top
            serial_region = np.vstack([
                np.full((top, serial_w), STITCH_BG_COLOR, dtype=np.uint8),
                serial_region,
                np.full((bottom, serial_w), STITCH_BG_COLOR, dtype=np.uint8)
            ])
            
        # Pad EPIC vertically to match height
        if epic_h < max_top_h:
            top = (max_top_h - epic_h) // 2
            bottom = max_top_h - epic_h - top
            epic_region = np.vstack([
                np.full((top, epic_w), STITCH_BG_COLOR, dtype=np.uint8),
                epic_region,
                np.full((bottom, epic_w), STITCH_BG_COLOR, dtype=np.uint8)
            ])
            
        # Horizontal spacer between Serial and EPIC (15px)
        h_spacer = np.full((max_top_h, 15), STITCH_BG_COLOR, dtype=np.uint8)
        
        top_row = np.hstack([serial_region, h_spacer, epic_region])
        
        # --- Stack Top Row with Fields (Vertical) ---
        top_h, top_w = top_row.shape[:2]
        fields_h, fields_w = fields_region.shape[:2]
        max_total_w = max(top_w, fields_w)
        
        # Pad Top Row horizontally (Center it)
        if top_w < max_total_w:
            left = (max_total_w - top_w) // 2
            right = max_total_w - top_w - left
            top_row = np.hstack([
                np.full((top_h, left), STITCH_BG_COLOR, dtype=np.uint8),
                top_row,
                np.full((top_h, right), STITCH_BG_COLOR, dtype=np.uint8)
            ])
            
        # Pad Fields horizontally (Center or Left - let's do Left + Fill to match width)
        if fields_w < max_total_w:
            right = max_total_w - fields_w
            fields_region = np.hstack([
                fields_region,
                np.full((fields_h, right), STITCH_BG_COLOR, dtype=np.uint8)
            ])
        
        # Vertical spacer between Top Row and Fields
        v_spacer = np.full((STITCH_SPACING, max_total_w), STITCH_BG_COLOR, dtype=np.uint8)
        
        # Final Stack
        return np.vstack([top_row, v_spacer, fields_region])
    
    def process(self) -> bool:
        """
        Process all page images in the document with smart validation.
        
        For Tamil documents:
        1. Skip first 3 pages and crop remaining pages
        2. Validate crop count against metadata total
        3. If crops < total, re-process page 3 (the third skipped page)
        4. If still mismatch, terminate with error
        
        Returns:
            True if processing succeeded
        """
        import time
        
        images_dir = self.context.images_dir
        all_page_images = self._list_images(images_dir)
        
        if not all_page_images:
            self.log_warning(f"No images found in {images_dir}")
            return False
        
        self.log_info(f"Found {len(all_page_images)} page image(s)")
        
        # Update skip_pages from metadata (now that MetadataExtractor has likely run)
        self.skip_pages = self._get_skip_pages_count()
        
        # Get expected total from metadata
        expected_total = self._get_expected_voter_count()
        
        # Store skipped pages for potential re-processing
        skipped_pages = []
        page_images = all_page_images
        
        # Skip initial non-voter pages based on language
        if self.skip_pages > 0 and len(all_page_images) > self.skip_pages:
            skipped_pages = all_page_images[:self.skip_pages]
            page_images = all_page_images[self.skip_pages:]
            self.log_info(
                f"Skipping first {self.skip_pages} pages (non-voter pages)",
                skipped=[p.name for p in skipped_pages]
            )
        
        run_start = time.perf_counter()
        
        # First pass: Process with skipped pages
        total_crops = self._process_pages_batch(page_images)
        
        elapsed_first = time.perf_counter() - run_start
        
        # Validate crop count for Tamil documents
        is_tamil = self._is_tamil_document()
        if is_tamil and expected_total is not None and expected_total > 0:
            self.log_info(
                f"Validation check",
                expected=expected_total,
                cropped=total_crops,
                match=(total_crops == expected_total)
            )
            
            # If crops < expected and we skipped 3 pages (page 3 exists)
            if total_crops < expected_total and len(skipped_pages) >= 3:
                page_3 = skipped_pages[2]  # 0-indexed, so page 3 is index 2
                self.log_warning(
                    f"Crop count mismatch: {total_crops} < {expected_total}. "
                    f"Re-processing page 3: {page_3.name}"
                )
                
                # Process page 3
                retry_start = time.perf_counter()
                additional_crops = self._process_pages_batch([page_3])
                total_crops += additional_crops
                elapsed_retry = time.perf_counter() - retry_start
                
                self.log_info(
                    f"Page 3 re-processed",
                    additional_crops=additional_crops,
                    new_total=total_crops,
                    time=f"{elapsed_retry:.2f}s"
                )
                
                # Final validation
                if total_crops != expected_total:
                    self.log_error(
                        f"CRITICAL: Crop count still mismatched after retry. "
                        f"Expected: {expected_total}, Got: {total_crops}. "
                        f"Terminating processing for this document."
                    )
                    return False
                else:
                    self.log_info(
                        f"✓ Crop count validated successfully after page 3 re-processing: {total_crops}"
                    )
        
        elapsed = time.perf_counter() - run_start
        
        # Calculate summary stats
        processed = len(self.page_results)
        skipped = sum(1 for r in self.page_results if r.crops_saved == 0)
        unreadable = len(page_images) - len(self.page_results)
        
        self.summary = CropSummary(
            pdf_name=self.context.pdf_name,
            total_pages=len(page_images),
            processed_pages=processed,
            skipped_pages=skipped,
            unreadable_pages=unreadable,
            total_crops=total_crops,
            elapsed_seconds=elapsed,
        )
        
        self.log_info(
            f"Cropping complete",
            pages=processed,
            crops=total_crops,
            skipped=skipped,
            time=f"{elapsed:.2f}s"
        )
        
        return True
    
    def _list_images(self, input_dir: Path) -> List[Path]:
        """List all image files in directory."""
        exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
        images = [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]
        return sorted(images, key=lambda p: p.name.lower())
    
    def _process_page(self, img_path: Path) -> Optional[PageCropResult]:
        """
        Process a single page image using two-step approach:
        Step 1: Detect and classify boxes, save full OCR-preprocessed crops
        Step 2: Apply field extraction to saved crops
        
        This separation ensures box detection/classification logic is not
        affected by ROI-based field extraction.
        
        Args:
            img_path: Path to page image
        
        Returns:
            PageCropResult or None if unreadable
        """
        import time
        
        page_start = time.perf_counter()
        page_id = derive_page_id(img_path)
        
        # Read image
        img_orig = cv2.imdecode(
            np.fromfile(str(img_path), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if img_orig is None:
            self.log_warning(f"Unreadable image: {img_path.name}")
            return None
        
        H0, W0 = img_orig.shape[:2]
        
        # Resize to canonical dimensions for detection
        img_canon = cv2.resize(img_orig, (CANON_W, CANON_H), interpolation=cv2.INTER_AREA)
        
        # Detect boxes
        boxes_canon = self._detect_boxes(img_canon)
        
        self.log_debug(
            f"Page {page_id}",
            size=f"{W0}x{H0}",
            boxes=len(boxes_canon)
        )
        
        if not boxes_canon:
            elapsed = time.perf_counter() - page_start
            return PageCropResult(
                page_id=page_id,
                page_path=img_path,
                total_boxes_detected=0,
                crops_saved=0,
                skipped_diagram=0,
                skipped_post_ocr=0,
                elapsed_seconds=elapsed,
            )
        
        # Scale boxes back to original dimensions
        sx = W0 / float(CANON_W)
        sy = H0 / float(CANON_H)
        boxes_orig = self._scale_boxes(boxes_canon, sx, sy)
        
        # Classify boxes
        voter_candidates, skipped_diagram, rejected_metrics = self._classify_boxes(
            img_orig, boxes_orig, W0, H0, page_id
        )
        
        self.log_debug(
            f"Page {page_id}: detected={len(boxes_orig)}, accepted={len(voter_candidates)}, rejected={skipped_diagram}"
        )
        
        # Auto mode: fail-open if uncertain
        if self.diagram_filter == "auto" and not voter_candidates and boxes_orig:
            if not self._is_confidently_non_voter(rejected_metrics):
                voter_candidates = self._all_boxes_as_candidates(boxes_orig, W0, H0)
                skipped_diagram = 0
                self.log_debug(
                    f"Auto mode: saving all {len(voter_candidates)} boxes (uncertain rejection)"
                )
        
        if not voter_candidates:
            elapsed = time.perf_counter() - page_start
            if skipped_diagram:
                self.log_debug(f"Skipped page {page_id}: all {skipped_diagram} boxes diagram-like")
            return PageCropResult(
                page_id=page_id,
                page_path=img_path,
                total_boxes_detected=len(boxes_orig),
                crops_saved=0,
                skipped_diagram=skipped_diagram,
                skipped_post_ocr=0,
                elapsed_seconds=elapsed,
            )
        
        # ===== STEP 1: Detect, classify, and save FULL crops (without field extraction) =====
        crops_dir = self.context.crops_dir / page_id
        crops_saved = 0
        skipped_post_ocr = 0
        saved_crops: List[CroppedBox] = []
        saved_paths: List[Path] = []  # Track paths for step 2
        
        for box in voter_candidates:
            crop = img_orig[box.y1:box.y2, box.x1:box.x2]
            ocr_img = self._ocr_preprocess(crop)
            
            # Post-OCR edge check (classification step)
            if self.diagram_filter != "off":
                edges = cv2.Canny(ocr_img, 50, 150)
                edge_frac = float(np.count_nonzero(edges)) / float(edges.size)
                if edge_frac < POST_OCR_MIN_EDGE_FRAC:
                    skipped_post_ocr += 1
                    continue
            
            # Create directory on first save
            if not crops_dir.exists():
                crops_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FULL OCR-preprocessed crop (without field extraction yet)
            crop_name = f"{page_id}-{box.box_index:03d}.png"
            out_path = crops_dir / crop_name
            
            cv2.imwrite(str(out_path), ocr_img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            
            box.output_path = out_path
            saved_crops.append(box)
            saved_paths.append(out_path)
            crops_saved += 1
        
        # ===== STEP 2: Apply field extraction to saved crops =====
        for out_path in saved_paths:
            self._apply_field_extraction(out_path)
        
        elapsed = time.perf_counter() - page_start
        
        self.log_debug(
            f"Page {page_id}: saved {crops_saved} crops",
            skipped_diagram=skipped_diagram,
            skipped_post_ocr=skipped_post_ocr,
            time=f"{elapsed:.2f}s"
        )
        
        return PageCropResult(
            page_id=page_id,
            page_path=img_path,
            total_boxes_detected=len(boxes_orig),
            crops_saved=crops_saved,
            skipped_diagram=skipped_diagram,
            skipped_post_ocr=skipped_post_ocr,
            elapsed_seconds=elapsed,
            crops=saved_crops,
        )
    
    def _apply_field_extraction(self, img_path: Path) -> None:
        """
        Apply field extraction to a saved crop image (Step 2).
        
        Reads the saved full crop, applies ROI-based field extraction,
        and overwrites the file with the compact version.
        
        Args:
            img_path: Path to the saved crop image
        """
        # Read the saved crop
        img = cv2.imdecode(
            np.fromfile(str(img_path), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
        
        if img is None:
            self.log_warning(f"Could not read crop for field extraction: {img_path.name}")
            return
        
        # Apply field extraction to create compact image
        compact_img = self._extract_fields(img)
        
        # Overwrite the file with the compact version
        cv2.imwrite(str(img_path), compact_img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    
    def _detect_boxes(self, img_canon: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect voter boxes using grid-line detection.
        
        Args:
            img_canon: Canonical-sized image
        
        Returns:
            List of (x, y, w, h) boxes
        """
        H, W = img_canon.shape[:2]
        page_area = W * H
        
        gray = cv2.cvtColor(img_canon, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31, 12
        )
        
        # Extract horizontal lines
        h_kernel_len = max(10, W // HLINE_SCALE)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        horiz = cv2.erode(bw, h_kernel, iterations=1)
        horiz = cv2.dilate(horiz, h_kernel, iterations=1)
        
        # Extract vertical lines
        v_kernel_len = max(10, H // VLINE_SCALE)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        vert = cv2.erode(bw, v_kernel, iterations=1)
        vert = cv2.dilate(vert, v_kernel, iterations=1)
        
        # Combine and close gaps
        grid = cv2.bitwise_or(horiz, vert)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, k, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Filter by area
            if area < page_area * MIN_BOX_AREA_FRAC:
                continue
            if area > page_area * MAX_BOX_AREA_FRAC:
                continue
            
            # Filter by aspect ratio
            ar = w / float(h)
            if not (MIN_AR <= ar <= MAX_AR):
                continue
            
            # Filter by polygon complexity
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) < 4:
                continue
            
            boxes.append((x, y, w, h))
        
        # Post-process
        boxes = self._dedupe_boxes(boxes)
        boxes = self._remove_contained_boxes(boxes)
        boxes = self._sort_reading_order(boxes)
        
        return boxes
    
    def _scale_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        sx: float,
        sy: float
    ) -> List[Tuple[int, int, int, int]]:
        """Scale boxes from canonical to original dimensions."""
        return [
            (int(round(x * sx)), int(round(y * sy)),
             int(round(w * sx)), int(round(h * sy)))
            for x, y, w, h in boxes
        ]
    
    def _dedupe_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        tol: int = 6
    ) -> List[Tuple[int, int, int, int]]:
        """Remove near-duplicate boxes."""
        boxes = sorted(boxes, key=lambda b: (b[0], b[1], b[2], b[3]))
        out = []
        for x, y, w, h in boxes:
            dup = False
            for x2, y2, w2, h2 in out:
                if (abs(x - x2) < tol and abs(y - y2) < tol and
                    abs(w - w2) < tol and abs(h - h2) < tol):
                    dup = True
                    break
            if not dup:
                out.append((x, y, w, h))
        return out
    
    def _remove_contained_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        margin: int = 6,
        max_area_ratio: float = 0.70
    ) -> List[Tuple[int, int, int, int]]:
        """Remove boxes fully contained within another."""
        if not boxes:
            return boxes
        
        boxes_sorted = sorted(boxes, key=lambda b: (b[2] * b[3]), reverse=True)
        kept = []
        
        for x, y, w, h in boxes_sorted:
            x2, y2 = x + w, y + h
            area = w * h
            contained = False
            
            for X, Y, W, H in kept:
                X2, Y2 = X + W, Y + H
                if (x >= X + margin and y >= Y + margin and
                    x2 <= X2 - margin and y2 <= Y2 - margin):
                    big_area = W * H
                    if area / float(max(1, big_area)) <= max_area_ratio:
                        contained = True
                        break
            
            if not contained:
                kept.append((x, y, w, h))
        
        return sorted(kept, key=lambda b: (b[1], b[0]))
    
    def _sort_reading_order(
        self,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Sort boxes in reading order (top-to-bottom, left-to-right)."""
        if not boxes:
            return boxes
        
        heights = sorted(h for _, _, _, h in boxes)
        median_h = heights[len(heights) // 2]
        row_thresh = max(10, int(median_h * 0.55))
        
        boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
        rows, cur = [], [boxes_sorted[0]]
        
        for b in boxes_sorted[1:]:
            if abs(b[1] - cur[-1][1]) <= row_thresh:
                cur.append(b)
            else:
                rows.append(sorted(cur, key=lambda x: x[0]))
                cur = [b]
        rows.append(sorted(cur, key=lambda x: x[0]))
        
        out = []
        for r in rows:
            out.extend(r)
        return out
    
    def _classify_boxes(
        self,
        img_orig: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        W: int,
        H: int,
        page_id: str = ""
    ) -> Tuple[List[CroppedBox], int, List[CropMetrics]]:
        """
        Classify boxes as voter/non-voter.
        
        Returns:
            Tuple of (voter candidates, skipped count, rejected metrics)
        """
        voter_candidates = []
        skipped = 0
        rejected_metrics = []
        
        for i, (x, y, w, h) in enumerate(boxes, start=1):
            x1 = max(0, x - PAD)
            y1 = max(0, y - PAD)
            x2 = min(W, x + w + PAD)
            y2 = min(H, y + h + PAD)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            box = CroppedBox(
                page_id="",  # Set later
                box_index=i,
                x1=x1, y1=y1, x2=x2, y2=y2
            )
            
            if self.diagram_filter == "off":
                voter_candidates.append(box)
                continue
            
            crop = img_orig[y1:y2, x1:x2]
            is_voter, metrics = self._classify_crop_metrics(crop)
            
            if is_voter:
                voter_candidates.append(box)
            else:
                skipped += 1
                rejected_metrics.append(metrics)
                # Debug log rejection reasons
                reasons = getattr(metrics, '_rejection_reasons', [])
                if reasons:
                    self.log_debug(f"{page_id} box {i} rejected: {'; '.join(reasons)}")
        
        return voter_candidates, skipped, rejected_metrics
    
    def _classify_crop_metrics(self, crop_bgr: np.ndarray) -> Tuple[bool, CropMetrics]:
        """
        Classify a crop as voter info or diagram/photo.
        
        Returns:
            Tuple of (is_voter, metrics)
        """
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_frac = float(np.count_nonzero(edges)) / float(edges.size)
        
        # Binarize
        bw = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31, 12
        )
        
        ink_px = int(np.count_nonzero(bw))
        ink_frac = float(ink_px) / float(bw.size)
        
        # Extract lines
        lines = self._extract_hv_lines(bw)
        line_px = int(np.count_nonzero(lines))
        line_ratio = float(line_px) / float(max(1, ink_px))
        
        # Remove lines and count small components
        bw_wo_lines = cv2.bitwise_and(bw, cv2.bitwise_not(lines))
        
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bw_wo_lines = cv2.morphologyEx(bw_wo_lines, cv2.MORPH_OPEN, k2, iterations=1)
        
        nlabels, _, stats, _ = cv2.connectedComponentsWithStats(bw_wo_lines, connectivity=8)
        
        area_total = bw_wo_lines.shape[0] * bw_wo_lines.shape[1]
        max_small_area = max(80, int(area_total * 0.002))
        min_small_area = 10
        small_components = 0
        largest_cc_area = 0
        
        for i in range(1, nlabels):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a > largest_cc_area:
                largest_cc_area = a
            if min_small_area <= a <= max_small_area:
                small_components += 1
        
        ink_wo_lines_px = int(np.count_nonzero(bw_wo_lines))
        largest_cc_ratio = float(largest_cc_area) / float(max(1, ink_wo_lines_px))
        
        metrics = CropMetrics(
            ink_frac=ink_frac,
            line_ratio=line_ratio,
            small_components=small_components,
            largest_cc_ratio=largest_cc_ratio,
            edge_frac=edge_frac,
        )
        
        # Classification logic - track rejection reasons
        is_voter = True
        rejection_reasons = []
        
        if not (VOTER_INK_FRAC_MIN <= ink_frac <= VOTER_INK_FRAC_MAX):
            is_voter = False
            rejection_reasons.append(f"ink_frac={ink_frac:.4f} not in [{VOTER_INK_FRAC_MIN}, {VOTER_INK_FRAC_MAX}]")
        if line_ratio > VOTER_MAX_LINE_RATIO:
            is_voter = False
            rejection_reasons.append(f"line_ratio={line_ratio:.4f} > {VOTER_MAX_LINE_RATIO}")
        if small_components < VOTER_MIN_SMALL_COMPONENTS:
            is_voter = False
            rejection_reasons.append(f"small_components={small_components} < {VOTER_MIN_SMALL_COMPONENTS}")
        if largest_cc_ratio > VOTER_MAX_LARGEST_CC_RATIO:
            is_voter = False
            rejection_reasons.append(f"largest_cc_ratio={largest_cc_ratio:.4f} > {VOTER_MAX_LARGEST_CC_RATIO}")
        if edge_frac < VOTER_MIN_EDGE_FRAC:
            is_voter = False
            rejection_reasons.append(f"edge_frac={edge_frac:.4f} < {VOTER_MIN_EDGE_FRAC}")
        
        # Store rejection reasons for debugging
        if rejection_reasons:
            metrics._rejection_reasons = rejection_reasons
        
        return is_voter, metrics
    
    def _extract_hv_lines(self, bw_inv: np.ndarray) -> np.ndarray:
        """Extract horizontal/vertical lines from binary-inverted image."""
        h, w = bw_inv.shape[:2]
        
        h_kernel_len = max(40, w // 6)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        horiz = cv2.erode(bw_inv, h_kernel, iterations=1)
        horiz = cv2.dilate(horiz, h_kernel, iterations=1)
        
        v_kernel_len = max(40, h // 6)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        vert = cv2.erode(bw_inv, v_kernel, iterations=1)
        vert = cv2.dilate(vert, v_kernel, iterations=1)
        
        lines = cv2.bitwise_or(horiz, vert)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, k, iterations=1)
        
        return lines
    
    def _is_confidently_non_voter(self, rejected_metrics: List[CropMetrics]) -> bool:
        """Check if rejected boxes are confidently non-voter."""
        if not rejected_metrics:
            return False
        
        avg_small = float(np.mean([m.small_components for m in rejected_metrics]))
        avg_line_ratio = float(np.mean([m.line_ratio for m in rejected_metrics]))
        avg_largest_cc = float(np.mean([m.largest_cc_ratio for m in rejected_metrics]))
        avg_edge = float(np.mean([m.edge_frac for m in rejected_metrics]))
        
        return (
            avg_small <= AUTO_SKIP_MAX_SMALL_COMPONENTS or
            avg_line_ratio >= AUTO_SKIP_MIN_LINE_RATIO or
            avg_largest_cc >= AUTO_SKIP_MIN_LARGEST_CC_RATIO or
            avg_edge <= AUTO_SKIP_MAX_EDGE_FRAC
        )
    
    def _all_boxes_as_candidates(
        self,
        boxes: List[Tuple[int, int, int, int]],
        W: int,
        H: int
    ) -> List[CroppedBox]:
        """Convert all boxes to candidates (for auto mode fail-open)."""
        candidates = []
        for i, (x, y, w, h) in enumerate(boxes, start=1):
            x1 = max(0, x - PAD)
            y1 = max(0, y - PAD)
            x2 = min(W, x + w + PAD)
            y2 = min(H, y + h + PAD)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            candidates.append(CroppedBox(
                page_id="",
                box_index=i,
                x1=x1, y1=y1, x2=x2, y2=y2
            ))
        
        return candidates
    
    def _ocr_preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess crop for OCR.
        
        Applies:
        - Grayscale conversion
        - Deskewing
        - Upscaling (2x)
        - Denoising
        - Contrast normalization
        """
        # Grayscale
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        
        # Deskew
        angle = self._estimate_skew(gray)
        if abs(angle) > 0.2:
            gray = self._rotate_image(gray, angle)
        
        # Upscale (Linear is faster than Cubic)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        
        # Denoise (Median blur is much faster than NlMeans)
        gray = cv2.medianBlur(gray, 3)
        
        # Contrast normalization
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        return gray
    
    def _estimate_skew(self, gray: np.ndarray) -> float:
        """Estimate skew angle using Hough lines."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=80,
            minLineLength=max(50, gray.shape[1] // 3),
            maxLineGap=10
        )
        
        if lines is None:
            return 0.0
        
        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                continue
            ang = np.degrees(np.arctan2(dy, dx))
            if -30 <= ang <= 30:
                angles.append(ang)
        
        if not angles:
            return 0.0
        
        return float(np.median(angles))
    
    def _rotate_image(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate image by angle."""
        if abs(angle_deg) < 0.2:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )


def crop_document(
    extracted_dir: Path,
    diagram_filter: str = "auto",
) -> CropSummary:
    """
    Convenience function to crop all pages in an extracted folder.
    
    Args:
        extracted_dir: Path to extracted folder
        diagram_filter: "auto", "on", or "off"
    
    Returns:
        Crop summary
    """
    from ..config import Config
    
    config = Config()
    context = ProcessingContext(config=config)
    context.setup_paths_from_extracted(extracted_dir)
    
    cropper = ImageCropper(context, diagram_filter=diagram_filter)
    
    if not cropper.run():
        raise CropExtractionError("Cropping failed", str(extracted_dir))
    
    return cropper.summary
