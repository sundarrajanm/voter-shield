"""
Header Extractor processor.

Extracts page header information from electoral roll pages:
- Assembly Constituency Number and Name
- Section Number and Name  
- Part Number

Supports both English and Tamil text extraction.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np

from .base import BaseProcessor, ProcessingContext
from ..exceptions import ProcessingError
from ..utils.file_utils import derive_page_id

# Tesseract for OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Canonical image dimensions (same as image_cropper)
CANON_W, CANON_H = 1187, 1679

# Header region: fixed 84 pixels from top of the page
# This captures the Assembly Constituency, Section, and Part number info
HEADER_HEIGHT_PIXELS = 82


def _configure_tesseract() -> bool:
    """Configure tesseract path for Windows if needed."""
    import os
    if os.name == "nt":
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(tesseract_path).exists():
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            return True
    return True  # On non-Windows, assume it's in PATH


@dataclass
class PageHeaderInfo:
    """Extracted header information from a page."""
    page_id: str
    assembly_constituency_number_and_name: str = ""
    street_name_and_number: str = ""
    part_number: Optional[int] = None
    header_image_path: Optional[Path] = None
    raw_text: str = ""


@dataclass
class HeaderExtractionResult:
    """Result of header extraction for a document."""
    pdf_name: str
    total_pages: int
    processed_pages: int
    elapsed_seconds: float
    page_headers: Dict[str, PageHeaderInfo] = field(default_factory=dict)


class HeaderExtractor(BaseProcessor):
    """
    Extract header information from electoral roll pages.
    
    Crops the top section of each page and uses OCR to extract:
    - Assembly Constituency Number and Name
    - Section Number and Name
    - Part Number
    """
    
    name = "HeaderExtractor"
    
    def __init__(
        self,
        context: ProcessingContext,
        languages: str = "eng+tam",
    ):
        """
        Initialize header extractor.
        
        Args:
            context: Processing context
            languages: OCR languages (default: eng+tam for English and Tamil)
        """
        super().__init__(context)
        self.languages = languages
        self.page_headers: Dict[str, PageHeaderInfo] = {}
        self.result: Optional[HeaderExtractionResult] = None
    
    def validate(self) -> bool:
        """Validate prerequisites."""
        if not TESSERACT_AVAILABLE:
            self.log_error("pytesseract not available")
            return False
        
        # Configure tesseract path for Windows
        _configure_tesseract()
        
        # Verify tesseract is working
        try:
            version = pytesseract.get_tesseract_version()
            self.log_debug(f"Tesseract version: {version}")
        except Exception as e:
            self.log_error(f"Tesseract not available: {e}")
            return False
        
        if not self.context.images_dir:
            self.log_error("Images directory not set")
            return False
        
        if not self.context.images_dir.exists():
            self.log_error(f"Images directory not found: {self.context.images_dir}")
            return False
        
        return True
    
    def process(self) -> bool:
        """
        Process all page images to extract headers.
        
        Returns:
            True if processing succeeded
        """
        images_dir = self.context.images_dir
        page_images = self._list_images(images_dir)
        
        if not page_images:
            self.log_warning(f"No images found in {images_dir}")
            return False
        
        self.log_info(f"Extracting headers from {len(page_images)} page(s)")
        
        total_pages = len(page_images)
        processed = 0
        
        run_start = time.perf_counter()
        
        # Ensure crop-top directory exists
        crop_top_dir = self.context.crop_top_dir
        if crop_top_dir:
            crop_top_dir.mkdir(parents=True, exist_ok=True)
        
        # Process pages sequentially to avoid Tesseract memory issues with parallel execution
        for img_path in page_images:
            try:
                header_info = self._process_page(img_path)
                if header_info:
                    self.page_headers[header_info.page_id] = header_info
                    processed += 1
            except Exception as e:
                self.log_error(f"Error processing header for {img_path.name}: {e}")
        
        elapsed = time.perf_counter() - run_start
        
        self.result = HeaderExtractionResult(
            pdf_name=self.context.pdf_name or "",
            total_pages=total_pages,
            processed_pages=processed,
            elapsed_seconds=elapsed,
            page_headers=self.page_headers,
        )
        
        self.log_info(
            f"Header extraction complete",
            pages=processed,
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
    
    def _process_page(self, img_path: Path) -> Optional[PageHeaderInfo]:
        """
        Process a single page image to extract header.
        
        Args:
            img_path: Path to page image
        
        Returns:
            PageHeaderInfo or None if extraction failed
        """
        page_id = derive_page_id(img_path)
        
        # Read image
        img_orig = cv2.imdecode(
            np.fromfile(str(img_path), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if img_orig is None:
            self.log_warning(f"Unreadable image: {img_path.name}")
            return None
        
        H, W = img_orig.shape[:2]
        
        # Crop header region (fixed 82 pixels from top)
        header_height = min(HEADER_HEIGHT_PIXELS, H)
        header_crop = img_orig[0:header_height, :]
        
        # Save header crop
        header_path = None
        if self.context.crop_top_dir:
            header_path = self.context.crop_top_dir / f"{page_id}.png"
            cv2.imwrite(str(header_path), header_crop)
        
        # Preprocess for OCR
        header_processed = self._preprocess_for_ocr(header_crop)
        
        # Run OCR
        raw_text = self._run_ocr(header_processed)
        
        # Parse the text to extract fields
        assembly_info, section_info, part_num = self._parse_header_text(raw_text)
        
        return PageHeaderInfo(
            page_id=page_id,
            assembly_constituency_number_and_name=assembly_info,
            street_name_and_number=section_info,
            part_number=part_num,
            header_image_path=header_path,
            raw_text=raw_text,
        )
    
    def _preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Upscale for better OCR
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        
        # Denoise
        gray = cv2.medianBlur(gray, 3)
        
        # Contrast normalization
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        return gray
    
    def _run_ocr(self, img: np.ndarray) -> str:
        """Run OCR on preprocessed image."""
        from PIL import Image
        
        try:
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(img)
            
            # Run Tesseract with English and Tamil
            # Use --oem 3 (default), --psm 6 (assume uniform block of text)
            config = "--oem 3 --psm 6"
            text = pytesseract.image_to_string(pil_img, lang=self.languages, config=config)
            return text.strip()
        except Exception as e:
            self.log_error(f"OCR failed: {e}")
            return ""
    
    def _parse_header_text(self, text: str) -> Tuple[str, str, Optional[int]]:
        """
        Parse OCR text to extract assembly, section, and part info.
        
        Supports both English and Tamil patterns.
        
        Returns:
            Tuple of (assembly_constituency, section, part_number)
        """
        assembly_info = ""
        section_info = ""
        part_number = None
        
        if not text:
            return assembly_info, section_info, part_number
        
        # Keep original lines for section parsing (to avoid artifacts from line joining)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        # Filter out lines that are just garbage (only dashes, special chars, etc.)
        clean_lines = [l for l in lines if not re.match(r'^[—_\-\s\|CL LT]+$', l, re.IGNORECASE)]
        full_text = ' '.join(clean_lines)
        
        # --- ASSEMBLY CONSTITUENCY PATTERNS ---
        
        # Note: OCR may produce ZWNJ (\u200c) after Tamil letters - handle with \u200c? optional matching
        
        # English: "Assembly Constituency No and Name : 116-SULUR"
        # Format: "Assembly Constituency No and Name: 116-SULUR"
        assembly_patterns_en = [
            r"Assembly\s*Constituency\s*(?:No\.?|Number)?\s*(?:and|&)?\s*Name\s*[:\s]*[:\s]*(\d+[-\s]*[A-Za-z\s\(\)]+?)(?=\s*(?:Section|Part|$))",
            r"Constituency\s*(?:No\.?|Number)?\s*(?:and|&)?\s*Name\s*[:\s]*(\d+[-\s]*[A-Za-z\s\(\)]+?)(?=\s*(?:Section|Part|$))",
            r"(\d{2,3})[-\s]*([A-Z][A-Za-z]+(?:\s*\([^)]+\))?)",  # Fallback: match pattern like "116-SULUR"
        ]
        
        # Tamil: "சட்டமன்றத் தொகுதியின் எண் மற்றும் பெயர் : 114-திருப்பூர் (தெற்கு)"
        # OCR often produces ZWNJ (\u200c) after Tamil letters, so we use [\u200c]? to handle it
        assembly_patterns_ta = [
            # Full pattern with ZWNJ handling
            r"சட்டமன்றத்[\u200c]?\s*தொகுதியின்[\u200c]?\s*எண்[\u200c]?\s*(?:மற்றும்[\u200c]?)?\s*பெயர்[\u200c]?\s*[:\s]*(.+?)(?=பிரிவு|பாகம்[\u200c]?\s*எண்|Section|Part|$)",
            r"தொகுதியின்[\u200c]?\s*எண்[\u200c]?\s*(?:மற்றும்[\u200c]?)?\s*பெயர்[\u200c]?\s*[:\s]*(.+?)(?=பிரிவு|பாகம்|$)",
            # Fallback: match Tamil constituency pattern like "114-திருப்பூர் (தெற்கு)"
            r"(\d{2,3})[-\s]*([\u0B80-\u0BFF\u200c]+(?:\s*\([^\)]+\))?)",
        ]
        
        for pattern in assembly_patterns_en + assembly_patterns_ta:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                # Handle the fallback pattern which has 2 groups
                if match.lastindex == 2:
                    assembly_info = f"{match.group(1)}-{match.group(2)}".strip()
                else:
                    assembly_info = match.group(1).strip()
                # Clean up: remove trailing punctuation and normalize
                assembly_info = re.sub(r'[:\s]+$', '', assembly_info)
                assembly_info = re.sub(r'\s+', ' ', assembly_info).strip()
                # Remove ZWNJ characters for cleaner output
                assembly_info = assembly_info.replace('\u200c', '')
                if assembly_info:
                    break
        
        # --- SECTION PATTERNS ---
        
        # English: "Section No and Name 1-Karupparayan Kovil Street Ward No-9"
        # Use [^\n] to only match within a single line
        section_patterns_en = [
            r"Section\s*(?:No\.?|Number)?\s*(?:and|&)?\s*Name\s*[:\s]*([^\n]+?)(?=Part\s*(?:No|Number|\.|:)|பாகம்|\n|$)",
            r"Section\s*[:\s]+([^\n]+?)(?=Part|பாகம்|\n|$)",
        ]
        
        # Tamil: "பிரிவு எண் மற்றும் பெயர் 1-திருப்பூர் (மா), முருங்கப்பாளையம் வார்டு எண்-27"
        # Also handles: "பிரிவு எண்மற்றும் பெயர்" (no space)
        # Note: OCR produces ZWNJ (\u200c) after many Tamil letters
        section_patterns_ta = [
            # Match "பிரிவு எண் மற்றும் பெயர்" with ZWNJ and capture what follows as section value
            # Use a capturing group that starts with a number (the section number)
            r"பிரிவு[\u200c]?\s*எண்[\u200c]?\s*(?:மற்றும்[\u200c]?)?\s*பெயர்[\u200c]?\s*(\d+[-\s]*[^\n]+?)(?=பாகம்|Part|\n|$)",
            r"பிரிவு[\u200c]?\s*[:\s]+([^\n]+?)(?=பாகம்|Part|\n|$)",
        ]
        
        for pattern in section_patterns_en + section_patterns_ta:
            match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
            if match:
                section_info = match.group(1).strip()
                # FIX: Read street name "as is" with minimal cleaning
                # Only clean up newlines - take only the first line which contains the actual section info
                if '\n' in section_info:
                    section_info = section_info.split('\n')[0].strip()
                # Remove trailing colons and normalize whitespace
                section_info = re.sub(r'[:\s]+$', '', section_info)
                section_info = re.sub(r'\s+', ' ', section_info).strip()
                # Remove Part No. text that might be captured at the end
                section_info = re.sub(r'\s*Part\s*(?:No\.?)?\s*:?\s*$', '', section_info, flags=re.IGNORECASE).strip()
                section_info = re.sub(r'\s*பாகம்\s*(?:எண்)?\s*:?\s*\d*\s*$', '', section_info).strip()
                # Remove ZWNJ characters for cleaner output
                section_info = section_info.replace('\u200c', '')
                if section_info:
                    break
        
        
        # --- PART NUMBER PATTERNS ---
        
        # English: "Part No.: 244" or "Part No. : 244"
        part_patterns_en = [
            r"Part\s*(?:No\.?|Number)?\s*[:\.\s]+\s*(\d+)",
            r"Part\s*[:\s]+(\d+)",
        ]
        
        # Tamil: "பாகம் எண்: 1" or "பாகம் எண் : 1" 
        # OCR produces ZWNJ (\u200c) after Tamil letters
        part_patterns_ta = [
            r"பாகம்[\u200c]?\s*எண்[\u200c]?\s*[:\s]*(\d+)",
            r"பாகம்[\u200c]?\s*[:\s]+(\d+)",
        ]
        
        for pattern in part_patterns_en + part_patterns_ta:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                try:
                    part_number = int(match.group(1))
                    break
                except ValueError:
                    pass
        
        # If still not found, try to find any standalone number at the end (likely part number)
        if part_number is None:
            # Look for "எண் : N" or ": N" pattern at end
            end_number_match = re.search(r'[:\s](\d{1,3})\s*$', full_text)
            if end_number_match:
                try:
                    part_number = int(end_number_match.group(1))
                except ValueError:
                    pass
        
        return assembly_info, section_info, part_number
    
    def get_header_info(self, page_id: str) -> Optional[PageHeaderInfo]:
        """Get header info for a specific page."""
        return self.page_headers.get(page_id)
    
    def get_all_headers(self) -> Dict[str, PageHeaderInfo]:
        """Get all extracted headers."""
        return self.page_headers


def extract_page_headers(
    extracted_dir: Path,
    languages: str = "eng+tam",
) -> Dict[str, PageHeaderInfo]:
    """
    Convenience function to extract headers from all pages.
    
    Args:
        extracted_dir: Path to extracted folder
        languages: OCR languages
    
    Returns:
        Dict mapping page_id to PageHeaderInfo
    """
    from ..config import Config
    
    config = Config()
    context = ProcessingContext(config=config)
    context.setup_paths_from_extracted(extracted_dir)
    
    extractor = HeaderExtractor(context, languages=languages)
    
    if not extractor.run():
        raise ProcessingError("Header extraction failed")
    
    return extractor.get_all_headers()
