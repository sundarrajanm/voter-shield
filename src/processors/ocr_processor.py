"""
OCR Processor for voter information extraction.

Extracts structured voter data from cropped images using Tamil OCR:
- EPIC number (via ROI extraction)
- Serial number (via ROI extraction)
- Name, relation, house number, age, gender (via line-based text extraction)

Uses ocr_tamil (https://github.com/gnana70/tamil_ocr) for better Tamil text recognition.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
from PIL import Image

# Tamil OCR - better accuracy for Tamil text
try:
    from ocr_tamil.ocr import OCR as TamilOCR
    TAMIL_OCR_AVAILABLE = True
except ImportError:
    TAMIL_OCR_AVAILABLE = False

# Tesseract as fallback
try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from .base import BaseProcessor, ProcessingContext
from ..config import ROIConfig
from ..models import Voter
from ..exceptions import OCRProcessingError
from ..utils.ai_deleted_detector import AIDeletedDetector


# Character whitelists
WL_EPIC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
WL_DIGITS = "0123456789"
WL_HOUSE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/- "

# Label variants for field extraction
NAME_LABELS = ["name", "பெயர்", "பெயர்‌"]

RELATION_PATTERNS = {
    "father": [
        r"தந்தையின்\s*பெயர்",
        r"தந்தை\s*பெயர்",
        r"father'?s?\s*name",
        r"father",
    ],
    "mother": [
        r"தாயின்\s*பெயர்",
        r"தாய்\s*பெயர்",
        r"mother'?s?\s*name",
        r"mother",
    ],
    "husband": [
        r"கணவர்\s*பெயர்",
        r"கணவர்",
        r"husband'?s?\s*name",
        r"husband",
    ],
}

HOUSE_PATTERNS = [
    r"வீட்டு\s*எண்",
    r"வீட்டுஎண்",
    r"ட்டு\s*எண்",
    r"வீட்டு\s*எண்",
    r"வீட்டுஎண்",
    r"ட்டு\s*எண்",
    r"வீட்டு",
    r"எண்",
    r"house\s*(?:no|number)?",
    r"Doorno",
    r"Door\s*no",
]

AGE_PATTERNS = [r"வயது", r"age"]
GENDER_PATTERNS = [r"பாலினம்", r"gender"]

GENDER_MAP = {
    "ஆண்": "Male", "ஆண": "Male", "male": "Male", "m": "Male",
    "பெண்": "Female", "பெண": "Female", "female": "Female", "f": "Female",
    "திருநங்கை": "Other", "other": "Other", "o": "Other",
}



# Voter separator markers (from voter_end.jpg)
VOTER_END_MARKERS = ["voter_end", "voter-end", "voterend", "VOTER_END", "VOTER-END", "VOTEREND"]

# Tamil digit to English digit mapping (௦-௯ → 0-9)
TAMIL_DIGIT_MAP = {
    "௦": "0", "௧": "1", "௨": "2", "௩": "3", "௪": "4",
    "௫": "5", "௬": "6", "௭": "7", "௮": "8", "௯": "9",
}

def convert_tamil_digits(text: str) -> str:
    """Convert Tamil digits to English digits."""
    for tamil, english in TAMIL_DIGIT_MAP.items():
        text = text.replace(tamil, english)
    return text


@dataclass
class WordBox:
    """OCR word bounding box."""
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: int
    line_num: int
    block_num: int
    par_num: int


@dataclass
class OCRResult:
    """Result of OCR processing for a single voter image."""
    image_name: str
    serial_no: str = ""
    epic_no: str = ""
    epic_valid: bool = False
    name: str = ""
    relation_type: str = ""
    relation_name: str = ""
    house_no: str = ""
    age: str = ""
    gender: str = ""
    deleted: str = ""  # Empty string = not deleted, "true" = deleted
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    
    def to_voter(self, sequence_in_page: int = 0, sequence_in_document: int = 0) -> Voter:
        """Convert to Voter model."""
        # Per user request: Serial number MUST be the sequence in document to avoid OCR manipulation
        if sequence_in_document > 0:
            serial_no = str(sequence_in_document)
        elif self.serial_no:
            serial_no = self.serial_no
        else:
            serial_no = str(sequence_in_page) if sequence_in_page > 0 else ""
        
        # Calculate extraction confidence based on field completeness
        fields_present = sum([
            bool(self.epic_no and self.epic_valid),  # 30%
            bool(self.name),  # 25%
            bool(self.relation_type and self.relation_name),  # 20%
            bool(self.house_no),  # 10%
            bool(self.age),  # 10%
            bool(self.gender),  # 5%
        ])
        confidence_weights = [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]
        extraction_confidence = sum(confidence_weights[i] for i in range(fields_present))
        
        return Voter(
            serial_no=serial_no,
            epic_no=self.epic_no,
            name=self.name,
            relation_type=self.relation_type,
            relation_name=self.relation_name,
            house_no=self.house_no,
            age=self.age,
            gender=self.gender,
            sequence_in_page=sequence_in_page,
            sequence_in_document=sequence_in_document,
            image_file=self.image_name,
            epic_valid=self.epic_valid,
            processing_time_ms=round(self.elapsed_seconds * 1000, 2),
            extraction_confidence=round(extraction_confidence, 2),
            deleted=self.deleted,
        )


@dataclass
class PageOCRResult:
    """OCR results for a page."""
    page_id: str
    images_processed: int
    total_seconds: float
    records: List[OCRResult] = field(default_factory=list)


class OCRProcessor(BaseProcessor):
    """
    Extract voter information from cropped images using Tamil OCR.
    
    Uses hybrid extraction:
    - EPIC and Serial: ROI-based extraction with character whitelisting
    - Other fields: Line-based text pattern matching via Tamil OCR
    """
    
    name = "OCRProcessor"
    
    # Singleton OCR instance to avoid reloading models
    _ocr_instance: Optional['TamilOCR'] = None
    
    def __init__(
        self,
        context: ProcessingContext,
        languages: str = "eng+tam",
        allow_next_line: bool = True,
        dump_raw_ocr: bool = False,
        use_cuda: bool = True,
        use_merged: bool = False,
        use_tesseract: bool = False,
        ai_id_processor: Optional[Any] = None,
        on_page_complete: Optional[callable] = None,
    ):
        """
        Initialize OCR processor.
        
        Args:
            context: Processing context
            languages: Language codes (will be auto-detected from metadata if available)
            allow_next_line: Allow value on next line if not found on label line
            dump_raw_ocr: Dump raw OCR text for debugging
            use_cuda: Enable CUDA for Tamil OCR (if available)
            use_merged: Use merged images from /merged folder for faster processing
            use_tesseract: Use Tesseract OCR instead of ocr_tamil (faster on CPU)
            on_page_complete: Callback(page_id, voters, page_time) called after each page
        """
        super().__init__(context)
        # Detect language from metadata and adjust OCR languages accordingly
        detected_lang = self._detect_language_from_metadata()
        if detected_lang:
            self.languages = detected_lang
            self.log_info(f"Language auto-detected from metadata: {detected_lang}")
        else:
            self.languages = languages
        self.allow_next_line = allow_next_line
        self.dump_raw_ocr = dump_raw_ocr
        self.use_cuda = use_cuda
        self.use_merged = use_merged
        self.use_tesseract = use_tesseract
        self.ocr_config = self.config.ocr  # Contains ROI configs
        self.ai_id_processor = ai_id_processor
        self.page_results: List[PageOCRResult] = []
        self._ocr_initialized = False
        self.on_page_complete = on_page_complete
        
        # Merged images directory - inside extracted folder
        self.merged_dir = self.context.extracted_dir / "merged" if self.context.extracted_dir else None
        
        # AI-based deleted detector for when OCR can't extract fields
        # (used when relation_name, age, gender are all empty)
        self._ai_deleted_detector = AIDeletedDetector(
            api_key=self.config.ai.api_key,
            base_url=self.config.ai.get_normalized_base_url(),
            model=self.config.ai.model,
        )
    
    def _detect_language_from_metadata(self) -> Optional[str]:
        """
        Detect language from metadata JSON file.
        
        Returns:
            Language string for OCR (e.g., "eng" or "eng+tam" or "tam+eng") or None if not found
        """
        try:
            # Look for metadata JSON file in output directory
            if not self.context.output_dir:
                return None
            
            metadata_path = self.context.output_dir / f"{self.context.pdf_name}-metadata.json"
            if not metadata_path.exists():
                return None
            
            # Read metadata
            import json
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            
            # Get language_detected field
            languages_detected = metadata.get("language_detected", [])
            if not languages_detected:
                return None
            
            # Convert to lowercase for comparison
            languages_detected_lower = [lang.lower() for lang in languages_detected]
            
            # Determine OCR language setting based on detected languages
            has_tamil = any("tamil" in lang for lang in languages_detected_lower)
            has_english = any("english" in lang for lang in languages_detected_lower)
            
            if has_tamil:
                # Tamil detected - use both Tamil and English
                return "tam+eng"
            elif has_english:
                # Only English detected - use English only
                return "eng"
            
            return None
            
        except Exception as e:
            self.log_debug(f"Could not detect language from metadata: {e}")
            return None
    
    def validate(self) -> bool:
        """Validate prerequisites."""
        # Check OCR availability based on selected engine
        if self.use_tesseract:
            if not TESSERACT_AVAILABLE:
                self.log_error("Tesseract OCR not available. Install: pip install pytesseract")
                return False
            self.log_info(f"Using Tesseract OCR (faster on CPU) with languages: {self.languages}")
        else:
            if not TAMIL_OCR_AVAILABLE:
                self.log_error("Tamil OCR not available. Install: pip install ocr-tamil")
                return False
        
        # For merged mode, check merged directory
        if self.use_merged:
            if not self.merged_dir or not self.merged_dir.exists():
                self.log_warning(f"Merged directory not found, falling back to individual crops")
                self.use_merged = False
        
        # Check crops directory for fallback or non-merged mode
        if not self.use_merged:
            if not self.context.crops_dir:
                self.log_error("Crops directory not set")
                return False
            
            if not self.context.crops_dir.exists():
                self.log_error(f"Crops directory not found: {self.context.crops_dir}")
                return False
        
        return True
    
    def _initialize_ocr(self) -> None:
        """Initialize OCR engine (singleton pattern for Tamil OCR)."""
        if self._ocr_initialized:
            return
        
        # Skip Tamil OCR initialization if using Tesseract
        if self.use_tesseract:
            self._initialize_tesseract()
            self._ocr_initialized = True
            return
        
        if OCRProcessor._ocr_instance is None:
            # Determine Tamil OCR language list based on detected languages
            tamil_ocr_langs = ["english"]  # Default to English only
            if "tam" in self.languages.lower():
                tamil_ocr_langs = ["tamil", "english"]  # Include Tamil if detected
            
            self.log_info(f"Initializing Tamil OCR engine with languages: {tamil_ocr_langs} (first time, may download models)...")
            try:
                # Initialize Tamil OCR with text detection enabled
                # detect=True enables CRAFT text detection
                # details=2 gives us text, confidence, and bbox info
                OCRProcessor._ocr_instance = TamilOCR(
                    detect=True,
                    enable_cuda=self.use_cuda,
                    batch_size=8,
                    details=0,  # Just text output for simplicity
                    lang=tamil_ocr_langs,
                    recognize_thres=0.5,  # Lower threshold for voter card text
                )
                self.log_info(f"Tamil OCR initialized successfully with languages: {tamil_ocr_langs}")
            except Exception as e:
                raise OCRProcessingError(f"Failed to initialize Tamil OCR: {e}")
        
        self._ocr_initialized = True
    
    def _initialize_tesseract(self) -> None:
        """Initialize Tesseract OCR."""
        import os
        if os.name == "nt":
            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if Path(tesseract_path).exists():
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                self.log_info(f"Using Tesseract from: {tesseract_path}")
        
        try:
            version = pytesseract.get_tesseract_version()
            self.log_info(f"Tesseract version: {version}")
        except Exception as e:
            raise OCRProcessingError(f"Tesseract not available: {e}")
        
        self.log_info(f"Tesseract initialized (CPU mode, languages: {self.languages})")
    
    @property
    def ocr(self) -> 'TamilOCR':
        """Get the OCR instance."""
        if OCRProcessor._ocr_instance is None:
            self._initialize_ocr()
        return OCRProcessor._ocr_instance
    
    def process(self) -> bool:
        """
        Process all cropped images in the document.
        
        If use_merged is True and merged images exist, process those for faster
        extraction. Otherwise, fall back to individual cropped images.
        
        Returns:
            True if processing succeeded
        """
        self._initialize_ocr()
        
        # Try merged image processing if enabled
        if self.use_merged and self.merged_dir and self.merged_dir.exists():
            return self._process_merged_images()
        
        # Fall back to individual crop processing
        return self._process_individual_crops()
    
    def _process_individual_crops(self) -> bool:
        """
        Process individual cropped images (original method).
        
        Returns:
            True if processing succeeded
        """
        crops_dir = self.context.crops_dir
        page_dirs = self._get_page_dirs(crops_dir)
        
        if not page_dirs:
            self.log_warning(f"No page directories found in {crops_dir}")
            return False
        
        self.log_info(f"Found {len(page_dirs)} page(s) with crops")
        
        total_images = 0
        total_voters = 0
        
        for page_dir in page_dirs:
            result = self._process_page(page_dir)
            self.page_results.append(result)
            total_images += result.images_processed
            total_voters += len(result.records)
            self.context.total_voters_found += len(result.records)
            
            # Call page complete callback for immediate saving
            if self.on_page_complete:
                page_voters = []
                for idx, ocr_result in enumerate(result.records, start=1):
                    if not ocr_result.error:
                        sequence = idx
                        voter = ocr_result.to_voter(
                            sequence_in_page=sequence,
                            sequence_in_document=total_voters - len(result.records) + idx
                        )
                        # Set serial_no to sequence_in_page if serial_no is empty 
                        # or if user explicitly requested serial_no to be sequence_in_page
                        if not voter.serial_no:
                            voter.serial_no = str(sequence)
                            
                        voter.page_id = result.page_id
                        page_voters.append(voter)
                self.on_page_complete(result.page_id, page_voters, result.total_seconds)
        
        self.log_info(f"OCR complete: {total_images} images processed")
        
        return True
    
    def _process_merged_images(self) -> bool:
        """
        Process merged batch images for faster OCR.
        
        Each batch image contains up to 10 voter images separated by voter_end markers.
        Uses template matching to split the batch into individual voter images,
        then runs OCR on each segment.
        
        Merged structure: merged/<page-XXX>/<batch-001.png, batch-002.png, ...>
        
        Returns:
            True if processing succeeded
        """
        if not self.merged_dir or not self.merged_dir.exists():
            self.log_warning(f"Merged directory not found at {self.merged_dir}, falling back to individual crops")
            return self._process_individual_crops()
        
        # Load voter_end template for template matching
        voter_end_path = self.config.base_dir / "voter_end.jpg"
        if not voter_end_path.exists():
            self.log_warning("voter_end.jpg not found, falling back to individual crops")
            return self._process_individual_crops()
        
        voter_end_template = cv2.imdecode(
            np.fromfile(str(voter_end_path), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        if voter_end_template is None:
            self.log_warning("Failed to load voter_end.jpg, falling back to individual crops")
            return self._process_individual_crops()
        
        # Get page directories (same structure as crops)
        page_dirs = self._get_page_dirs(self.merged_dir)
        
        if not page_dirs:
            self.log_warning(f"No page directories found in {self.merged_dir}, falling back to individual crops")
            return self._process_individual_crops()
        
        self.log_info(f"Processing {len(page_dirs)} page(s) with merged batches")
        
        total_voters = 0
        total_batches = 0
        
        # AI processor available - build GLOBAL lookup by serial_no across ALL pages
        # This ensures correct matching even when AI page numbers don't match OCR page numbers
        has_ai_processor = self.ai_id_processor is not None
        global_ai_by_serial = {}
        if has_ai_processor:
            # Build global lookup from ALL AI results across ALL pages
            all_ai_results = self.ai_id_processor.get_all_results()
            for ai_result in all_ai_results:
                if ai_result.serial_no and ai_result.serial_no.strip():
                    global_ai_by_serial[ai_result.serial_no.strip()] = ai_result
            self.log_info(f"AI ID processor available with {len(global_ai_by_serial)} results for global serial_no matching")
            # Debug: show first 20 serial keys
            first_keys = sorted(global_ai_by_serial.keys(), key=lambda x: int(x) if x.isdigit() else 0)[:20]
            self.log_debug(f"First 20 AI serial keys: {first_keys}")
        
        for page_dir in page_dirs:
            page_id = page_dir.name
            batch_images = self._get_batch_images(page_dir)
            
            if not batch_images:
                self.log_debug(f"No batch images in {page_dir}")
                continue
            
            self.log_debug(f"Processing {len(batch_images)} batches from {page_id}")
            
            page_records: List[OCRResult] = []
            page_start_time = time.perf_counter()

            for batch_path in batch_images:
                batch_records = self._process_batch_image(batch_path, page_id, voter_end_template)
                page_records.extend(batch_records)
                total_batches += 1
            
            # Apply AI house numbers by matching SERIAL NUMBER from GLOBAL lookup
            # Use calculated sequence_in_document (not OCR serial_no) for reliable matching
            if has_ai_processor and global_ai_by_serial:
                ai_applied_count = 0
                ocr_fallback_count = 0
                
                for idx, record in enumerate(page_records):
                    ocr_house = record.house_no  # Save original OCR value
                    
                    # Calculate sequence_in_document for this record
                    # At this point, total_voters only contains count from PREVIOUS pages
                    # So sequence_in_document = previous_count + current_position (1-indexed)
                    seq_in_doc = total_voters + idx + 1
                    serial_key = str(seq_in_doc)
                    
                    # Debug logging for first 3 records on each page
                    if idx < 3:
                        exists = serial_key in global_ai_by_serial
                        ai_val = global_ai_by_serial.get(serial_key)
                        self.log_debug(
                            f"[{page_id}] idx={idx}, total_voters={total_voters}, "
                            f"seq_in_doc={seq_in_doc}, serial_key='{serial_key}', "
                            f"exists={exists}, ai_house='{ai_val.house_no if ai_val else None}'"
                        )
                    
                    # Match house_no by calculated sequence_in_document (which becomes serial_no)
                    ai_result = global_ai_by_serial.get(serial_key)
                    
                    if ai_result and ai_result.house_no and ai_result.house_no.strip():
                        ai_house = ai_result.house_no.strip()
                        
                        # NEW LOGIC: If AI house_no is all non-numeric characters (no digits)
                        # AND OCR has numeric characters, prefer OCR value
                        should_use_ocr = (
                            self._is_all_non_numeric(ai_house) and 
                            self._contains_numeric(ocr_house)
                        )
                        
                        if should_use_ocr:
                            # Keep OCR value instead of AI
                            # record.house_no already has ocr_house, so we don't need to set it
                            ocr_fallback_count += 1
                            self.log_debug(
                                f"Preferring OCR house_no '{ocr_house}' over AI '{ai_house}' for serial {serial_key} "
                                f"on {page_id} (AI has no numbers, OCR has numbers)"
                            )
                        else:
                            # Use AI value (default behavior)
                            record.house_no = ai_house
                            ai_applied_count += 1
                            if ocr_house != ai_house:
                                self.log_debug(
                                    f"Applied AI house_no '{ai_house}' for serial {serial_key} "
                                    f"on {page_id} (OCR was '{ocr_house}')"
                                )
                    else:
                        # No AI result found for this serial, keep OCR value
                        ocr_fallback_count += 1
                        if idx < 3:  # Log first 3 misses for debugging
                            available_keys = list(global_ai_by_serial.keys())[:10]
                            self.log_debug(
                                f"No AI result for serial {serial_key} on {page_id}, "
                                f"keeping OCR house '{ocr_house}'. Available keys (first 10): {available_keys}"
                            )
                
                self.log_info(
                    f"Applied {ai_applied_count} AI house numbers on {page_id} "
                    f"({ocr_fallback_count} used OCR fallback)"
                )
            elif not has_ai_processor:
                # No AI processor available - all voters keep their OCR values
                self.log_debug(f"No AI processor available, using OCR values for {page_id}")
            
            
            page_time = time.perf_counter() - page_start_time
            merged_count = sum(1 for r in page_records if r.epic_valid and r.epic_no)
            self.log_info(f"Completed page {page_id}: {len(page_records)} voters")
            
            result = PageOCRResult(
                page_id=page_id,
                images_processed=len(page_records),
                total_seconds=page_time,
                records=page_records,
            )
            
            self.page_results.append(result)
            total_voters += len(page_records)
            self.context.total_voters_found += len(page_records)
            
            # Call page complete callback for immediate saving
            if self.on_page_complete:
                # Convert OCR results to Voters for the callback
                page_voters = []
                for idx, ocr_result in enumerate(page_records, start=1):
                    if not ocr_result.error:
                        sequence = idx
                        voter = ocr_result.to_voter(
                            sequence_in_page=sequence,
                            sequence_in_document=total_voters - len(page_records) + idx
                        )
                        # Set serial_no to sequence_in_page if serial_no is empty 
                        if not voter.serial_no:
                            voter.serial_no = str(sequence)

                        voter.page_id = page_id
                        page_voters.append(voter)
                self.on_page_complete(page_id, page_voters, page_time)
            
            self.log_info(f"Processed {page_id}: {len(page_records)} voters from {len(batch_images)} batch(es)")
        
        self.log_info(f"OCR complete (merged batches): {total_voters} voters from {total_batches} batches")
        
        return True
    
    def _get_batch_images(self, page_dir: Path) -> List[Path]:
        """Get sorted batch images from a page directory."""
        if not page_dir.exists():
            return []
        exts = {".png", ".jpg", ".jpeg"}
        images = [p for p in page_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        return sorted(images)
    
    def _process_batch_image(
        self, 
        batch_path: Path, 
        page_id: str,
        voter_end_template: np.ndarray
    ) -> List[OCRResult]:
        """
        Process a single batch image containing multiple voters.
        
        Optimized Strategy (Batch Inference):
        1. Find visual separators to split into voter segments.
        2. Run OCR on ALL segments in one batch call (very fast, no I/O).
        3. Extract fields for each voter from the batch results.
        
        Args:
            batch_path: Path to batch image
            page_id: Parent page ID for naming
            voter_end_template: Template image for matching separator
        
        Returns:
            List of OCRResult for each voter in the batch
        """
        self.log_debug(f"Processing batch: {batch_path.name}")
        
        img_bgr = cv2.imdecode(
            np.fromfile(str(batch_path), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if img_bgr is None:
            self.log_error(f"Failed to load batch image: {batch_path}")
            return []
        
        # 1. Visual Splitting
        separator_positions = self._find_voter_end_positions(img_bgr, voter_end_template)
        
        if not separator_positions:
            self.log_warning(f"No voter_end separators found in {batch_path.name}")
            return []
        
        template_height = voter_end_template.shape[0]
        voter_segments = self._split_by_separators(img_bgr, separator_positions, template_height)
        
        # Filter valid segments
        valid_segments = []
        valid_indices = []
        for idx, seg in enumerate(voter_segments, start=1):
            if seg.shape[0] >= 50 and seg.shape[1] >= 50:
                valid_segments.append(seg)
                valid_indices.append(idx)
        
        if not valid_segments:
             return []

        # Extract batch number
        batch_name = batch_path.stem
        batch_num = batch_name.replace("batch-", "").replace("batch", "")
        BATCH_SIZE = self.config.merge.batch_size 
        try:
            batch_offset = (int(batch_num) - 1) * BATCH_SIZE
        except ValueError:
            batch_offset = 0
            
        # 2. OCR Inference - use batch for Tamil OCR, sequential for Tesseract
        batch_start = time.perf_counter()
        records: List[OCRResult] = []
        
        if self.use_tesseract:
            # Tesseract: process each segment sequentially
            for i, segment in enumerate(valid_segments):
                idx = valid_indices[i]
                voter_num = batch_offset + idx
                image_name = f"{page_id}-{voter_num:03d}.png"
                
                # Run Tesseract OCR on this segment
                lines = self._run_tesseract_ocr_on_segment(segment)
                
                # Process the result
                result = self._process_voter_result_from_lines(segment, lines, image_name)
                
                # Retry EPIC extraction from individual crop if invalid
                if not result.epic_valid:
                    retry_epic = self._retry_epic_from_crop(page_id, image_name)
                    if retry_epic:
                        result.epic_no = retry_epic
                        result.epic_valid = True

                # Retry age extraction from individual crop if invalid
                if not self._is_valid_age(result.age):
                    retry_age = self._retry_age_from_crop(page_id, image_name)
                    if retry_age:
                        result.age = retry_age
                
                # Retry house number extraction from individual crop if invalid/empty
                # Only retry if AI processor is NOT being used (AI takes precedence later)
                if not self.ai_id_processor and not self._is_valid_house_number(result.house_no):
                    retry_house = self._retry_house_from_crop(page_id, image_name)
                    if retry_house:
                        result.house_no = retry_house
                        self.log_debug(f"House number retry successful: {retry_house}")
                
                self.log_debug(f"Processed {image_name} in {result.elapsed_seconds:.4f}s")
                records.append(result)
            
            inference_time = time.perf_counter() - batch_start
            self.log_debug(f"Tesseract inference for {len(valid_segments)} segments took {inference_time:.4f}s")
        else:
            # Tamil OCR: batch inference (faster)
            try:
                ocr_results = self.ocr.predict(valid_segments)
            except Exception as e:
                self.log_error(f"Batch OCR failed: {e}")
                return []
                
            inference_time = time.perf_counter() - batch_start
            self.log_debug(f"Batch inference for {len(valid_segments)} segments took {inference_time:.4f}s")
            
            # 3. Process Results
            for i, ocr_output in enumerate(ocr_results):
                idx = valid_indices[i]
                segment = valid_segments[i]
                voter_num = batch_offset + idx
                image_name = f"{page_id}-{voter_num:03d}.png"
                
                # Use total batch time / count as approximate per-item time
                approx_time = inference_time / len(valid_segments)
                
                result = self._process_voter_result_from_ocr(segment, ocr_output, image_name)
                
                # Retry EPIC extraction from individual crop if invalid
                if not result.epic_valid:
                    retry_epic = self._retry_epic_from_crop(page_id, image_name)
                    if retry_epic:
                        result.epic_no = retry_epic
                        result.epic_valid = True

                # Retry age extraction from individual crop if invalid
                if not self._is_valid_age(result.age):
                    retry_age = self._retry_age_from_crop(page_id, image_name)
                    if retry_age:
                        result.age = retry_age
                
                # Retry house number extraction from individual crop if invalid/empty
                # Only retry if AI processor is NOT being used (AI takes precedence later)
                if not self.ai_id_processor and not self._is_valid_house_number(result.house_no):
                    retry_house = self._retry_house_from_crop(page_id, image_name)
                    if retry_house:
                        result.house_no = retry_house
                        self.log_debug(f"House number retry successful: {retry_house}")
                
                # Adjust reported time to include inference overhead
                result.elapsed_seconds += approx_time
                
                self.log_debug(f"Processed {image_name} in {result.elapsed_seconds:.4f}s")
                records.append(result)
        
        return records

    def _process_voter_result_from_ocr(
        self, 
        segment: np.ndarray, 
        ocr_output: Any,
        image_name: str
    ) -> OCRResult:
        """
        Process a single voter result using pre-computed OCR output.
        
        Args:
            segment: Image segment (numpy array)
            ocr_output: Raw output from OCR engine for this segment
            image_name: Virtual image name
        
        Returns:
            OCRResult with extracted fields
        """
        start_time = time.perf_counter()
        result = OCRResult(image_name=image_name)
        
        try:
            # Parse OCR output into lines and words
            lines, full_text_for_extract = self._parse_ocr_output(ocr_output)
            
            # OPTIMIZATION: Try to extract EPIC from full text first to avoid Tesseract/ROI overhead
            # This saves ~100-200ms per voter if successful
            # Epic ID format: Always 3 letters + 7 digits = 10 characters total
            epic_found_in_text = False
            # Use negative lookahead to ensure no extra digits after the 7 digits
            # This prevents matching 11-char EPICs (3 letters + 8 digits) as valid
            match = re.search(r'(?<![A-Z0-9])([A-Z]{3}\d{7})(?!\d)', full_text_for_extract.upper())
            if match:
                epic_candidate = match.group(1)
                # Validate: Must be exactly 10 characters (3 letters + 7 digits)
                if len(epic_candidate) == 10:
                    result.epic_no = epic_candidate
                    result.epic_valid = True
                    epic_found_in_text = True
                else:
                    # This shouldn't happen with the regex, but safety check
                    self.log_debug(f"Epic ID length mismatch ({epic_candidate}), treating as invalid")
                    epic_found_in_text = False
            
            if not epic_found_in_text:
                # Fallback to expensive ROI extraction
                result.epic_no = self._extract_epic(segment)
                # FIX: If Epic ID has exactly 11 characters (3 letters + 8 digits), treat as invalid
                # and force AI retry. Valid Epic IDs are exactly 10 characters (3 letters + 7 digits)
                if len(result.epic_no) == 11:
                    self.log_debug(f"ROI Epic ID has 11 characters ({result.epic_no}), marking as invalid")
                    result.epic_valid = False
                elif len(result.epic_no) == 10:
                    # Validate format: exactly 3 letters + 7 digits
                    result.epic_valid = bool(re.fullmatch(r"[A-Z]{3}\d{7}", result.epic_no))
                else:
                    # Any other length is also invalid
                    result.epic_valid = False
            
            # Serial is harder to regex reliably (often just digits), so we keep ROI for now.
            # Extract Serial from ROI (Opencv/Tesseract)
            result.serial_no = self._extract_serial(segment)
            
            # Extract fields from text
            result.name = self._extract_name(lines)
            result.relation_type, result.relation_name = self._extract_relation(lines)
            result.house_no = self._extract_house(lines, segment)
            result.age = self._extract_age(lines)
            result.gender = self._extract_gender(lines)
            
            # AI-based fallback when age is missing:
            # If age could not be extracted, send to AI to get age and check for DELETED
            if not result.age and self._ai_deleted_detector.is_available():
                self.log_debug(f"Age missing for {image_name}, checking with AI...")
                ai_result = self._ai_deleted_detector.extract_fields(image_array=segment)
                if ai_result.age:
                    result.age = ai_result.age
                    self.log_info(f"AI extracted age: {ai_result.age} from {image_name}")
                if ai_result.deleted:
                    result.deleted = "true"
                    self.log_info(f"AI detected DELETED mark on {image_name}")
            
        except Exception as e:
            result.error = str(e)
            self.log_debug(f"Extraction error for {image_name}: {e}")
            
        result.elapsed_seconds = time.perf_counter() - start_time
        return result

    def _parse_ocr_output(self, ocr_output: Any) -> Tuple[List[str], str]:
        """
        Convert raw OCR output to list of lines and full text string.
        output depends on ocr-tamil library structure.
        """
        lines = []
        words = []
        
        if isinstance(ocr_output, list):
            # It could be list of strings or list of [text, conf, box]
            for item in ocr_output:
                if isinstance(item, str):
                    clean_text = item.strip()
                    if clean_text:
                        lines.append(clean_text)
                        words.extend(clean_text.split())
                elif isinstance(item, (list, tuple)) and len(item) > 0:
                    # Assuming [bbox, text, conf] or similar
                    # Try to find the text part
                    text_part = str(item[0]) # naive fallback
                    # Let's try to stringify
                    clean_text = text_part.strip()
                    if clean_text:
                        lines.append(clean_text)
                        words.extend(clean_text.split())
                        
        elif isinstance(ocr_output, str):
            lines = [line.strip() for line in ocr_output.split('\n') if line.strip()]
            words = ocr_output.split()
            
        # Add raw words line for regex helpers
        if words:
            lines.insert(0, "__RAW__:" + "|".join(words))
            
        return lines, "".join(lines)
    
    def _is_valid_age(self, age: str) -> bool:
        """
        Check if extracted age is valid.
        
        Age is invalid if:
        - Empty or missing
        - Single digit (likely OCR error - ages should be 2+ digits for adults)
        - Outside reasonable range (18-120 for voters)
        """
        if not age:
            return False
        
        try:
            age_val = int(age)
            # Single digit ages are likely OCR errors (voters are typically 18+)
            if age_val < 10:
                return False
            # Valid voter age range
            return 18 <= age_val <= 120
        except ValueError:
            return False
    
    def _retry_age_from_crop(self, page_id: str, image_name: str) -> str:
        """
        Retry age extraction from individual crop image.
        
        Used when merged image OCR fails to extract valid age.
        Falls back to /crops folder for individual voter images.
        
        Args:
            page_id: Page identifier (e.g., 'page-004')
            image_name: Image name (e.g., 'page-004-001.png')
            
        Returns:
            Extracted age string, or empty string if not found
        """
        if not self.context.crops_dir:
            return ""
        
        # Find the individual crop image
        # Structure: crops/<page_id>/images/<image_name>
        crop_path = self.context.crops_dir / page_id / "images" / image_name
        
        if not crop_path.exists():
            # Try without 'images' subdirectory
            crop_path = self.context.crops_dir / page_id / image_name
        
        if not crop_path.exists():
            self.log_debug(f"Crop image not found for age retry: {crop_path}")
            return ""
        
        self.log_debug(f"Retrying age extraction from individual crop: {crop_path}")
        
        try:
            # Load the individual crop image
            img_bgr = cv2.imdecode(
                np.fromfile(str(crop_path), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            
            if img_bgr is None:
                return ""
            
            # Run OCR on the individual image
            if self.use_tesseract:
                lines = self._run_tesseract_ocr(crop_path, img_bgr)
            else:
                lines = self._run_tamil_ocr(crop_path, img_bgr)
            
            # Extract age from the OCR result
            age = self._extract_age(lines)
            
            if self._is_valid_age(age):
                self.log_debug(f"Age retry successful: {age}")
                return age
            else:
                self.log_debug(f"Age retry failed, got: '{age}'")
                return ""
                
        except Exception as e:
            self.log_debug(f"Age retry error: {e}")
            return ""

    def _retry_house_from_crop(self, page_id: str, image_name: str) -> str:
        """
        Retry house number extraction from individual crop image.
        
        Used when merged image OCR fails to extract valid house number.
        Falls back to /crops folder for individual voter images.
        
        Args:
            page_id: Page identifier (e.g., 'page-004')
            image_name: Image name (e.g., 'page-004-001.png')
            
        Returns:
            Extracted house number string, or empty string if not found
        """
        if not self.context.crops_dir:
            return ""
        
        # Find the individual crop image
        # Structure: crops/<page_id>/images/<image_name>
        crop_path = self.context.crops_dir / page_id / "images" / image_name
        
        if not crop_path.exists():
            # Try without 'images' subdirectory
            crop_path = self.context.crops_dir / page_id / image_name
        
        if not crop_path.exists():
            self.log_debug(f"Crop image not found for house retry: {crop_path}")
            return ""
        
        self.log_debug(f"Retrying house number extraction from individual crop: {crop_path}")
        
        try:
            # Load the individual crop image
            img_bgr = cv2.imdecode(
                np.fromfile(str(crop_path), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            
            if img_bgr is None:
                return ""
            
            # Run OCR on the individual image
            if self.use_tesseract:
                lines = self._run_tesseract_ocr(crop_path, img_bgr)
            else:
                lines = self._run_tamil_ocr(crop_path, img_bgr)
            
            # Extract house number from the OCR result
            house_no = self._extract_house(lines, img_bgr)
            
            if self._is_valid_house_number(house_no):
                self.log_debug(f"House number retry successful: {house_no}")
                return house_no
            else:
                self.log_debug(f"House number retry failed, got: '{house_no}'")
                return ""
                
        except Exception as e:
            self.log_debug(f"House number retry error: {e}")
            return ""

    def _retry_epic_from_crop(self, page_id: str, image_name: str) -> str:
        """
        Retry EPIC extraction from individual crop image.
        """
        if not self.context.crops_dir:
            return ""
        
        # Find the individual crop image
        crop_path = self.context.crops_dir / page_id / "images" / image_name
        
        if not crop_path.exists():
            crop_path = self.context.crops_dir / page_id / image_name
        
        if not crop_path.exists():
            return ""
        
        try:
            img_bgr = cv2.imdecode(
                np.fromfile(str(crop_path), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            
            if img_bgr is None:
                return ""
            
            # 1. Try ROI extraction first (fastest and often most accurate for EPIC)
            epic = self._extract_epic(img_bgr)
            # FIX: If Epic ID has exactly 11 characters, skip this and try AI
            # Valid Epic IDs are exactly 10 characters (3 letters + 7 digits)
            if len(epic) == 10 and re.fullmatch(r"[A-Z]{3}\d{7}", epic):
                self.log_debug(f"EPIC retry successful (ROI): {epic}")
                return epic
            elif len(epic) == 11:
                self.log_debug(f"ROI Epic ID has 11 characters ({epic}), skipping to AI retry")
            
            # 2. Try OCR on just the cropped EPIC region
            epic = self._retry_epic_from_roi_crop(img_bgr, image_name)
            if epic:
                self.log_debug(f"EPIC retry successful (ROI Crop OCR): {epic}")
                return epic
            
            # 3. Try AI fallback on cropped EPIC region
            if self._ai_deleted_detector.is_available():
                self.log_info(f"EPIC invalid for {image_name}, trying AI extraction...")
                epic = self._retry_epic_with_ai(img_bgr, image_name)
                if epic:
                    self.log_info(f"EPIC retry successful (AI): {epic}")
                    return epic
                
        except Exception as e:
            self.log_debug(f"EPIC retry error: {e}")
            
        return ""

    def _retry_epic_from_roi_crop(self, img_bgr: np.ndarray, image_name: str) -> str:
        """
        Run OCR specifically on the EPIC ROI region.
        
        This crops just the EPIC field and runs full OCR on it,
        which can help when the full-image OCR is noisy.
        """
        try:
            # Crop the EPIC ROI
            roi = self.ocr_config.epic_roi.as_tuple()
            epic_crop = self._crop_roi(img_bgr, roi)
            
            if epic_crop.size == 0:
                return ""
            
            # Add white border for better recognition
            epic_crop = cv2.copyMakeBorder(
                epic_crop, 10, 10, 10, 10, 
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            
            # Run OCR on the cropped region
            if self.use_tesseract:
                lines = self._run_tesseract_ocr_on_segment(epic_crop)
            else:
                # For Tamil OCR, need to save temp file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                    cv2.imwrite(str(tmp_path), epic_crop)
                
                try:
                    lines = self._run_tamil_ocr(tmp_path, epic_crop)
                finally:
                    import os
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            
            # Extract EPIC from OCR text
            # Epic ID format: Always 3 letters + 7 digits = 10 characters total
            full_text = " ".join(line for line in lines if not line.startswith("__RAW__:"))
            match = re.search(r'(?<![A-Z0-9])([A-Z]{3}\d{7})(?!\d)', full_text.upper())
            if match:
                epic_candidate = match.group(1)
                # Reject if it's 11 characters (likely OCR error)
                if len(epic_candidate) == 10:
                    return epic_candidate
                elif len(epic_candidate) == 11:
                    self.log_debug(f"ROI crop OCR returned 11-character Epic ID ({epic_candidate}), rejecting")
                
        except Exception as e:
            self.log_debug(f"EPIC ROI crop OCR error: {e}")
            
        return ""
    
    def _retry_epic_with_ai(self, img_bgr: np.ndarray, image_name: str) -> str:
        """
        Use AI to extract EPIC number from the cropped EPIC ROI.
        
        This is the final fallback when OCR fails.
        """
        try:
            # Crop the EPIC ROI
            roi = self.ocr_config.epic_roi.as_tuple()
            epic_crop = self._crop_roi(img_bgr, roi)
            
            if epic_crop.size == 0:
                return ""
            
            # Save debug image to see what's being sent to AI
            if hasattr(self.context, 'debug_dir') and self.context.debug_dir:
                debug_path = self.context.debug_dir / f"epic_ai_retry_{image_name}"
                cv2.imwrite(str(debug_path), epic_crop)
                self.log_info(f"Saved AI retry EPIC crop for {image_name} to {debug_path}")
            
            # Send to AI
            ai_result = self._ai_deleted_detector.extract_fields(image_array=epic_crop)
            
            self.log_info(f"AI EPIC retry for {image_name}: AI returned '{ai_result.epic_no}' (length: {len(ai_result.epic_no) if ai_result.epic_no else 0})")
            
            if ai_result.epic_no:
                # Validate the format: Must be exactly 3 letters + 7 digits = 10 characters total
                # FIX: Explicitly reject 11-character Epic IDs even from AI results
                if len(ai_result.epic_no) == 10 and re.fullmatch(r"[A-Z]{3}\d{7}", ai_result.epic_no):
                    self.log_info(f"AI EPIC retry SUCCESS for {image_name}: {ai_result.epic_no}")
                    return ai_result.epic_no
                elif len(ai_result.epic_no) == 11:
                    self.log_warning(f"AI returned 11-character Epic ID for {image_name} ({ai_result.epic_no}), rejecting")
                else:
                    self.log_warning(f"AI returned Epic ID with invalid format/length for {image_name} ({ai_result.epic_no})")
            else:
                self.log_info(f"AI EPIC retry for {image_name}: No EPIC returned")
                    
        except Exception as e:
            self.log_error(f"EPIC AI extraction error for {image_name}: {e}")
            
        return ""

    def _split_by_voter_end(self, text: str) -> List[str]:
        """Split OCR text by voter_end markers."""
        pattern = r'|'.join(re.escape(m) for m in VOTER_END_MARKERS)
        segments = re.split(pattern, text, flags=re.IGNORECASE)
        return [s.strip() for s in segments if s.strip()]

    def _extract_voter_from_text(self, text: str, image_name: str) -> OCRResult:
        """Extract voter information from a text segment."""
        start_time = time.perf_counter()
        result = OCRResult(image_name=image_name)
        
        try:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if not lines: lines = [text]
            
            words = text.split()
            if words: lines.insert(0, "__RAW__:" + "|".join(words))
            
            # Epic ID format: Always 3 letters + 7 digits = 10 characters total
            epic_match = re.search(r'(?<![A-Z0-9])([A-Z]{3}\d{7})(?!\d)', text.upper())
            if epic_match:
                epic_candidate = epic_match.group(1)
                # Only accept if exactly 10 characters (reject 11-character IDs)
                if len(epic_candidate) == 10:
                    result.epic_no = epic_candidate
                    result.epic_valid = True
            
            serial_match = re.search(r'^\s*(\d{1,4})\b', text)
            if serial_match:
                result.serial_no = serial_match.group(1)
            
            result.name = self._extract_name(lines)
            result.relation_type, result.relation_name = self._extract_relation(lines)
            result.age = self._extract_age(lines)
            result.gender = self._extract_gender(lines)
            result.house_no = self._extract_house_from_text(text)
            
        except Exception as e:
            result.error = str(e)
        
        result.elapsed_seconds = time.perf_counter() - start_time
        return result
    
    def _find_voter_end_positions(
        self, 
        img: np.ndarray, 
        template: np.ndarray,
        threshold: float = 0.7
    ) -> List[int]:
        """
        Find vertical positions of voter_end separators using template matching.
        
        Args:
            img: The merged image
            template: The voter_end template image
            threshold: Matching threshold (0-1)
        
        Returns:
            List of y-coordinates where separators were found
        """
        # Convert to grayscale for matching
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # In ImageMerger, the voter_end image is padded, not scaled.
        # So we should primarily look for the template at its original scale.
        
        positions = []
        scales = [1.0]
        
        # Try slight variations just in case of minor resizing artifacts
        if img.shape[1] > template.shape[1] * 2:
             scales = [1.0, 0.95, 1.05]
        
        best_score = 0.0
        
        for scale in scales:
            if scale != 1.0:
                new_width = int(template.shape[1] * scale)
                new_height = int(template.shape[0] * scale)
                if new_width > img.shape[1] or new_height > img.shape[0]:
                    continue
                scaled_template = cv2.resize(template_gray, (new_width, new_height))
            else:
                scaled_template = template_gray
            
            # Template matching
            try:
                result = cv2.matchTemplate(img_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                
                # Find all locations above threshold
                locations = np.where(result >= threshold)
                
                for y in locations[0]:
                    # Check if this position is not too close to an existing one
                    is_duplicate = False
                    for existing_y in positions:
                        if abs(y - existing_y) < template.shape[0] * 2:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        positions.append(int(y))
                        
            except cv2.error as e:
                self.log_debug(f"Template matching error at scale {scale}: {e}")
                continue
        
        if not positions:
            self.log_debug(f"No separators found. Best score was {best_score:.4f} (threshold: {threshold})")
        
        # Sort positions by y-coordinate
        return sorted(positions)
    
    def _split_by_separators(
        self,
        img: np.ndarray,
        separator_positions: List[int],
        template_height: int
    ) -> List[np.ndarray]:
        """
        Split image into segments based on separator positions.
        
        Args:
            img: The merged image
            separator_positions: Y-coordinates of separators
            template_height: Height of the separator template
        
        Returns:
            List of image segments (one per voter)
        """
        segments = []
        
        # Add start and end boundaries
        start_y = 0
        
        for sep_y in separator_positions:
            # Segment from start_y to separator
            if sep_y > start_y:
                segment = img[start_y:sep_y, :]
                segments.append(segment)
            
            # Move start to after the separator
            start_y = sep_y + template_height
        
        # Don't add remaining segment after last separator 
        # (it's just the last voter_end with nothing after)
        
        return segments
    
    def _process_voter_segment(self, segment: np.ndarray, image_name: str) -> OCRResult:
        """
        Process a single voter segment extracted from merged image.
        
        Args:
            segment: Image segment for one voter
            image_name: Virtual image name for this voter
        
        Returns:
            OCRResult with extracted fields
        """
        start_time = time.perf_counter()
        result = OCRResult(image_name=image_name)
        
        try:
            # Save segment temporarily for OCR (Tamil OCR expects file path)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, segment)
            
            try:
                # Run Tamil OCR on the segment
                lines = self._run_tamil_ocr(Path(tmp_path), segment)
                
                # Extract EPIC from ROI
                result.epic_no = self._extract_epic(segment)
                result.epic_valid = bool(re.fullmatch(r"[A-Z]{3}\d+", result.epic_no))
                
                # Extract Serial from ROI
                result.serial_no = self._extract_serial(segment)
                
                # Extract fields from lines
                result.name = self._extract_name(lines)
                result.relation_type, result.relation_name = self._extract_relation(lines)
                result.house_no = self._extract_house(lines, segment)
                result.age = self._extract_age(lines)
                result.gender = self._extract_gender(lines)
                
                # Detect deleted mark (ROI-based detection)
                if self._detect_deleted_mark(segment):
                    result.deleted = "true"
                
            finally:
                # Clean up temp file
                import os
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
        except Exception as e:
            result.error = str(e)
            self.log_debug(f"OCR error for {image_name}: {e}")
        
        result.elapsed_seconds = time.perf_counter() - start_time
        return result

    def _extract_house_from_text(self, text: str) -> str:
        """Extract house number from text segment."""
        for pattern in HOUSE_PATTERNS:
            match = re.search(
                rf"(?:{pattern})\s*[:\-–—]?\s*([A-Za-z0-9\-/]+)",
                text,
                re.IGNORECASE | re.UNICODE
            )
            if match:
                return self._clean_house_number(match.group(1).strip())
        return ""
    
    def _get_merged_images(self) -> List[Path]:
        """Legacy method - get all merged page images (deprecated)."""
        if not self.merged_dir or not self.merged_dir.exists():
            return []
        exts = {".png", ".jpg", ".jpeg"}
        images = [p for p in self.merged_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        return sorted(images)
    

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
    
    def _process_page(self, page_dir: Path) -> PageOCRResult:
        """Process all images in a page directory."""
        page_id = page_dir.name
        
        # Check for images subdirectory
        images_dir = page_dir / "images" if (page_dir / "images").exists() else page_dir
        images = self._get_images(images_dir)
        
        if not images:
            self.log_debug(f"No images in {page_dir}")
            return PageOCRResult(page_id=page_id, images_processed=0, total_seconds=0.0)
        
        start_time = time.perf_counter()
        records: List[OCRResult] = []
        
        for idx, image_path in enumerate(images, start=1):
            result = self._process_image(image_path)
            records.append(result)
            
            self.log_debug(
                f"[{idx}/{len(images)}] {image_path.name}",
                epic=result.epic_no,
                time=f"{result.elapsed_seconds:.2f}s"
            )
        
        total_time = time.perf_counter() - start_time
        
        return PageOCRResult(
            page_id=page_id,
            images_processed=len(images),
            total_seconds=total_time,
            records=records,
        )
    
    def _process_image(self, image_path: Path) -> OCRResult:
        """
        Process a single voter image.
        
        Uses hybrid extraction:
        1. EPIC via ROI extraction
        2. Serial via ROI extraction
        3. Other fields via Tamil OCR or Tesseract (based on use_tesseract flag)
        """
        start_time = time.perf_counter()
        
        result = OCRResult(image_name=image_path.name)
        
        # Load image
        img_bgr = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if img_bgr is None:
            result.error = "Failed to read image"
            result.elapsed_seconds = time.perf_counter() - start_time
            return result
        
        try:
            # Full OCR - use Tesseract or Tamil OCR based on flag
            if self.use_tesseract:
                lines = self._run_tesseract_ocr(image_path, img_bgr)
            else:
                lines = self._run_tamil_ocr(image_path, img_bgr)
            
            # Extract EPIC from ROI (always uses Tesseract if available for best results)
            result.epic_no = self._extract_epic(img_bgr)
            # Validate: Must be exactly 10 characters (3 letters + 7 digits)
            if len(result.epic_no) == 11:
                self.log_debug(f"ROI Epic ID has 11 characters ({result.epic_no}), marking as invalid")
                result.epic_valid = False
            elif len(result.epic_no) == 10:
                result.epic_valid = bool(re.fullmatch(r"[A-Z]{3}\d{7}", result.epic_no))
            else:
                result.epic_valid = False
            
            # Fallback: Check full text for EPIC if ROI failed
            if not result.epic_valid:
                full_text = " ".join(line for line in lines if not line.startswith("__RAW__:"))
                match = re.search(r'(?<![A-Z0-9])([A-Z]{3}\d{7})(?!\d)', full_text.upper())
                if match:
                    epic_candidate = match.group(1)
                    # Only accept if exactly 10 characters
                    if len(epic_candidate) == 10:
                        result.epic_no = epic_candidate
                        result.epic_valid = True
            
            # Extract Serial from ROI
            result.serial_no = self._extract_serial(img_bgr)
            
            # Extract fields from lines
            result.name = self._extract_name(lines)
            result.relation_type, result.relation_name = self._extract_relation(lines)
            result.house_no = self._extract_house(lines, img_bgr)
            result.age = self._extract_age(lines)
            result.gender = self._extract_gender(lines)
            
            # AI-based fallback when age is missing:
            # If age could not be extracted, send to AI to get age and check for DELETED
            if not result.age and self._ai_deleted_detector.is_available():
                self.log_debug(f"Age missing for {image_path.name}, checking with AI...")
                ai_result = self._ai_deleted_detector.extract_fields(image_path=image_path)
                if ai_result.age:
                    result.age = ai_result.age
                    self.log_info(f"AI extracted age: {ai_result.age} from {image_path.name}")
                if ai_result.deleted:
                    result.deleted = "true"
                    self.log_info(f"AI detected DELETED mark on {image_path.name}")
            
        except Exception as e:
            result.error = str(e)
            self.log_debug(f"OCR error for {image_path.name}: {e}")
        
        result.elapsed_seconds = time.perf_counter() - start_time
        return result
    
    def _run_tesseract_ocr(self, image_path: Path, img_bgr: np.ndarray) -> List[str]:
        """
        Run Tesseract OCR on image and return lines.
        
        Tesseract is faster on CPU compared to Tamil OCR (which uses deep learning models).
        Uses image_to_data for word-level extraction with position info.
        """
        try:
            # Open image with PIL for Tesseract
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            
            # Run OCR with word-level data
            config = "--oem 1 --psm 6"
            data = pytesseract.image_to_data(
                pil_img, 
                lang=self.languages, 
                config=config, 
                output_type=Output.DICT
            )
            
            # Parse into lines
            lines = []
            raw_words = []
            
            # Get the dictionary from the data
            n = len(data.get("text", []))
            current_line_num = -1
            current_line_words = []
            
            for i in range(n):
                txt = (data["text"][i] or "").strip()
                if not txt:
                    continue
                    
                conf = int(float(data.get("conf", [-1])[i]))
                if conf != -1 and conf < 20:
                    continue
                
                line_num = data.get("line_num", [0])[i]
                raw_words.append(txt)
                
                # Group by line
                if line_num != current_line_num:
                    if current_line_words:
                        lines.append(" ".join(current_line_words))
                    current_line_words = [txt]
                    current_line_num = line_num
                else:
                    current_line_words.append(txt)
            
            # Add last line
            if current_line_words:
                lines.append(" ".join(current_line_words))
            
            # Add raw words as special line for alternative parsing
            if raw_words:
                lines.insert(0, "__RAW__:" + "|".join(raw_words))
            
            if self.dump_raw_ocr:
                self.log_debug(f"Tesseract raw: {raw_words}")
            
            return lines
            
        except Exception as e:
            self.log_debug(f"Tesseract OCR error: {e}")
            return []
    
    def _run_tesseract_ocr_on_segment(self, segment: np.ndarray) -> List[str]:
        """
        Run Tesseract OCR on a numpy array segment directly.
        
        Similar to _run_tesseract_ocr but takes a numpy array instead of file path.
        """
        try:
            # Convert BGR to RGB for PIL
            pil_img = Image.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
            
            # Run OCR with word-level data
            config = "--oem 1 --psm 6"
            data = pytesseract.image_to_data(
                pil_img, 
                lang=self.languages, 
                config=config, 
                output_type=Output.DICT
            )
            
            # Parse into lines
            lines = []
            raw_words = []
            
            n = len(data.get("text", []))
            current_line_num = -1
            current_line_words = []
            
            for i in range(n):
                txt = (data["text"][i] or "").strip()
                if not txt:
                    continue
                    
                conf = int(float(data.get("conf", [-1])[i]))
                if conf != -1 and conf < 20:
                    continue
                
                line_num = data.get("line_num", [0])[i]
                raw_words.append(txt)
                
                if line_num != current_line_num:
                    if current_line_words:
                        lines.append(" ".join(current_line_words))
                    current_line_words = [txt]
                    current_line_num = line_num
                else:
                    current_line_words.append(txt)
            
            if current_line_words:
                lines.append(" ".join(current_line_words))
            
            if raw_words:
                lines.insert(0, "__RAW__:" + "|".join(raw_words))
            
            return lines
            
        except Exception as e:
            self.log_debug(f"Tesseract OCR on segment error: {e}")
            return []
    
    def _process_voter_result_from_lines(
        self, 
        segment: np.ndarray, 
        lines: List[str],
        image_name: str
    ) -> OCRResult:
        """
        Process a single voter result from pre-computed OCR lines.
        
        Similar to _process_voter_result_from_ocr but takes parsed lines instead of raw OCR output.
        """
        start_time = time.perf_counter()
        result = OCRResult(image_name=image_name)
        
        try:
            # Reconstruct full text for EPIC extraction
            full_text = " ".join(line for line in lines if not line.startswith("__RAW__:"))
            
            # Try to extract EPIC from full text first
            # Epic ID format: Always 3 letters + 7 digits = 10 characters total
            epic_found_in_text = False
            match = re.search(r'(?<![A-Z0-9])([A-Z]{3}\d{7})(?!\d)', full_text.upper())
            if match:
                epic_candidate = match.group(1)
                # Only accept if exactly 10 characters
                if len(epic_candidate) == 10:
                    result.epic_no = epic_candidate
                    result.epic_valid = True
                    epic_found_in_text = True
            
            if not epic_found_in_text:
                # Fallback to ROI extraction
                result.epic_no = self._extract_epic(segment)
                # Validate: Must be exactly 10 characters (3 letters + 7 digits)
                if len(result.epic_no) == 11:
                    self.log_debug(f"ROI Epic ID has 11 characters ({result.epic_no}), marking as invalid")
                    result.epic_valid = False
                elif len(result.epic_no) == 10:
                    result.epic_valid = bool(re.fullmatch(r"[A-Z]{3}\d{7}", result.epic_no))
                else:
                    result.epic_valid = False
            
            # Extract Serial from ROI
            result.serial_no = self._extract_serial(segment)
            
            # Extract fields from lines
            result.name = self._extract_name(lines)
            result.relation_type, result.relation_name = self._extract_relation(lines)
            result.house_no = self._extract_house(lines, segment)
            result.age = self._extract_age(lines)
            result.gender = self._extract_gender(lines)
            
            # AI-based fallback when age is missing:
            # If age could not be extracted, send to AI to get age and check for DELETED
            if not result.age and self._ai_deleted_detector.is_available():
                self.log_debug(f"Age missing for {image_name}, checking with AI...")
                ai_result = self._ai_deleted_detector.extract_fields(image_array=segment)
                if ai_result.age:
                    result.age = ai_result.age
                    self.log_info(f"AI extracted age: {ai_result.age} from {image_name}")
                if ai_result.deleted:
                    result.deleted = "true"
                    self.log_info(f"AI detected DELETED mark on {image_name}")
            
        except Exception as e:
            result.error = str(e)
            self.log_debug(f"Extraction error for {image_name}: {e}")
            
        result.elapsed_seconds = time.perf_counter() - start_time
        return result
    
    # ==================== ROI Extraction ====================
    
    def _crop_roi(self, img: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
        """Crop image by relative ROI coordinates."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = roi
        X1 = max(0, min(w - 1, int(round(x1 * w))))
        Y1 = max(0, min(h - 1, int(round(y1 * h))))
        X2 = max(1, min(w, int(round(x2 * w))))
        Y2 = max(1, min(h, int(round(y2 * h))))
        return img[Y1:Y2, X1:X2]
    
    def _extract_epic(self, img_bgr: np.ndarray) -> str:
        """Extract EPIC number from ROI."""
        roi = self.ocr_config.epic_roi.as_tuple()
        epic_crop = self._crop_roi(img_bgr, roi)
        
        # Add white border for better Tesseract recognition
        epic_crop = cv2.copyMakeBorder(
            epic_crop, 10, 10, 10, 10, 
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        
        # Preprocess
        gray = cv2.cvtColor(epic_crop, cv2.COLOR_BGR2GRAY)
        # Use Linear interpolation (cleaner for text than Cubic)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        
        # Use Otsu's binarization
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Ensure black text on white background
        if float(np.mean(gray)) < 127:
            gray = 255 - gray
        
        # Use Tesseract if available (better for alphanumeric with whitelist)
        if TESSERACT_AVAILABLE:
            try:
                # Use PSM 6 (uniform block) instead of 7 (single line)
                config = f"--oem 1 --psm 6 -c tessedit_char_whitelist={WL_EPIC}"
                txt = pytesseract.image_to_string(Image.fromarray(gray), lang="eng", config=config)
                return self._clean_epic(txt.strip())
            except Exception:
                pass
        
        # Use Tamil OCR and filter for EPIC pattern
        try:
            # Convert grayscale to BGR for Tamil OCR
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            result = self.ocr.predict(gray_bgr)
            if result and isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    text = " ".join(str(item) for item in result[0] if item)
                else:
                    text = str(result[0]) if result[0] else ""
                return self._clean_epic(text)
        except Exception:
            pass
        
        return ""
    
    def _clean_epic(self, raw: str) -> str:
        """Clean EPIC number."""
        s = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
        if len(s) >= 3:
            prefix, rest = s[:3], s[3:]
            # Fix common OCR confusions in prefix (should be letters)
            prefix = prefix.replace("0", "O").replace("1", "I").replace("2", "Z").replace("5", "S").replace("8", "B")
            # Fix common OCR confusions in rest (should be digits)
            rest = rest.replace("O", "0").replace("I", "1").replace("Z", "2").replace("S", "5").replace("B", "8")
            s = prefix + rest
        return s
    
    def _extract_serial(self, img_bgr: np.ndarray) -> str:
        """Extract serial number from ROI."""
        roi = self.ocr_config.serial_roi.as_tuple()
        serial_crop = self._crop_roi(img_bgr, roi)
        
        # Add white border
        serial_crop = cv2.copyMakeBorder(
            serial_crop, 10, 10, 10, 10, 
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        
        # Preprocess
        gray = cv2.cvtColor(serial_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        
        # Otsu's binarization for cleaner digits
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Ensure black text on white background for Tesseract
        if float(np.mean(gray)) < 127:
            gray = 255 - gray
        
        # Use Tesseract if available (better for digits with whitelist)
        if TESSERACT_AVAILABLE:
            try:
                # PSM 6 (Uniform text block)
                config = f"--oem 1 --psm 6 -c tessedit_char_whitelist={WL_DIGITS}"
                txt = pytesseract.image_to_string(Image.fromarray(gray), lang="eng", config=config)
                digits = re.sub(r"[^0-9]", "", txt)
                if digits:
                    if len(digits) > 4:
                        digits = digits[-4:]
                    return digits.lstrip("0") or digits
            except Exception:
                pass
        
        # Use Tamil OCR and extract digits
        try:
            # Convert grayscale to BGR for Tamil OCR
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            result = self.ocr.predict(gray_bgr)
            if result and isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    text = " ".join(str(item) for item in result[0] if item)
                else:
                    text = str(result[0]) if result[0] else ""
                digits = re.sub(r"[^0-9]", "", text)
                if digits:
                    if len(digits) > 4:
                        digits = digits[-4:]
                    return digits.lstrip("0") or digits
        except Exception:
            pass
        
        return ""
    
    # ==================== Full OCR ====================
    
    def _run_tamil_ocr(self, image_path: Path, img_bgr: np.ndarray) -> List[str]:
        """
        Run Tamil OCR on image and return lines.
        
        Tamil OCR provides better accuracy for Tamil text compared to Tesseract.
        Tamil OCR with detect=True returns a list of detected words.
        We combine them into a single text blob and parse as lines.
        """
        try:
            # Tamil OCR expects the image path or numpy array
            result = self.ocr.predict(str(image_path))
            
            # Process result - Tamil OCR returns [[word1, word2, ...]]
            lines = []
            raw_words = []  # Keep raw words for alternative parsing
            
            if result:
                if isinstance(result, list):
                    if len(result) > 0:
                        # Result is typically [[word1, word2, ...]]
                        if isinstance(result[0], list):
                            # Get all words
                            raw_words = [str(w).strip() for w in result[0] if w and str(w).strip()]
                            if raw_words:
                                # Create combined text
                                combined_text = " ".join(raw_words)
                                
                                # Also store raw words as a special "raw" line for direct parsing
                                lines.append("__RAW__:" + "|".join(raw_words))
                                
                                # Try to reconstruct lines based on field markers
                                field_markers = [
                                    r"பெயர்",        # Name
                                    r"தந்தை",        # Father
                                    r"தாய்",         # Mother
                                    r"கணவர்",        # Husband
                                    r"வீட்டு",       # House
                                    r"எண்",          # Number
                                    r"வயது",        # Age
                                    r"பாலினம்",      # Gender

                                ]
                                
                                # Split by field markers while keeping the markers
                                pattern = r'(' + '|'.join(field_markers) + r')'
                                parts = re.split(pattern, combined_text, flags=re.IGNORECASE | re.UNICODE)
                                
                                # Reconstruct lines by pairing markers with their values
                                current_line = ""
                                for part in parts:
                                    part = part.strip()
                                    if not part:
                                        continue
                                    if re.search(r'^(' + '|'.join(field_markers) + r')', part, re.IGNORECASE | re.UNICODE):
                                        # This is a field marker - start a new line
                                        if current_line:
                                            lines.append(current_line.strip())
                                        current_line = part
                                    else:
                                        # This is a value - append to current line
                                        current_line += " " + part
                                
                                if current_line:
                                    lines.append(current_line.strip())
                                
                                # Also add the combined text as a line for fallback parsing
                                lines.append(combined_text)
                        else:
                            # Simple list of words
                            for item in result:
                                if item and str(item).strip():
                                    lines.append(str(item).strip())
                elif isinstance(result, str):
                    lines = [line.strip() for line in result.split('\n') if line.strip()]
            
            if self.dump_raw_ocr:
                self.log_debug(f"Tamil OCR result for {image_path.name}: {lines}")
            
            return lines
            
        except Exception as e:
            self.log_debug(f"Tamil OCR error for {image_path.name}: {e}")
            # Fallback to Tesseract if available
            if TESSERACT_AVAILABLE:
                return self._run_tesseract_fallback(image_path)
            return []
    
    def _run_tesseract_fallback(self, image_path: Path) -> List[str]:
        """Fallback to Tesseract OCR if Tamil OCR fails."""
        try:
            img = Image.open(image_path)
            config = "--oem 1 --psm 3"
            ocr_data = pytesseract.image_to_data(
                img,
                lang=self.languages,
                config=config,
                output_type=Output.DICT
            )
            img.close()
            return self._reconstruct_lines_from_tesseract(ocr_data)
        except Exception:
            return []
    
    def _reconstruct_lines_from_tesseract(self, ocr_data: Dict[str, Any]) -> List[str]:
        """Reconstruct text lines from Tesseract OCR data."""
        n = len(ocr_data.get("text", []))
        lines_dict: Dict[Tuple[int, int, int], List[Tuple[int, str]]] = {}
        
        for i in range(n):
            txt = (ocr_data["text"][i] or "").strip()
            if not txt:
                continue
            
            block = int(ocr_data.get("block_num", [0] * n)[i])
            par = int(ocr_data.get("par_num", [0] * n)[i])
            line = int(ocr_data.get("line_num", [0] * n)[i])
            x = int(ocr_data["left"][i])
            
            key = (block, par, line)
            if key not in lines_dict:
                lines_dict[key] = []
            lines_dict[key].append((x, txt))
        
        result = []
        for key in sorted(lines_dict.keys()):
            words = sorted(lines_dict[key], key=lambda t: t[0])
            line_text = " ".join(w[1] for w in words)
            result.append(line_text)
        
        return result
    
    # ==================== Field Extraction ====================
    
    def _normalize_line(self, line: str) -> str:
        """Normalize line for matching."""
        s = line.strip()
        s = s.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
        # Tamil OCR sometimes outputs 'ஃ' or other chars as separator - treat them as colon
        s = s.replace("ஃ", ":").replace(";", ":").replace(",", ":")
        s = re.sub(r"\s+", " ", s)
        return s
    
    def _clean_extracted_value(self, value: str) -> str:
        """Clean extracted value - remove leading colons, Tamil chars in English, etc."""
        if not value:
            return ""
        
        # Issue 1: Remove "தாயின்" or similar relation markers/labels from anywhere
        # Aggressive removal for possessives/labels unlikely to be part of names
        aggressive_words = ["தாயின்", "தந்தையின்", "கணவரின்", "வீட்டு", "வயது", "பாலினம்"]
        for word in aggressive_words:
            if word in value:
                value = re.sub(rf"{word}", " ", value)
        
        # Word-boundary removal for words that might be part of names (e.g. தாய் in காத்தாயி)
        # Only remove if they are standalone words
        bounded_words = ["தாய்", "தந்தை", "கணவர்", "பெயர்", "பெயர", "எண்", 
                         "Name", "Father", "Mother", "Husband", "Age", "Gender", "Photo"]
        
        for word in bounded_words:
            if word in value:
                # Use \b for word boundaries (works for Tamil in Python 3 regex)
                # Case-insensitive for English words
                value = re.sub(rf"\b{word}\b", " ", value, flags=re.IGNORECASE | re.UNICODE)

        # Remove leading colons, dashes, and special chars
        value = re.sub(r"^[:\-–—;,ஃ\s]+", "", value)
        
        # Remove trailing colons, dashes, and special chars
        value = re.sub(r"[:\-–—;,ஃ\s]+$", "", value)

        # Issue 2: Keep only Tamil, English characters, spaces
        # Replace other characters with space to separate words
        value = re.sub(r"[^a-zA-Z\u0B80-\u0BFF\s]", " ", value)
        
        # Collapse multiple spaces
        value = re.sub(r"\s+", " ", value).strip()
        
        # If the language context is English, remove Tamil characters
        # Check if value is predominantly English (Latin chars)
        has_english = len(re.findall(r'[A-Za-z]', value))
        has_tamil = len(re.findall(r'[\u0B80-\u0BFF]', value))
        
        # If it has English letters and some Tamil mixed in, keep only English
        if has_english > 0 and has_tamil > 0:
            # Check ratio - if more English than Tamil, filter out Tamil
            if has_english >= has_tamil:
                value = re.sub(r'[\u0B80-\u0BFF]', '', value)
                value = re.sub(r'\s+', ' ', value).strip()
        
        # If it's purely Tamil (no English), it might be a placeholder like இடன் (blank)
        # These should be returned as empty
        if has_tamil > 0 and has_english == 0:
            # Check for common Tamil placeholder words
            tamil_placeholders = ["இடன்", "நடது", "கடம்", "வெற்று"]
            if any(p in value for p in tamil_placeholders):
                return ""
        
        return value.strip()
    
    def _extract_value_after_colon(self, line: str, label_pattern: str) -> str:
        """Extract value after label and separator."""
        norm_line = self._normalize_line(line)
        
        # Try pattern with separator (colon, dash, space etc.) - more flexible for Tamil OCR output
        # Allow any separator: :, -, space, or nothing after label
        pattern = rf"(?:{label_pattern})\s*[:\-–—\s]*(.+?)(?:\s*[\-–—]\s*$|\s*$)"
        match = re.search(pattern, norm_line, re.IGNORECASE | re.UNICODE)
        
        if match:
            value = match.group(1).strip()
            value = re.sub(r"\s*[\-–—]\s*$", "", value)

            return value.strip()
        
        return ""
    
    def _extract_name(self, lines: List[str]) -> str:
        """Extract name from lines."""
        # First, try to extract from raw words if available
        for line in lines:
            if line.startswith("__RAW__:"):
                words = line[8:].split("|")
                name = self._extract_name_from_words(words)
                if name:
                    # Clean the extracted name
                    name = self._clean_extracted_value(name)
                    if name:
                        return name
        
        # Patterns for name label - flexible to handle Tamil OCR variations
        name_patterns = [r"பெயர்", r"name\b"]
        exclude_patterns = [r"தந்தை", r"தாய்", r"கணவர்", r"father", r"mother", r"husband"]
        
        for line in lines:
            if line.startswith("__RAW__:"):
                continue
            norm = self._normalize_line(line)
            norm_lower = norm.lower()
            
            # Skip if this is a relation line (father/mother/husband name)
            if any(re.search(p, norm_lower, re.IGNORECASE | re.UNICODE) for p in exclude_patterns):
                continue
            
            for pattern in name_patterns:
                if re.search(pattern, norm, re.IGNORECASE | re.UNICODE):
                    # Extract value after the label
                    value = self._extract_value_after_colon(line, r"(?:பெயர்|name)")
                    if value:
                        # Clean up any additional noise
                        value = re.sub(r"^[:\-–—\s]+", "", value)

                        # Remove house number or other fields that might have leaked in
                        value = re.sub(r"\s*-\s*வீட்டு.*$", "", value)  # Remove - வீட்டு... (house no pattern)
                        value = re.sub(r"\s*எண்.*$", "", value)
                        value = re.sub(r"\s*வயது.*$", "", value)
                        value = re.sub(r"\s*பாலினம்.*$", "", value)
                        # Remove any trailing dash/hyphen pattern
                        value = re.sub(r"\s*[-–—]\s*$", "", value)
                        # Clean pipe characters
                        value = value.replace("|", " ").strip()
                        # Apply final cleaning
                        value = self._clean_extracted_value(value)
                        if value and len(value) > 1:
                            return value
        
        return ""
    
    def _extract_name_from_words(self, words: List[str]) -> str:
        """Extract name from raw word list - first பெயர் or Name marker."""
        # Pattern: பெயர்; பழனிவேல் or Name: John or name: John
        # OCR output example: ['Name', ':', 'Rangaraj', 'Father', 'Name:', 'Makali', ...]
        # Or: ['Name', ':Lakshmi', 'Husband', 'Name:', 'Rangaraj', ...]
        # We want to get 'Rangaraj' or 'Lakshmi' (the first name after first 'Name' marker)
        
        # Markers that indicate we've passed to relation section
        relation_markers = ["father", "mother", "husband", "தந்தை", "தாய்", "கணவர்"]
        skip_markers = ["வீட்டு", "house", "age", "gender", "எண்", "வயது", "பாலினம்"]
        
        for i, word in enumerate(words):
            word_clean = word.strip()
            word_lower = word_clean.lower()
            
            # Check if this word contains பெயர் (Tamil name marker) or "name" (English marker)
            is_tamil_name_marker = "பெயர்" in word_clean or "பெயர" in word_clean
            is_english_name_marker = word_lower == "name" or word_lower == "name:" or word_lower.startswith("name:")
            
            if is_tamil_name_marker or is_english_name_marker:
                # Look for the name value
                # Skip colon if it's a separate token
                name_start_idx = i + 1
                if name_start_idx >= len(words):
                    continue
                    
                next_word = words[name_start_idx].strip()
                
                # Handle case where colon is separate: ['Name', ':', 'John', 'Father', ...]
                if next_word in [":", "-", ";", ","]:
                    name_start_idx += 1
                    if name_start_idx >= len(words):
                        continue
                    next_word = words[name_start_idx].strip()
                
                # Handle ":Lakshmi" case - name attached to colon
                if next_word.startswith(":"):
                    name_val = next_word.lstrip(":").strip()
                    if name_val and not any(m in name_val.lower() for m in skip_markers + relation_markers):
                        # This is the name! Clean and return
                        name_val = re.sub(r'^[:\-–—;,ஃ\s]+', '', name_val)
                        if name_val:
                            return name_val.strip()
                    name_start_idx += 1
                    if name_start_idx >= len(words):
                        continue
                    next_word = words[name_start_idx].strip()
                
                # Check if next word is a relation marker - if so, skip this Name marker
                # because this is "Father Name:" or "Husband Name:", not the person's name
                next_word_lower = next_word.lower().rstrip(":,;")
                if next_word_lower in relation_markers or any(rm in next_word_lower for rm in relation_markers):
                    continue  # This is a relation prefix like "Husband" before "Name:"
                
                # Now check if the word at name_start_idx is valid
                name_word = next_word.rstrip(",;:ஃ%&")
                name_word = name_word.lstrip(":").strip()  # Remove leading colon
                
                # Skip if it's a marker or empty
                if not name_word or any(m in name_word.lower() for m in skip_markers + relation_markers):
                    continue
                
                # Check if it's a valid name (has letters)
                has_english = re.search(r'[A-Za-z]', name_word)
                has_tamil = re.search(r'[\u0B80-\u0BFF]', name_word)
                
                if has_english or has_tamil:
                    # For English names, try to get full name (multiple words until we hit a marker)
                    if has_english and not has_tamil:
                        full_name = name_word
                        for j in range(name_start_idx + 1, min(name_start_idx + 4, len(words))):
                            extra_word = words[j].strip().rstrip(",;:ஃ%&")
                            extra_lower = extra_word.lower()
                            # Stop if we hit a marker
                            if any(m in extra_lower for m in skip_markers + relation_markers):
                                break
                            if extra_word.startswith("-") or extra_word.startswith(":"):
                                break
                            if extra_word and re.match(r'^[A-Za-z]', extra_word):
                                full_name += " " + extra_word
                            else:
                                break
                        # Clean the name before returning
                        full_name = re.sub(r'^[:\-–—;,ஃ\s]+', '', full_name)
                        return full_name.strip()
                    # Tamil name - try to get full name (2-3 words like ராட்ாம செல்வம், ஹாஜா நிஜா மைஹைதீன)
                    full_tamil_name = name_word
                    for j in range(name_start_idx + 1, min(name_start_idx + 4, len(words))):
                        extra_word = words[j].strip().rstrip(",;:ஃ%&")
                        extra_lower = extra_word.lower()
                        # Stop if we hit a marker
                        if any(m in extra_lower for m in skip_markers + relation_markers):
                            break
                        if extra_word.startswith("-") or extra_word.startswith(":"):
                            break
                        # Check if it's a Tamil word (has Tamil characters)
                        if extra_word and re.search(r'[\u0B80-\u0BFF]', extra_word):
                            # Skip if it's a label word like பெயர்
                            if any(label in extra_word for label in ["பெயர்", "பெயர", "எண்", "வயது", "பாலினம்", "வீட்டு", "வட்டு", "ட்டு"]):
                                break
                            full_tamil_name += " " + extra_word
                        else:
                            break
                    # Clean the name before returning
                    full_tamil_name = re.sub(r'^[:\-–—;,ஃ\s]+', '', full_tamil_name)
                    return full_tamil_name.strip()
        
        return ""
    
    def _extract_relation(self, lines: List[str]) -> Tuple[str, str]:
        """Extract relation type and name."""
        # First, try to extract from raw words if available
        for line in lines:
            if line.startswith("__RAW__:"):
                words = line[8:].split("|")
                rtype, rname = self._extract_relation_from_words(words)
                if rname:
                    rname = self._clean_extracted_value(rname)
                return rtype, rname
        
        # Fallback to line-based extraction
        for line in lines:
            if line.startswith("__RAW__:"):
                continue
            norm = self._normalize_line(line)
            
            for rel_type, patterns in RELATION_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, norm, re.IGNORECASE | re.UNICODE):
                        label_pattern = r"(?:" + "|".join(patterns) + r")"
                        value = self._extract_value_after_colon(line, label_pattern)
                        
                        # Clean up residual label text
                        value = re.sub(r"^பெயர்\s*[:\-–—]?\s*", "", value)
                        value = re.sub(r"^name\s*[:\-–—]?\s*", "", value, flags=re.IGNORECASE)
                        value = re.sub(r"^[:\-–—\s]+", "", value)
                        
                        # Remove trailing content that's not part of the name

                        value = re.sub(r"\s*எண்.*$", "", value)
                        value = re.sub(r"\s*வீட்டு.*$", "", value)
                        value = re.sub(r"\s*வட்டு.*$", "", value)
                        value = re.sub(r"\s+ட்டு.*$", "", value)
                        value = re.sub(r"\s*வயது.*$", "", value)
                        value = re.sub(r"\s*பாலினம்.*$", "", value)
                        value = value.strip()
                        # Apply final cleaning to remove leading colons
                        value = self._clean_extracted_value(value)
                        
                        # Skip if the extracted value is just a label word (e.g., "பயர்" corrupted from "பெயர்")
                        name_label_words = ["பெயர்", "பெயர", "பயர்", "பயர", "யர்", "யர", "name", "பேர்", "பேர"]
                        is_label = any(label.lower() in value.lower() for label in name_label_words) if value else True
                        is_too_short = len(value) <= 3 if value else True
                        
                        if value and len(value) > 1 and not (is_label and is_too_short):
                            return rel_type, value
        
        return "", ""
    
    def _extract_relation_from_words(self, words: List[str]) -> Tuple[str, str]:
        """Extract relation from raw word list - handles both Tamil and English OCR output."""
        # Pattern: ... கணவர் பெயர்; ராமசாமி ... (Tamil)
        # or: ... Father's Name: John ... (English)
        # or: ... father name: John ... (English)
        
        relation_type = ""
        relation_name = ""
        
        # Find relation type markers (Tamil and English)
        relation_markers = {
            "father": ["தந்தை", "தந்தையின்", "father", "father's", "fathers"],
            "mother": ["தாய்", "தாயின்", "mother", "mother's", "mothers"],
            "husband": ["கணவர்", "கணவரின்", "husband", "husband's", "husbands"],
        }
        
        # First, find if there's a relation type in the words
        found_relation_idx = -1
        for i, word in enumerate(words):
            word_clean = word.strip().rstrip(",;:ஃ%&").lower()
            for rel_type, markers in relation_markers.items():
                for marker in markers:
                    if marker.lower() in word_clean or word_clean == marker.lower():
                        relation_type = rel_type
                        found_relation_idx = i
                        break
                if relation_type:
                    break
            if relation_type:
                break
        
        # Skip markers for both Tamil and English
        skip_markers_tamil = ["வீட்டு", "வட்டு", "ட்டு", "எண்", "வயது", "பாலினம்"]
        skip_markers_english = ["house", "age", "gender", "address", "no", "number"]
        skip_markers = skip_markers_tamil + skip_markers_english
        
        # Label words that should NOT be extracted as names (e.g., "பெயர்" = "name")
        # These often get corrupted by OCR to "பயர்", "யர்", etc.
        name_label_words = ["பெயர்", "பெயர", "பயர்", "பயர", "யர்", "யர", "name", "பேர்", "பேர"]
        
        def is_name_label(word: str) -> bool:
            """Check if word looks like a label word (not an actual name)."""
            word_clean = word.strip().rstrip(",;:ஃ%&").lower()
            # Check exact match or near match for name labels
            for label in name_label_words:
                if word_clean == label.lower() or label.lower() in word_clean:
                    return True
            # Very short words that are likely corrupted labels
            if len(word_clean) <= 3 and re.search(r'[\u0B80-\u0BFF]', word_clean):
                return True
            return False
        
        # If we found a relation type, look for the name after the name marker
        if relation_type and found_relation_idx >= 0:
            # Look for பெயர் or "name" after the relation marker
            for i in range(found_relation_idx + 1, len(words)):
                word = words[i].strip()
                word_lower = word.lower()
                is_tamil_name = "பெயர்" in word or "பெயர" in word
                is_english_name = "name" in word_lower
                
                if is_tamil_name or is_english_name:
                    # The next word(s) should be the name
                    # First check for name attached after colon like "பெயர்:மணி"
                    if ":" in word:
                        name_after_colon = word.split(":", 1)[-1].strip().rstrip(",;ஃ%&-")
                        if name_after_colon and re.search(r'[\u0B80-\u0BFF]', name_after_colon):
                            relation_name = name_after_colon.strip()
                            break
                    
                    if i + 1 < len(words):
                        name_word = words[i + 1].strip()
                        
                        # Handle case where name starts with colon ":மணி"
                        name_word = name_word.lstrip(":").strip()
                        name_word = name_word.rstrip(",;:ஃ%&")
                        
                        # Skip dash-only words, try next word
                        if name_word == "-" or name_word == "–" or name_word == "—":
                            if i + 2 < len(words):
                                name_word = words[i + 2].strip().lstrip(":").rstrip(",;:ஃ%&")
                        
                        # Skip if it's another marker OR if it's a label word
                        if name_word and not any(m.lower() in name_word.lower() for m in skip_markers) and not is_name_label(name_word):
                            # For English names, try to get full name (multiple words)
                            has_english = re.search(r'[A-Za-z]', name_word)
                            has_tamil = re.search(r'[\u0B80-\u0BFF]', name_word)
                            
                            if has_english and not has_tamil:
                                full_name = name_word
                                for j in range(i + 2, min(i + 5, len(words))):
                                    extra_word = words[j].strip().rstrip(",;:ஃ%&")
                                    if any(m.lower() in extra_word.lower() for m in skip_markers):
                                        break
                                    if extra_word.startswith("-") or extra_word.startswith(":"):
                                        break
                                    if extra_word and re.match(r'^[A-Za-z]', extra_word):
                                        full_name += " " + extra_word
                                    else:
                                        break
                                # Clean leading colons from the name
                                full_name = re.sub(r'^[:\-–—;,ஃ\s]+', '', full_name)
                                relation_name = full_name.strip()
                            else:
                                # Tamil relation name - try to get full name (2-3 words)
                                full_tamil_rel = name_word
                                start_idx = i + 2
                                # If we skipped a dash, start from i + 3
                                if i + 1 < len(words) and words[i + 1].strip() in ["-", "–", "—"]:
                                    start_idx = i + 3
                                for k in range(start_idx, min(start_idx + 3, len(words))):
                                    extra_word = words[k].strip().rstrip(",;:ஃ%&")
                                    if any(m.lower() in extra_word.lower() for m in skip_markers):
                                        break
                                    if extra_word.startswith("-") or extra_word.startswith(":"):
                                        break
                                    # Check if it's a Tamil word
                                    if extra_word and re.search(r'[\u0B80-\u0BFF]', extra_word):
                                        # Skip label words
                                        if any(label in extra_word for label in ["பெயர்", "பெயர", "எண்", "வயது", "பாலினம்", "வீட்டு", "வட்டு", "ட்டு"]):
                                            break
                                        full_tamil_rel += " " + extra_word
                                    else:
                                        break
                                # Clean leading colons from Tamil name
                                full_tamil_rel = re.sub(r'^[:\-–—;,ஃ\s]+', '', full_tamil_rel)
                                relation_name = full_tamil_rel.strip()
                            break
            
            # If no name found after name marker, try direct value after relation type
            if not relation_name and found_relation_idx + 1 < len(words):
                # Check for pattern like "Father: John" or "Father's John"
                for i in range(found_relation_idx + 1, min(found_relation_idx + 4, len(words))):
                    word = words[i].strip().rstrip(",;:ஃ%&")
                    word_lower = word.lower()
                    # Skip "name" keyword, markers, and label words
                    if word_lower in ["name", "name:", "'s"] or any(m.lower() in word_lower for m in skip_markers):
                        continue
                    # Skip if it's a label word (e.g., "பயர்" = corrupted "பெயர்")
                    if is_name_label(word):
                        continue
                    # Check if it's a valid name (has letters, not a marker)
                    if word and re.match(r'^[A-Za-z\u0B80-\u0BFF]', word):
                        has_english = re.search(r'[A-Za-z]', word)
                        has_tamil = re.search(r'[\u0B80-\u0BFF]', word)
                        if has_english or has_tamil:
                            if has_english and not has_tamil:
                                full_name = word
                                for j in range(i + 1, min(i + 4, len(words))):
                                    extra_word = words[j].strip().rstrip(",;:ஃ%&")
                                    if any(m.lower() in extra_word.lower() for m in skip_markers):
                                        break
                                    if extra_word.startswith("-") or extra_word.startswith(":"):
                                        break
                                    if extra_word and re.match(r'^[A-Za-z]', extra_word):
                                        full_name += " " + extra_word
                                    else:
                                        break
                                # Clean leading colons from the name
                                full_name = re.sub(r'^[:\-–—;,ஃ\s]+', '', full_name)
                                relation_name = full_name.strip()
                            else:
                                # Tamil relation name - try to get full name (2-3 words)
                                full_tamil_rel = word
                                for k in range(i + 1, min(i + 4, len(words))):
                                    extra_word = words[k].strip().rstrip(",;:ஃ%&")
                                    if any(m.lower() in extra_word.lower() for m in skip_markers + ["name", "'s"]):
                                        break
                                    if extra_word.startswith("-") or extra_word.startswith(":"):
                                        break
                                    # Check if it's a Tamil word
                                    if extra_word and re.search(r'[\u0B80-\u0BFF]', extra_word):
                                        # Skip label words
                                        if any(label in extra_word for label in ["பெயர்", "பெயர", "எண்", "வயது", "பாலினம்", "வீட்டு"]):
                                            break
                                        full_tamil_rel += " " + extra_word
                                    else:
                                        break
                                # Clean leading colons from Tamil name
                                full_tamil_rel = re.sub(r'^[:\-–—;,ஃ\s]+', '', full_tamil_rel)
                                relation_name = full_tamil_rel.strip()
                            break
        
        # If no explicit relation type found, check if there's a second name pattern
        # (indicates relation name even without explicit type marker)
        if not relation_type:
            name_count = 0
            name_indices = []
            for i, word in enumerate(words):
                word_lower = word.lower()
                if "பெயர்" in word or "பெயர" in word or word_lower.startswith("name"):
                    name_count += 1
                    name_indices.append(i)
            
            # If there are two name markers, the second one is likely relation
            if name_count >= 2 and len(name_indices) >= 2:
                second_name_idx = name_indices[1]
                if second_name_idx + 1 < len(words):
                    name_word = words[second_name_idx + 1].strip().rstrip(",;:ஃ%&")
                    if name_word and not any(m.lower() in name_word.lower() for m in skip_markers):
                        # For English names, get full name
                        has_english = re.search(r'[A-Za-z]', name_word)
                        has_tamil = re.search(r'[\u0B80-\u0BFF]', name_word)
                        
                        if has_english and not has_tamil:
                            full_name = name_word
                            for j in range(second_name_idx + 2, min(second_name_idx + 5, len(words))):
                                extra_word = words[j].strip().rstrip(",;:ஃ%&")
                                if any(m.lower() in extra_word.lower() for m in skip_markers):
                                    break
                                if extra_word.startswith("-") or extra_word.startswith(":"):
                                    break
                                if extra_word and re.match(r'^[A-Za-z]', extra_word):
                                    full_name += " " + extra_word
                                else:
                                    break
                            relation_name = full_name.strip()
                        else:
                            relation_name = name_word
                        # Default to father if no type specified
                        relation_type = "father"
        
        # Clean the relation_name before returning
        if relation_name:
            # Remove leading colons, dashes, etc.
            relation_name = re.sub(r'^[:\-–—;,ஃ\s]+', '', relation_name)
            # Remove trailing artifacts that might have slipped through
            relation_name = re.sub(r"\s*வீட்டு.*$", "", relation_name)
            relation_name = re.sub(r"\s*வட்டு.*$", "", relation_name)
            relation_name = re.sub(r"\s+ட்டு.*$", "", relation_name)
            relation_name = relation_name.strip()
        
        return relation_type, relation_name
    
    def _extract_house(self, lines: List[str], img_bgr: np.ndarray) -> str:
        """Extract house number from lines or ROI fallback."""
        # First, try to extract from raw words if available
        for line in lines:
            if line.startswith("__RAW__:"):
                words = line[8:].split("|")
                house = self._extract_house_from_words(words)
                if house:
                    # Convert Tamil digits and clean
                    house = convert_tamil_digits(house)
                    return self._clean_house_number(house)
        
        combined_pattern = r"(?:" + "|".join(HOUSE_PATTERNS) + r")"
        
        for line in lines:
            if line.startswith("__RAW__:"):
                continue
            norm = self._normalize_line(line)
            
            if re.search(combined_pattern, norm, re.IGNORECASE | re.UNICODE):
                # Try direct extraction with single colon/separator
                # Pattern: வீட்டு எண் : 1 or house no : 1
                direct_match = re.search(
                    rf"(?:{combined_pattern})\s*[:\-–—]\s*([A-Za-z]?\d[\dA-Za-z/\-]*)",
                    norm,
                    re.IGNORECASE | re.UNICODE
                )
                if direct_match:
                    house_val = direct_match.group(1).strip()
                    # Clean up any trailing noise

                    house_val = convert_tamil_digits(house_val.strip())
                    cleaned = self._clean_house_number(house_val)
                    if cleaned and len(cleaned) <= 20:
                        return cleaned
                
                # Fallback to _extract_value_after_colon
                value = self._extract_value_after_colon(line, combined_pattern)

                value = re.sub(r"\s+", "", value)
                value = convert_tamil_digits(value)
                
                house_match = re.search(r"^([A-Za-z]?\d[\dA-Za-z/\-]{0,15})", value)
                if house_match:
                    return house_match.group(1)
                
                cleaned = re.sub(r"[^\dA-Za-z/\-]", "", value)
                cleaned = self._clean_house_number(cleaned)
                if cleaned and len(cleaned) <= 20:
                    return cleaned
        
        # Fallback to ROI extraction
        return self._extract_house_from_roi(img_bgr)
    
    def _extract_house_from_words(self, words: List[str]) -> str:
        """Extract house number from raw word list."""
        # Pattern: ... வீட்டு ... எண்ஃ1 ... or எண் 1 or எண்:1
        # Look for எண் followed by a number AFTER வீட்டு marker
        
        found_veedu = False  # வீட்டு marker
        
        for i, word in enumerate(words):
            word_clean = word.strip()
            
            # Track if we've seen வீட்டு (house) marker
            if "வீட்டு" in word_clean:
                found_veedu = True
                continue
            
            # Only look for எண் (number) AFTER வீட்டு marker
            if found_veedu and ("எண்" in word_clean or "எண" in word_clean):
                # Try to extract number from the same word (e.g., எண்ஃ1, எண்:1)
                num_match = re.search(r"எண்?\s*[ஃ;:,&]?\s*(\d+[A-Za-z/\-]*)", word_clean)
                if num_match:
                    house_val = num_match.group(1)
                    # Validate - house numbers are typically short
                    if len(house_val) <= 20:
                        return house_val
                
                # Try to extract from next few words (sometimes OCR splits the number)
                # Look ahead up to 3 words
                extracted_parts = []
                for k in range(1, 4):
                    if i + k < len(words):
                        next_word = words[i + k].strip()
                        # Stop if it hits another field marker
                        if any(marker in next_word for marker in ["எண்", "வயது", "பாலினம்", "Photo", "Age", "Gender", "Name"]):
                            break
                        
                        # Apply basic cleaning
                        clean_next = next_word.strip(".,:;-")
                        if not clean_next: 
                             continue
                             
                        # Check if it looks like part of a house number (digits/letters)
                        if re.match(r"^[A-Z0-9/\-]+$", clean_next, re.IGNORECASE):
                             extracted_parts.append(clean_next)
                        else:
                             break
                    else:
                        break
                
                if extracted_parts:
                    # Combine parts found
                    full_val = "".join(extracted_parts) if len(extracted_parts) == 1 else "-".join(extracted_parts)
                    # Use stricter check on the combined string
                    if len(full_val) < 20:
                         return full_val
        
        # Also look for standalone number pattern after வீட்டுஎண் combined
        for i, word in enumerate(words):
            word_clean = word.strip()
            if "வீட்டுஎண்" in word_clean or "வீட்டு" in word_clean or "house" in word_clean.lower():
                # Check if house number is embedded like வீட்டுஎயின் or similar with number
                num_match = re.search(r"[:\-–—\s]+([A-Za-z0-9/\-]+)", word_clean)
                if num_match:
                    house_val = num_match.group(1)
                    if len(house_val) <= 20:
                        return house_val
        
        return ""
    
    def _extract_house_from_roi(self, img_bgr: np.ndarray) -> str:
        """Extract house number from ROI as fallback using Tesseract."""
        roi = self.ocr_config.house_roi.as_tuple()
        house_crop = self._crop_roi(img_bgr, roi)

        # Add white border
        house_crop = cv2.copyMakeBorder(
            house_crop, 10, 10, 10, 10, 
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        
        gray = cv2.cvtColor(house_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        # Use Otsu's binarization for consistency
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if float(np.mean(gray)) < 127:
            gray = 255 - gray
        
        txt = ""
        
        # Use Tesseract for house number (alphanumeric with whitelist) if available
        if TESSERACT_AVAILABLE:
            try:
                # PSM 6 for better block recognition
                config = f"--oem 1 --psm 6 -c tessedit_char_whitelist={WL_HOUSE}"
                txt = pytesseract.image_to_string(Image.fromarray(gray), lang="eng", config=config)
            except Exception:
                pass
        
        # Fallback to Tamil OCR
        if not txt:
            try:
                # Convert grayscale to BGR for Tamil OCR
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                result = self.ocr.predict(gray_bgr)
                if result and isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        txt = " ".join(str(item) for item in result[0] if item)
                    else:
                        txt = str(result[0]) if result[0] else ""
            except Exception:
                txt = ""
        
        # Convert Tamil digits first
        txt = convert_tamil_digits(txt)
        
        # Clean house value - preserve Tamil letters, Latin letters and numbers
        s = txt.upper()
        # Preserve Tamil characters (\u0B80-\u0BFF) along with A-Z, 0-9, /, -, and spaces
        s = re.sub(r"[^A-Z0-9\u0B80-\u0BFF/\-\s]", " ", s)
        
        # Clean the result using the smarter function
        return self._clean_house_number(s)
    
    def _clean_house_number(self, raw_text: str) -> str:
        """
        Clean and correct OCR'd house number text.
        
        Requirements:
        - Must contain at least one digit
        - Can contain Tamil letters, Latin letters, digits
        - Allowed special characters: /, -
        - Fixes obvious OCR digit errors (O->0, I->1)
        - Preserves alphanumeric patterns like "11-A", "D35-1", "6வ283"
        """
        if not raw_text:
            return ""
        
        # Basic cleanup of noise chars that are definitely not part of house no
        noise_chars = ' .,:;()[]{}\'"'
        text = raw_text.strip(noise_chars)
        
        # Split by likely separators to handle "House No: 123" if extracted loosely
        # We assume the extraction logic passed us the value part, but just in case
        tokens = text.split()
        
        # Use the last token if multiple are present, or try to find the best candidate
        # But 'process as is' suggests we should try to keep what we found.
        # Let's clean the longest token that looks alphanumeric.
        
        best_token = text # Default to full text if splitting is ambiguous
        
        if len(tokens) > 1:
            # Pick the token with digits
            for t in tokens:
                if any(c.isdigit() for c in t):
                     best_token = t
                     break
        
        # Apply digit fixes (O->0, I->1)
        fixed_token = self._fix_ocr_digits(best_token)
        
        # Convert to uppercase for Latin letters (Tamil characters remain unchanged)
        # This ensures both 'a' and 'A' are treated the same
        fixed_token = fixed_token.upper()
        
        # Check if it matches a broad house number pattern
        # Allowed: Uppercase Letters, Tamil Letters (Unicode \u0B80-\u0BFF), Digits, /, -
        # Pattern allows Tamil characters in addition to A-Z
        if re.match(r"^[A-Z0-9\u0B80-\u0BFF/\-]+$", fixed_token):
            # Must contain at least one digit
            if any(c.isdigit() for c in fixed_token):
                return fixed_token
            else:
                # No digits found, invalid house number
                return ""
        
        # If not matching pattern, try to strip non-allowed chars
        # Keep Tamil characters, Latin letters (both cases), digits, /, -
        # Include both A-Z and a-z to catch any remaining lowercase before final uppercase conversion
        cleaned = re.sub(r"[^A-Za-z0-9\u0B80-\u0BFF/\-]", "", fixed_token)
        # Convert to uppercase for consistency
        cleaned = cleaned.upper()
        
        # Ensure at least one digit is present
        if cleaned and any(c.isdigit() for c in cleaned):
            return cleaned
        else:
            # No digits found, invalid house number
            return ""
    
    def _contains_numeric(self, text: str) -> bool:
        """
        Check if text contains any numeric characters.
        
        Args:
            text: String to check
            
        Returns:
            True if text contains at least one digit, False otherwise
        """
        if not text:
            return False
        return any(c.isdigit() for c in text)
    
    def _is_all_non_numeric(self, text: str) -> bool:
        """
        Check if text contains ONLY non-numeric characters (no digits at all).
        
        Args:
            text: String to check
            
        Returns:
            True if text has no digits, False if it has any digits
        """
        if not text or not text.strip():
            return False
        return not any(c.isdigit() for c in text)
    
    def _is_valid_house_number(self, house_no: str) -> bool:
        """
        Validate if a house number is valid.
        
        A valid house number must:
        - Not be empty
        - Contain at least one digit (to avoid pure Tamil text like "வீட்டு")
        - Only contain allowed characters including Tamil letters
        
        Examples of valid formats:
        - "2", "2A", "2-12", "2/12"
        - "6 (2)", "2 (A)", "10 (1A)"
        - "1ஏ", "5ஏ" (Tamil letter suffix)
        - "1,ராஜபாளையம்" (Tamil place name with comma)
        
        Returns:
            True if valid, False otherwise
        """
        if not house_no or not house_no.strip():
            return False
        
        # Must contain at least one digit
        if not any(c.isdigit() for c in house_no):
            return False
        
        # Should only contain alphanumeric (including Tamil letters), -, /, spaces, parentheses, and commas
        # Tamil Unicode range: \u0B80-\u0BFF
        if not re.match(r"^[A-Za-z0-9\u0B80-\u0BFF/\-\s\(\),]+$", house_no.strip()):
            return False
        
        return True
    
    def _fix_ocr_digits(self, text: str) -> str:
        """
        Fix common OCR letter/digit confusions in text.
        
        Only converts O->0 and I->1 when they appear in positions
        that are likely meant to be digits (surrounded by digits).
        """
        if not text:
            return text
        
        result = []
        chars = list(text)
        
        for i, c in enumerate(chars):
            prev_is_digit = (i > 0 and chars[i-1].isdigit())
            next_is_digit = (i < len(chars) - 1 and chars[i+1].isdigit())
            
            # Only convert O->0 or I->1 if surrounded by or adjacent to digits
            if c == 'O' and (prev_is_digit or next_is_digit):
                result.append('0')
            elif c == 'I' and (prev_is_digit or next_is_digit):
                result.append('1')
            else:
                result.append(c)
        
        return "".join(result)
    
    def _extract_age(self, lines: List[str]) -> str:
        """Extract age from lines."""
        # First, try to extract from raw words if available
        for line in lines:
            if line.startswith("__RAW__:"):
                words = line[8:].split("|")
                age = self._extract_age_from_words(words)
                if age:
                    return age
        
        combined_pattern = r"(?:" + "|".join(AGE_PATTERNS) + r")"
        
        for line in lines:
            if line.startswith("__RAW__:"):
                continue
            norm = self._normalize_line(line)
            
            if re.search(combined_pattern, norm, re.IGNORECASE | re.UNICODE):
                age_match = re.search(
                    rf"(?:{combined_pattern})\s*[:\-–—]?\s*(\d{{1,3}})",
                    norm,
                    re.IGNORECASE | re.UNICODE
                )
                if age_match:
                    return age_match.group(1)
        
        # Also try to find any age value in the text (fallback)
        for line in lines:
            if line.startswith("__RAW__:"):
                continue
            norm = self._normalize_line(line)
            # Look for pattern like "79" after any text containing age indicator
            age_pattern = re.search(r"(?:வயது|age)\s*[:\-–—,]?\s*(\d{1,3})", norm, re.IGNORECASE | re.UNICODE)
            if age_pattern:
                age_val = age_pattern.group(1)
                if 1 <= int(age_val) <= 120:
                    return age_val
        
        return ""
    
    def _extract_age_from_words(self, words: List[str]) -> str:
        """Extract age from raw word list."""
        # Pattern: வயது,77 or வயது&77 or வயது:77 or வயது 77
        # Also handles English: age:77, Age: 77, age 77
        # OCR may output: Age இடன் 57 (with Tamil placeholder between Age and number)
        
        # Tamil placeholder words that may appear between Age and number
        tamil_placeholders = ["இடன்", "கடம்", "நடது", "வெற்று"]
        
        for i, word in enumerate(words):
            word_clean = word.strip()
            word_lower = word_clean.lower()
            
            # Check if this word contains வயது (Tamil age marker) or "age" (English marker)
            is_tamil_age = "வயது" in word_clean
            is_english_age = word_lower.startswith("age") or word_lower == "age" or word_lower == "age:"
            
            if is_tamil_age or is_english_age:
                # Try to extract age from the same word (e.g., வயது,77, வயது&77, age:77)
                # Handle various separators: , & ; : - and Tamil special chars
                if is_tamil_age:
                    age_match = re.search(r"வயது\s*[,&;:\-ஃ]?\s*(\d{1,3})", word_clean)
                else:
                    age_match = re.search(r"age\s*[,&;:\-ஃ]?\s*(\d{1,3})", word_clean, re.IGNORECASE)
                
                if age_match:
                    age_val = age_match.group(1)
                    if 1 <= int(age_val) <= 120:
                        return age_val
                
                # Look for age in next few words (skip Tamil placeholders)
                for j in range(i + 1, min(i + 4, len(words))):
                    next_word = words[j].strip().lstrip(",&;:-ஃ:")
                    
                    # Skip Tamil placeholder words
                    if any(p in next_word for p in tamil_placeholders):
                        continue
                    
                    # Check if this word contains a number
                    age_match = re.search(r"(\d{1,3})", next_word)
                    if age_match:
                        age_val = age_match.group(1)
                        if 1 <= int(age_val) <= 120:
                            return age_val
                    
                    # If it's not a number and not a placeholder, stop looking
                    if next_word and not re.search(r'[\u0B80-\u0BFF]', next_word):
                        break
        
        # Also look for age pattern anywhere in words (fallback)
        all_text = " ".join(words)
        # Try Tamil pattern first
        age_match = re.search(r"வயது\s*[,&;:\-ஃ]?\s*(\d{1,3})", all_text)
        if age_match:
            age_val = age_match.group(1)
            if 1 <= int(age_val) <= 120:
                return age_val
        
        # Try English pattern - allow Tamil placeholder words between Age and number
        # Pattern: Age [optional Tamil placeholder] number
        age_match = re.search(r"\bage\s*[,&;:\-ஃ:]?\s*(?:[\u0B80-\u0BFF]+\s*)?(\d{1,3})", all_text, re.IGNORECASE)
        if age_match:
            age_val = age_match.group(1)
            if 1 <= int(age_val) <= 120:
                return age_val
        
        return ""
    
    def _extract_gender(self, lines: List[str]) -> str:
        """Extract gender from lines."""
        combined_pattern = r"(?:" + "|".join(GENDER_PATTERNS) + r")"
        
        # First, look for gender in the combined text of all lines
        all_text = " ".join(lines)
        
        # Look for explicit gender words anywhere
        if re.search(r"ஆண்|ஆண\b", all_text):
            return "Male"
        if re.search(r"பெண்|பெண\b", all_text):
            return "Female"
        if re.search(r"\bmale\b", all_text, re.IGNORECASE):
            return "Male"
        if re.search(r"\bfemale\b", all_text, re.IGNORECASE):
            return "Female"
        
        for line in lines:
            norm = self._normalize_line(line)
            
            if re.search(combined_pattern, norm, re.IGNORECASE | re.UNICODE):
                # Look for gender value after the label
                gender_match = re.search(
                    rf"(?:{combined_pattern})\s*[:\-–—]?\s*(\S+)",
                    norm,
                    re.IGNORECASE | re.UNICODE
                )
                if gender_match:
                    raw_gender = gender_match.group(1).strip()
                    
                    # Map to standard gender values
                    for key, value in GENDER_MAP.items():
                        if key.lower() in raw_gender.lower():
                            return value
                    
                    # Check if it's a Tamil gender word
                    if "ஆண" in raw_gender:
                        return "Male"
                    if "பெண" in raw_gender:
                        return "Female"
        
        return ""
    
    def get_all_voters(self) -> List[Voter]:
        """Get all extracted voters as Voter models."""
        voters = []
        sequence_in_doc = 0
        
        for page_result in self.page_results:
            for idx, ocr_result in enumerate(page_result.records, start=1):
                if ocr_result.error:
                    continue
                
                sequence_in_doc += 1
                voter = ocr_result.to_voter(sequence_in_page=idx, sequence_in_document=sequence_in_doc)
                voter.page_id = page_result.page_id
                voter.sequence_in_document = sequence_in_doc
                voters.append(voter)
        
        return voters


def process_ocr(
    extracted_dir: Path,
    languages: str = "eng+tam",
    allow_next_line: bool = True,
) -> List[Voter]:
    """
    Convenience function to run OCR on an extracted folder.
    
    Args:
        extracted_dir: Path to extracted folder
        languages: Tesseract language codes
        allow_next_line: Allow value on next line
    
    Returns:
        List of extracted voters
    """
    from ..config import Config
    
    config = Config()
    context = ProcessingContext(config=config)
    context.setup_paths_from_extracted(extracted_dir)
    
    processor = OCRProcessor(
        context,
        languages=languages,
        allow_next_line=allow_next_line,
    )
    
    if not processor.run():
        raise OCRProcessingError("OCR processing failed", str(extracted_dir))
    
    return processor.get_all_voters()
