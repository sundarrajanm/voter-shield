"""
Missing Name Processor.

Identifies records with missing or invalid names, attempts to recover them via:
1. Re-running TamilOCR on the specific crop.
2. If OCR fails/is insufficient, sending the crop to AI for extraction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

import cv2
import numpy as np

from .base import BaseProcessor, ProcessingContext
from ..models import Voter, ProcessedDocument
from ..logger import get_logger
from .ocr_processor import OCRProcessor
from .ai_ocr_processor import AIOCRProcessor

logger = get_logger("missing_name_processor")

@dataclass
class MissingNameResult:
    """Result from missing name reprocessing."""
    total_records: int = 0
    missing_name_count: int = 0
    ocr_recovered_count: int = 0
    ai_recovered_count: int = 0
    failed_count: int = 0
    processing_time_sec: float = 0.0


class MissingNameProcessor(BaseProcessor):
    """
    Processes records with missing names by retrying OCR and falling back to AI.
    """
    
    name = "MissingNameProcessor"
    
    def __init__(
        self,
        context: ProcessingContext,
        document: ProcessedDocument,
    ):
        super().__init__(context)
        self.document = document
        self.result: Optional[MissingNameResult] = None
        
        # Initialize sub-processors
        # We need an OCR processor specifically configured for Tamil
        self.ocr_processor = OCRProcessor(
            self.context,
            use_tesseract=False, # Force TamilOCR
            use_cuda=True,
            languages="tam+eng"
        )
        
        # AI Processor for fallback
        self.ai_processor = AIOCRProcessor(self.context)
        
        # Initialization flags
        self._ocr_initialized = False
        self._ai_initialized = False

    def _ensure_processors_ready(self):
        """Lazy initialization of heavy engines."""
        if not self._ocr_initialized:
            try:
                self.ocr_processor._initialize_ocr()
                self.ocr = self.ocr_processor.ocr
                self._ocr_initialized = True
                self.log_info("OCR Engine initialized for reprocessing")
            except Exception as e:
                self.log_error(f"Failed to initialize OCR: {e}")
        
        if not self._ai_initialized:
            try:
                self.ai_processor._initialize_client()
                self._ai_initialized = True
                self.log_info("AI Client initialized for reprocessing")
            except Exception as e:
                self.log_error(f"Failed to initialize AI: {e}")

    def process(self) -> bool:
        """Process records with missing names or relation names."""
        start_time = time.perf_counter()
        
        if not self.document or not self.document.pages:
            self.log_warning("No document data to process")
            return False
        
        # Collect all voters
        all_voters: List[Voter] = []
        for page_data in self.document.pages:
            all_voters.extend(page_data.voters)
        
        self.log_info(f"Checking {len(all_voters)} voters for missing names/relations...")
        
        # Find voters with missing names OR missing relation names
        # Criteria: empty, None, or very short (len < 3)
        targets = []
        for v in all_voters:
            missing_name = not v.name or len(v.name.strip()) < 3
            missing_relation = not v.relation_name or len(v.relation_name.strip()) < 3
            
            if missing_name or missing_relation:
                targets.append(v)
        
        if not targets:
            self.log_info("No missing names or relation names found")
            self.result = MissingNameResult(
                total_records=len(all_voters),
                processing_time_sec=time.perf_counter() - start_time
            )
            return True
        
        self.log_info(f"Found {len(targets)} voters with missing/invalid names or relations")
        self._ensure_processors_ready()
        
        ocr_recovered = 0
        ai_recovered = 0
        failed = 0
        
        for voter in targets:
            missing_name = not voter.name or len(voter.name.strip()) < 3
            missing_relation = not voter.relation_name or len(voter.relation_name.strip()) < 3
            
            # Construct crop path
            if not self.context.crops_dir:
                 self.log_error("Crops directory not configured")
                 break
                 
            crop_path = self.context.crops_dir / voter.page_id / voter.image_file
            
            if not crop_path.exists():
                alt_path = self.context.crops_dir / voter.image_file
                if alt_path.exists():
                    crop_path = alt_path
                else:
                    self.log_warning(f"Crop image not found for voter {voter.serial_no}: {crop_path}")
                    failed += 1
                    continue
            
            # --- Attempt 1: OCR Retry ---
            recovered_any = False
            try:
                # Load image
                img = cv2.imdecode(np.fromfile(str(crop_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    # Run OCR
                    results = self.ocr.predict([img])
                    if results:
                        lines, _ = self.ocr_processor._parse_ocr_output(results[0])
                        
                        # Fix Name
                        if missing_name:
                            ocr_name = self.ocr_processor._extract_name(lines)
                            if ocr_name and len(ocr_name.strip()) > 2:
                                self.log_info(f"Refined name via OCR for {voter.serial_no}: '{ocr_name}' (was: '{voter.name}')")
                                voter.name = ocr_name
                                voter.processing_method = (voter.processing_method or "") + "+ocr_name"
                                recovered_any = True
                                missing_name = False # marked as fixed
                        
                        # Fix Relation Name
                        if missing_relation:
                            _, ocr_rel_name = self.ocr_processor._extract_relation(lines)
                            if ocr_rel_name and len(ocr_rel_name.strip()) > 2:
                                self.log_info(f"Refined relation via OCR for {voter.serial_no}: '{ocr_rel_name}'")
                                voter.relation_name = ocr_rel_name
                                voter.processing_method = (voter.processing_method or "") + "+ocr_rel"
                                recovered_any = True
                                missing_relation = False # marked as fixed

                        if recovered_any:
                            ocr_recovered += 1
            except Exception as e:
                self.log_debug(f"OCR retry failed for {voter.serial_no}: {e}")
            
            # If we fixed everything needed, skip AI
            if not missing_name and not missing_relation:
                continue

            # --- Attempt 2: AI Fallback ---
            # If still missing something, call AI
            if self._ai_initialized:
                try:
                    self.log_info(f"Falling back to AI for {voter.serial_no} (Needs: {'Name ' if missing_name else ''}{'Relation' if missing_relation else ''})")
                    # Single image call
                    ai_results = self.ai_processor._call_ai_api_single(crop_path)
                    
                    if ai_results and len(ai_results) > 0:
                        ai_data = ai_results[0]
                        ai_made_change = False
                        
                        # Fix Name
                        if missing_name:
                            ai_name = ai_data.get('name')
                            if ai_name and len(ai_name.strip()) > 0:
                                self.log_info(f"Recovered name via AI for {voter.serial_no}: '{ai_name}'")
                                voter.name = ai_name
                                ai_made_change = True
                        
                        # Fix Relation Name
                        if missing_relation:
                            ai_rel = ai_data.get('relation_name')
                            if ai_rel and len(ai_rel.strip()) > 0:
                                self.log_info(f"Recovered relation via AI for {voter.serial_no}: '{ai_rel}'")
                                voter.relation_name = ai_rel
                                ai_made_change = True
                                
                        # Update other fields opportunistically (but NOT epic_no - we don't want to overwrite valid EPICs)
                        # epic_no is excluded because AI often misparses and corrupts valid EPIC numbers
                        for field in ['relation_type', 'house_no', 'age', 'gender']:
                             if ai_data.get(field):
                                 setattr(voter, field, ai_data[field])
                        
                        # Only update epic_no if current one is invalid AND AI returns a valid format
                        if not voter.epic_valid and ai_data.get('epic_no'):
                            import re
                            ai_epic = ai_data['epic_no'].strip().upper()
                            if re.fullmatch(r"[A-Z]{3}\d{7}", ai_epic):
                                self.log_info(f"Recovered EPIC via AI for {voter.serial_no}: '{ai_epic}'")
                                voter.epic_no = ai_epic
                                voter.epic_valid = True
                                ai_made_change = True

                        if ai_made_change:
                            voter.processing_method = (voter.processing_method or "") + "+ai_fallback"
                            ai_recovered += 1
                            recovered_any = True
                        else:
                             self.log_warning(f"AI returned no useful data for {voter.serial_no}")
                    else:
                        self.log_warning(f"AI returned no results for {voter.serial_no}")
                        
                except Exception as e:
                    self.log_error(f"AI fallback failed for {voter.serial_no}: {e}")
            
            if not recovered_any:
                failed += 1

        elapsed = time.perf_counter() - start_time
        self.result = MissingNameResult(
            total_records=len(all_voters),
            missing_name_count=len(targets),
            ocr_recovered_count=ocr_recovered,
            ai_recovered_count=ai_recovered,
            failed_count=failed,
            processing_time_sec=elapsed
        )
        
        self.log_info(
            f"Missing data processing complete: "
            f"OCR-fixed={ocr_recovered}, AI-fixed={ai_recovered}, Failed={failed}, Time={elapsed:.2f}s"
        )
        
        return True
