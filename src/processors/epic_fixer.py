"""
EPIC Fixer Processor.

Retries invalid EPICs using AI with full crop images before CSV export.
"""

from __future__ import annotations

import time
import re
from pathlib import Path
from typing import Optional

from .base import BaseProcessor, ProcessingContext
from ..models import ProcessedDocument, Voter
from ..utils.ai_deleted_detector import AIDeletedDetector
from ..logger import get_logger

logger = get_logger("epic_fixer")


class EPICFixerProcessor(BaseProcessor):
    """
    Retry invalid EPICs using AI with full crop images.
    
    This runs before CSV export to give one final chance to correct
    EPICs that failed validation during OCR.
    """
    
    name = "EPICFixerProcessor"
    
    def __init__(self, context: ProcessingContext):
        super().__init__(context)
        self.ai_detector = None
        self.fixed_count = 0
        self.attempted_count = 0
        
    def validate(self) -> bool:
        """Validate prerequisites."""
        if not self.config.ai.api_key:
            self.log_warning("AI API Key not set - skipping EPIC fix")
            return False
            
        if not self.context.crops_dir:
            self.log_warning("Crops directory not set - skipping EPIC fix")
            return False
            
        return True
    
    def _init_ai(self):
        """Initialize AI detector."""
        if self.ai_detector:
            return
            
        try:
            self.ai_detector = AIDeletedDetector(
                api_key=self.config.ai.api_key,
                base_url=self.config.ai.get_normalized_base_url(),
                model=self.config.ai.model,
                max_retries=self.config.ai.max_retries,
                retry_delay=self.config.ai.retry_delay_sec,
            )
            self.log_info("Initialized AI detector for EPIC fixing")
        except Exception as e:
            self.log_error(f"Failed to initialize AI detector: {e}")
            raise
    
    def process_document(self, document, pdf_name: str = None) -> bool:
        """
        Process document and fix invalid EPICs.
        
        Args:
            document: Processed document with voters (ProcessedDocument or dict)
            pdf_name: PDF name for loading page-wise files (required if document is dict)
            
        Returns:
            True if processing succeeded
        """
        if not self.validate():
            return False
            
        self._init_ai()
        
        # Handle both ProcessedDocument and dict
        if isinstance(document, dict):
            doc_name = document.get('pdf_name', pdf_name or 'unknown')
            # For dict documents, voters are in 'records' array at the top level
            all_voters = document.get('records', [])
            if not all_voters:
                self.log_warning(f"No records found in document {doc_name}")
                return False
            pages_data = [(None, all_voters)]  # Single group with all voters
        else:
            doc_name = document.pdf_name
            pages_data = [(page, page.voters) for page in document.pages]
        
        self.log_info(f"Checking for invalid EPICs in {doc_name}...")
        
        self.log_debug(f"Loaded {len(pages_data)} pages")
        
        # Collect all invalid EPICs
        invalid_voters = []
        for page_info, voters in pages_data:
            self.log_debug(f"Page has {len(voters)} voters")
            for voter_data in voters:
                # Handle both dict and Voter object
                if isinstance(voter_data, dict):
                    epic_no = voter_data.get('epic_no', '')
                    epic_valid = voter_data.get('epic_valid', False)
                    if epic_no and not epic_valid:
                        invalid_voters.append(voter_data)
                else:
                    if voter_data.epic_no and not voter_data.epic_valid:
                        invalid_voters.append(voter_data)
        
        if not invalid_voters:
            self.log_info("No invalid EPICs found - skipping")
            return True
        
        self.log_info(f"Found {len(invalid_voters)} invalid EPICs - attempting to fix with AI...")
        
        # Attempt to fix each invalid EPIC
        for voter_data in invalid_voters:
            self.attempted_count += 1
            
            # Handle both dict and Voter object
            if isinstance(voter_data, dict):
                page_id = voter_data.get('page_id', '')
                image_file = voter_data.get('image_file', '')
                epic_no = voter_data.get('epic_no', '')
            else:
                page_id = voter_data.page_id
                image_file = voter_data.image_file
                epic_no = voter_data.epic_no
            
            # Get the crop image path
            crop_path = self.context.crops_dir / page_id / image_file
            
            if not crop_path.exists():
                # Try alternate location
                crop_path = self.context.crops_dir / page_id / "images" / image_file
            
            if not crop_path.exists():
                self.log_warning(f"Crop image not found for {image_file}: {crop_path}")
                continue
            
            # Try to fix EPIC
            fixed_epic = self._fix_epic_with_ai(crop_path, image_file, epic_no)
            
            if fixed_epic and fixed_epic != epic_no:
                old_epic = epic_no
                # Update the voter data
                if isinstance(voter_data, dict):
                    voter_data['epic_no'] = fixed_epic
                    voter_data['epic_valid'] = True
                else:
                    voter_data.epic_no = fixed_epic
                    voter_data.epic_valid = True
                self.fixed_count += 1
                self.log_info(f"✓ Fixed EPIC for {image_file}: {old_epic} → {fixed_epic}")
            else:
                self.log_warning(f"✗ Could not fix EPIC for {image_file}: {epic_no}")
        
        self.log_info(f"EPIC Fix Results: Fixed {self.fixed_count}/{self.attempted_count} invalid EPICs")
        
        # Save updated document if we fixed any EPICs
        if self.fixed_count > 0 and isinstance(document, dict):
            self._save_updated_document(document, doc_name)
        
        return True
    
    def _fix_epic_with_ai(self, crop_path: Path, image_name: str, current_epic: str) -> Optional[str]:
        """
        Use AI to extract EPIC from full crop image.
        
        Args:
            crop_path: Path to voter crop image
            image_name: Name of the image for logging
            current_epic: Current (invalid) EPIC value
            
        Returns:
            Fixed EPIC or None if unable to fix
        """
        try:
            import cv2
            import numpy as np
            
            # Read the full crop image
            img_bgr = cv2.imdecode(
                np.fromfile(str(crop_path), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            
            if img_bgr is None:
                self.log_warning(f"Failed to read image: {crop_path}")
                return None
            
            self.log_debug(f"Sending full crop image to AI: {image_name} (current EPIC: {current_epic})")
            
            # Send full crop to AI
            ai_result = self.ai_detector.extract_fields(image_array=img_bgr)
            
            if not ai_result.epic_no:
                self.log_debug(f"AI returned no EPIC for {image_name}")
                return None
            
            # Validate: Must be exactly 3 letters + 7 digits = 10 characters
            if len(ai_result.epic_no) == 10 and re.fullmatch(r"[A-Z]{3}\d{7}", ai_result.epic_no):
                self.log_debug(f"AI returned valid EPIC for {image_name}: {ai_result.epic_no}")
                return ai_result.epic_no
            else:
                self.log_warning(
                    f"AI returned invalid EPIC format for {image_name}: "
                    f"{ai_result.epic_no} (length: {len(ai_result.epic_no)})"
                )
                return None
                
        except Exception as e:
            self.log_error(f"Error fixing EPIC for {image_name}: {e}")
            return None
    
    def _save_updated_document(self, document: dict, pdf_name: str):
        """Save updated main document JSON file."""
        import json
        
        output_dir = self.context.extracted_dir / pdf_name / "output"
        doc_file = output_dir / f"{pdf_name}.json"
        
        try:
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(document, f, indent=2, ensure_ascii=False)
            
            self.log_info(f"Saved updated document: {doc_file}")
            
        except Exception as e:
            self.log_error(f"Failed to save updated document: {e}")
    
    def process(self) -> bool:
        """Not used - use process_document instead."""
        return True
