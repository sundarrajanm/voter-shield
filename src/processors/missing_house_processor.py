"""
Missing House Number Processor.

Identifies records with missing house numbers, merges their images,
and uses AI to extract the house numbers before CSV export.
"""

from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

from .base import BaseProcessor, ProcessingContext
from ..models import Voter, ProcessedDocument
from ..logger import get_logger

logger = get_logger("missing_house_processor")


@dataclass
class MissingHouseResult:
    """Result from missing house number processing."""
    total_records: int = 0
    missing_house_count: int = 0
    extracted_count: int = 0
    failed_count: int = 0
    processing_time_sec: float = 0.0


class MissingHouseNumberProcessor(BaseProcessor):
    """
    Processes records with missing house numbers by merging images
    and using AI to extract house numbers.
    """
    
    def __init__(
        self,
        context: ProcessingContext,
        document: ProcessedDocument,
        batch_size: int = 20,
    ):
        super().__init__(context)
        self.document = document
        self.batch_size = batch_size
        self.result: Optional[MissingHouseResult] = None
        
        # Setup AI client
        self.client = None
        self.model = None
        self._setup_ai_client()
    
    def _setup_ai_client(self):
        """Setup OpenAI-compatible AI client."""
        try:
            from openai import OpenAI
            
            ai_config = self.config.ai
            base_url = ai_config.get_normalized_base_url()
            self.client = OpenAI(
                api_key=ai_config.api_key,
                base_url=base_url if base_url else None,
            )
            self.model = ai_config.model
            
            self.log_info(f"Initialized AI client: {ai_config.provider} ({self.model})")
        except Exception as e:
            self.log_error(f"Failed to initialize AI client: {e}")
            raise
    
    def process(self) -> bool:
        """Process records with missing house numbers."""
        start_time = time.perf_counter()
        
        if not self.document or not self.document.pages:
            self.log_warning("No document data to process")
            return False
        
        # Collect all voters
        all_voters: List[Voter] = []
        for page_data in self.document.pages:
            all_voters.extend(page_data.voters)
        
        self.log_info(f"Total voters: {len(all_voters)}")
        
        # Find voters with missing house numbers
        missing_house_voters = [v for v in all_voters if not v.house_no or v.house_no.strip() == ""]
        
        if not missing_house_voters:
            self.log_info("No voters with missing house numbers found")
            self.result = MissingHouseResult(
                total_records=len(all_voters),
                missing_house_count=0,
                extracted_count=0,
                failed_count=0,
                processing_time_sec=time.perf_counter() - start_time
            )
            return True
        
        self.log_info(f"Found {len(missing_house_voters)} voters with missing house numbers")
        
        # Group by page for efficient processing
        voters_by_page: Dict[str, List[Voter]] = {}
        for voter in missing_house_voters:
            page_id = voter.page_id
            if page_id not in voters_by_page:
                voters_by_page[page_id] = []
            voters_by_page[page_id].append(voter)
        
        # Process each page's missing house numbers
        extracted_count = 0
        failed_count = 0
        
        for page_id, page_voters in voters_by_page.items():
            self.log_info(f"Processing {len(page_voters)} missing house numbers in {page_id}")
            
            # Process in batches
            for i in range(0, len(page_voters), self.batch_size):
                batch = page_voters[i:i + self.batch_size]
                
                try:
                    # Merge images for this batch
                    merged_image = self._merge_voter_images(batch, page_id)
                    if merged_image is None:
                        self.log_warning(f"Failed to merge images for batch {i//self.batch_size + 1}")
                        failed_count += len(batch)
                        continue
                    
                    # Extract house numbers using AI
                    house_numbers = self._extract_house_numbers_ai(merged_image, len(batch))
                    
                    if not house_numbers or len(house_numbers) != len(batch):
                        self.log_warning(
                            f"AI returned {len(house_numbers) if house_numbers else 0} house numbers, "
                            f"expected {len(batch)}"
                        )
                        failed_count += len(batch)
                        continue
                    
                    # Update voters with extracted house numbers
                    for voter, house_no in zip(batch, house_numbers):
                        if house_no and house_no.strip():
                            voter.house_no = house_no.strip()
                            extracted_count += 1
                            self.log_debug(f"Updated {voter.serial_no}: house_no = {house_no}")
                        else:
                            failed_count += 1
                            self.log_debug(f"No house number extracted for {voter.serial_no}")
                
                except Exception as e:
                    self.log_error(f"Error processing batch: {e}")
                    failed_count += len(batch)
        
        elapsed = time.perf_counter() - start_time
        
        self.result = MissingHouseResult(
            total_records=len(all_voters),
            missing_house_count=len(missing_house_voters),
            extracted_count=extracted_count,
            failed_count=failed_count,
            processing_time_sec=elapsed
        )
        
        self.log_info(
            f"Missing house number processing complete: "
            f"extracted={extracted_count}, failed={failed_count}, time={elapsed:.2f}s"
        )
        
        return True
    
    def _merge_voter_images(self, voters: List[Voter], page_id: str) -> Optional[np.ndarray]:
        """
        Merge multiple voter images vertically into a single image.
        
        Args:
            voters: List of voters whose images to merge
            page_id: Page ID to locate images
            
        Returns:
            Merged image as numpy array, or None if failed
        """
        # Locate the crops directory for this page
        crops_dir = self.context.crops_dir / page_id
        if not crops_dir.exists():
            self.log_error(f"Crops directory not found: {crops_dir}")
            return None
        
        images = []
        for voter in voters:
            # Find the image file for this voter
            img_file = crops_dir / voter.image_file
            
            if not img_file.exists():
                self.log_warning(f"Image file not found: {img_file}")
                # Create a blank placeholder
                blank = np.ones((100, 800, 3), dtype=np.uint8) * 255
                images.append(blank)
                continue
            
            # Read image
            try:
                img = cv2.imdecode(
                    np.fromfile(str(img_file), dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
                
                if img is None:
                    self.log_warning(f"Failed to read image: {img_file}")
                    blank = np.ones((100, 800, 3), dtype=np.uint8) * 255
                    images.append(blank)
                    continue
                
                images.append(img)
            except Exception as e:
                self.log_error(f"Error reading image {img_file}: {e}")
                blank = np.ones((100, 800, 3), dtype=np.uint8) * 255
                images.append(blank)
        
        if not images:
            return None
        
        # Find max width
        max_width = max(img.shape[1] for img in images)
        
        # Resize all images to same width (pad if needed)
        resized_images = []
        for img in images:
            h, w = img.shape[:2]
            if w < max_width:
                # Pad right with white
                padding = np.ones((h, max_width - w, 3), dtype=np.uint8) * 255
                img = np.hstack([img, padding])
            resized_images.append(img)
        
        # Add separator lines between images
        separator = np.ones((5, max_width, 3), dtype=np.uint8) * 200  # Gray separator
        
        merged_parts = []
        for i, img in enumerate(resized_images):
            merged_parts.append(img)
            if i < len(resized_images) - 1:
                merged_parts.append(separator)
        
        # Merge vertically
        merged = np.vstack(merged_parts)
        
        return merged
    
    def _extract_house_numbers_ai(self, merged_image: np.ndarray, expected_count: int) -> List[str]:
        """
        Extract house numbers from merged image using AI.
        
        Args:
            merged_image: Merged image containing multiple voter records
            expected_count: Expected number of house numbers to extract
            
        Returns:
            List of house numbers (empty strings for missing)
        """
        try:
            # Encode image to base64
            _, buffer = cv2.imencode('.png', merged_image)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare prompt
            prompt = f"""Extract the house numbers from this merged electoral roll image.

The image contains {expected_count} voter records stacked vertically.
Extract the house number (வீட்டு எண் / House No) for each record from top to bottom.

IMPORTANT: Preserve ALL characters including Tamil letters (வீட்டு, தெரு, ஏ, வ, etc.).

Return ONLY a JSON array of house numbers as strings in order.
Use empty string "" if house number is not visible or illegible.

Examples: ["12", "34/A", "5வ283", "2ஏ"] or ["12", "34/A", "", "56-B"]

Return raw JSON only, no markdown code blocks."""
            
            # Create message
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Call AI
            max_retries = self.config.ai.max_retries
            retry_delay = self.config.ai.retry_delay_sec
            
            for attempt in range(max_retries + 1):
                try:
                    start_time = time.perf_counter()
                    
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0,
                        max_completion_tokens=512
                    )
                    
                    elapsed = time.perf_counter() - start_time
                    
                    if not completion.choices:
                        raise ValueError("No response from AI")
                    
                    content = completion.choices[0].message.content
                    if not content:
                        raise ValueError("Empty response from AI")
                    
                    # Track AI usage
                    if hasattr(completion, 'usage') and completion.usage:
                        self.context.ai_usage.add_call(
                            input_tokens=completion.usage.prompt_tokens,
                            output_tokens=completion.usage.completion_tokens,
                            cost_usd=self.config.ai.estimate_cost(
                                completion.usage.prompt_tokens,
                                completion.usage.completion_tokens
                            )
                        )
                    
                    # Parse JSON response
                    content = content.strip()
                    
                    # Remove markdown code blocks if present
                    if content.startswith("```"):
                        lines = content.split('\n')
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].strip() == "```":
                            lines = lines[:-1]
                        content = '\n'.join(lines).strip()
                    
                    # Parse JSON
                    house_numbers = json.loads(content)
                    
                    if not isinstance(house_numbers, list):
                        raise ValueError(f"Expected list, got {type(house_numbers)}")
                    
                    # Convert all to strings
                    house_numbers = [str(h).strip() for h in house_numbers]
                    
                    self.log_info(f"Extracted {len(house_numbers)} house numbers in {elapsed:.2f}s")
                    
                    return house_numbers
                
                except Exception as e:
                    if attempt < max_retries:
                        self.log_warning(f"AI call attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(retry_delay)
                    else:
                        raise
            
            return []
        
        except Exception as e:
            self.log_error(f"Failed to extract house numbers using AI: {e}")
            return []
