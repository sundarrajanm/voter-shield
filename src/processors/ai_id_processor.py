"""
AI ID Processor.

Extracts House Numbers from cropped ID strips using AI Vision.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import BaseProcessor, ProcessingContext
from ..models import AIUsage
from ..logger import get_logger

logger = get_logger("ai_id_processor")


@dataclass
class IdExtractionResult:
    """Result from AI ID extraction."""
    serial_no: str
    house_no: str


@dataclass
class PageIdResult:
    """ID extraction results for a page."""
    page_id: str
    results: List[IdExtractionResult] = field(default_factory=list)


class AIIdProcessor(BaseProcessor):
    """
    Extracts ID data from merged ID strips using AI.
    """
    
    name = "AIIdProcessor"
    
    def __init__(self, context: ProcessingContext):
        super().__init__(context)
        self.client = None
        self.model = self.config.ai.model
        self.batch_size = self.config.ai.id_batch_size
        self.log_info(f"Initialized with ID Batch Size: {self.batch_size}")
        
        # We need to map extracted data back to voters
        self.page_results: Dict[str, List[IdExtractionResult]] = {}

    def validate(self) -> bool:
        """Validate prerequisites."""
        if not self.config.ai.api_key:
            self.log_error("AI API Key not set")
            return False
            
        if not self.context.id_merged_dir:
            self.log_error("ID merged directory not set")
            return False
            
        if not self.context.id_merged_dir.exists():
            self.log_error(f"ID merged directory not found: {self.context.id_merged_dir}")
            return False
            
        return True
        
    def _init_client(self):
        """Initialize AI client (using Groq/OpenAI compatible interface)."""
        if self.client:
            return
            
        try:
            from openai import OpenAI
            
            base_url = self.config.ai.get_normalized_base_url()
            self.client = OpenAI(
                api_key=self.config.ai.api_key,
                base_url=base_url if base_url else None
            )
            self.log_info(f"Initialized AI client with model {self.model}")
        except ImportError:
            self.log_error("openai package not installed. Please install it: pip install openai")
            raise

    def process(self) -> bool:
        """Process all ID merged batches."""
        self._init_client()
        
        id_merged_dir = self.context.id_merged_dir
        merged_pages = sorted([p for p in id_merged_dir.iterdir() if p.is_dir()])
        
        if not merged_pages:
            self.log_warning(f"No pages found in {id_merged_dir}")
            return False
            
        total_pages = len(merged_pages)
        
        # Collect all images with their page IDs
        all_image_info = []  # List of (image_path, page_id)
        
        for page_dir in merged_pages:
            page_id = page_dir.name
            batch_images = sorted([
                p for p in page_dir.iterdir() 
                if p.is_file() and p.name.startswith("batch-") and p.suffix.lower() == ".png"
            ])
            
            for img_path in batch_images:
                all_image_info.append((img_path, page_id))
        
        total_images = len(all_image_info)
        self.log_info(f"Processing {total_images} images from {total_pages} pages in batches of {self.batch_size}...")
        
        # Process in batches across pages
        for i in range(0, total_images, self.batch_size):
            batch_info = all_image_info[i:i + self.batch_size]
            image_paths = [info[0] for info in batch_info]
            page_ids = [info[1] for info in batch_info]
            
            # Process batch and get page-wise results
            page_results_dict = self._process_image_batch(image_paths, page_ids)
            
            # Accumulate results
            for page_id, results in page_results_dict.items():
                if page_id not in self.page_results:
                    self.page_results[page_id] = []
                self.page_results[page_id].extend(results)
        
        # Save debug info for each page
        for page_id, results in self.page_results.items():
            self.save_debug_info(f"id_extract_{page_id}", [
                {"serial_no": r.serial_no, "house_no": r.house_no}
                for r in results
            ])
            self.log_info(f"Extracted {len(results)} voter records (serial+house) for {page_id}")
            
        return True

    def _process_image_batch(self, image_paths: List[Path], page_ids: List[str]) -> Dict[str, List[IdExtractionResult]]:
        """Send a batch of images to AI and parse response."""
        import base64
        
        # Build page mapping info for the prompt
        page_mapping = {}
        current_page = None
        image_idx = 0
        
        for i, page_id in enumerate(page_ids):
            if page_id not in page_mapping:
                page_mapping[page_id] = []
            page_mapping[page_id].append(i + 1)  # 1-indexed for clarity
        
        # Create mapping description
        mapping_desc = ", ".join([f"{page_id}: image {','.join(map(str, imgs))}" for page_id, imgs in page_mapping.items()])
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"""
Extract serial and house numbers from electoral roll ID strips.

Images: {mapping_desc}

Return TOML format. NO markdown blocks. Use "" for empty fields.

Format:
[page-XXX]
records = [
  ["serial1", "house1"],
  ["serial2", "house2"]
]

Example:
[page-004]
records = [
  ["211", "12"],
  ["212", "34/A"],
  ["213", "5-B"]
]

[page-005]
records = [
  ["214", "16(A)"],
  ["215", "17"]
]

IMPORTANT: Extract complete house numbers with ALL characters including:
- Tamil letters (வீட்டு, தெரு, ஏ, வ, etc.)
- English letters (A, B, C, etc.)
- Numbers (0-9)
- Special characters: /, -, (, ), comma
Do NOT skip or omit Tamil letters - they are part of the house number.
Process top-to-bottom, maintain sequence.
                        """
                    }
                ]
            }
        ]
        
        # Append images
        for img_path in image_paths:
            try:
                with open(img_path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode('utf-8')
                    
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            except Exception as e:
                self.log_error(f"Failed to read image {img_path}: {e}")
                
        max_retries = self.config.ai.max_retries
        retry_delay = self.config.ai.retry_delay_sec
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.perf_counter()
                
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=4096  # Increased to handle larger batches (5 pages × 30 voters)
                )
                
                elapsed = time.perf_counter() - start_time
                
                # Track usage (separate from ai_metadata_voter)
                if hasattr(completion, 'usage') and completion.usage:
                    input_tokens = completion.usage.prompt_tokens
                    output_tokens = completion.usage.completion_tokens
                    cost = self.config.ai.estimate_cost(input_tokens, output_tokens)
                    
                    # Track only in context.ai_usage (not in stats for metadata export)
                    self.context.ai_usage.add_call(input_tokens, output_tokens, cost)
                
                
                content = completion.choices[0].message.content
                
                # Save raw AI response for debugging
                raw_response_debug = {
                    "raw_response": content,
                    "image_count": len(image_paths),
                    "pages": list(set(page_ids)),
                    "timestamp": elapsed
                }
                if page_ids:
                    debug_filename = f"ai_raw_response_{page_ids[0]}_batch"
                    self.save_debug_info(debug_filename, raw_response_debug)
                
                # Strip markdown code blocks if present
                content = content.strip()
                if content.startswith("```"):
                    lines = content.split('\n')
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    content = '\n'.join(lines).strip()
                
                # Parse TOML with improved validation
                try:
                    import toml
                    data = toml.loads(content)
                    
                    results_dict = {}
                    
                    # Validate structure
                    if not isinstance(data, dict):
                        self.log_error(f"AI response is not a valid TOML object, got: {type(data)}")
                        return {}
                    
                    for page_id, page_data in data.items():
                        # Skip non-page keys
                        if not page_id.startswith("page-"):
                            self.log_warning(f"Skipping non-page key: {page_id}")
                            continue
                        
                        if not isinstance(page_data, dict):
                            self.log_warning(f"Page data for {page_id} is not a dict, skipping")
                            continue
                        
                        # Look for 'records' key (new format) or 'voter_records' (legacy)
                        voter_records = page_data.get("records") or page_data.get("voter_records", [])
                        if not isinstance(voter_records, list):
                            self.log_warning(f"Records for {page_id} is not a list, skipping")
                            continue
                        
                        page_results = []
                        for idx, record in enumerate(voter_records):
                            if isinstance(record, list) and len(record) >= 2:
                                serial_no = str(record[0]).strip()
                                house_no = str(record[1]).strip()
                                page_results.append(IdExtractionResult(
                                    serial_no=serial_no,
                                    house_no=house_no
                                ))
                            elif isinstance(record, list) and len(record) == 1:
                                # Fallback: only house_no provided
                                page_results.append(IdExtractionResult(
                                    serial_no="",
                                    house_no=str(record[0]).strip()
                                ))
                            else:
                                self.log_warning(f"Invalid record format in {page_id} at index {idx}: {record}")
                        
                        # Only add to DICT if we got valid data for THIS page
                        if page_results:
                            results_dict[page_id] = page_results
                            self.log_debug(f"Extracted {len(page_results)} records for {page_id}")
                        else:
                            # Empty results for this page - don't carry over from previous page
                            self.log_warning(f"No valid records extracted for {page_id}")
                    
                    if results_dict:
                        total_extracted = sum(len(v) for v in results_dict.values())
                        self.log_info(f"Extracted {total_extracted} voter records from {len(image_paths)} images across {len(results_dict)} pages in {elapsed:.2f}s")
                        return results_dict
                    else:
                        self.log_error(f"No valid page data found in TOML response")
                        return {}
                    
                except Exception as e:
                    self.log_error(f"Failed to parse AI response as TOML: {e}")
                    self.log_debug(f"Response content (first 500 chars): {content[:500]}...")
                    return {}
                    
            except Exception as e:
                is_last_attempt = attempt == max_retries
                error_msg = f"AI API call failed (attempt {attempt+1}/{max_retries+1}): {e}"
                
                if is_last_attempt:
                    self.log_error(f"AI API call completely failed after {max_retries+1} attempts: {e}")
                    return {}
                else:
                    self.log_warning(f"{error_msg}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

    def get_results_for_page(self, page_id: str) -> List[IdExtractionResult]:
        return self.page_results.get(page_id, [])
    
    def get_all_results(self) -> List[IdExtractionResult]:
        """
        Get ALL AI extraction results across ALL pages as a flat list.
        
        This is used for global serial_no matching since AI page IDs
        may not match OCR page IDs.
        
        Returns:
            List of all IdExtractionResult objects
        """
        all_results = []
        for page_id, results in self.page_results.items():
            all_results.extend(results)
        return all_results
    
    def get_global_serial_house_map(self) -> Dict[str, str]:
        """
        Get a global mapping of serial_no -> house_no across ALL pages.
        
        This is needed because AI may assign records to wrong page IDs,
        but the serial numbers are usually correct. By using this global map,
        we can match by serial number regardless of which page the AI thought
        the record belonged to.
        
        Returns:
            Dict mapping serial_no to house_no
        """
        serial_map = {}
        for page_id, results in self.page_results.items():
            for result in results:
                if result.serial_no and result.serial_no.strip():
                    serial_no = result.serial_no.strip()
                    # Only add if not already present (prefer earlier entries)
                    if serial_no not in serial_map:
                        serial_map[serial_no] = result.house_no.strip() if result.house_no else ""
        return serial_map

