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
                {"house": r.house_no}
                for r in results
            ])
            self.log_info(f"Extracted {len(results)} house numbers for {page_id}")
            
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
Extract house numbers from electoral roll images. Return JSON object with page-wise arrays.

Images belong to: {mapping_desc}

Process all rows top-to-bottom. Use "" for empty/illegible fields.
DO NOT use markdown code blocks. Return raw JSON only.

Example output: {{"page-004": ["12", "34/A"], "page-005": ["5-B", "", "123"]}}
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
                    max_completion_tokens=1024  # Reduced from 4096 since we only need simple array
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
                
                # Strip markdown code blocks if present
                content = content.strip()
                if content.startswith("```"):
                    # Remove opening ```json or ``` 
                    lines = content.split('\n')
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    # Remove closing ```
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    content = '\n'.join(lines).strip()
                
                # Parse JSON - expect page-wise object: {"page-004": ["12", "34/A"], ...}
                try:
                    data = json.loads(content)
                    
                    results_dict = {}
                    
                    # Handle page-wise format (preferred)
                    if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
                        # Page-wise format: {"page-004": ["12", "34"], "page-005": ["56"]}
                        for page_id, house_numbers in data.items():
                            page_results = []
                            for house_no in house_numbers:
                                page_results.append(IdExtractionResult(
                                    house_no=str(house_no).strip()
                                ))
                            results_dict[page_id] = page_results
                            
                        total_extracted = sum(len(v) for v in results_dict.values())
                        self.log_info(f"Extracted {total_extracted} house numbers from {len(image_paths)} images across {len(results_dict)} pages in {elapsed:.2f}s")
                        return results_dict
                    
                    # Fallback: simple array format (distribute to first page_id)
                    elif isinstance(data, list):
                        house_numbers = data
                        results = []
                        for house_no in house_numbers:
                            results.append(IdExtractionResult(
                                house_no=str(house_no).strip()
                            ))
                        # Assign all results to first page
                        if page_ids:
                            results_dict[page_ids[0]] = results
                        self.log_info(f"Extracted {len(results)} house numbers from {len(image_paths)} images in {elapsed:.2f}s")
                        return results_dict
                    
                    else:
                        self.log_error(f"Unexpected JSON format: {type(data)}")
                        return {}
                    
                except json.JSONDecodeError:
                    self.log_error(f"Failed to parse AI response as JSON: {content[:100]}...")
                    # Don't retry on parsing errors as the model output is likely deterministic or the issue is with the response handling
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
