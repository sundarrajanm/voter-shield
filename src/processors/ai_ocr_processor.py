
from __future__ import annotations

import base64
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from .base import BaseProcessor, ProcessingContext
from ..models import Voter
from ..models.processing_stats import PageTiming
from ..utils.ai_parser import parse_ai_response


@dataclass
class FailedImage:
    """Information about a failed image for retry purposes."""
    image_path: Path
    page_id: str
    image_index: int
    error_message: str
    retry_count: int = 0
    last_error_time: float = field(default_factory=time.time)


class AIOCRProcessor(BaseProcessor):
    """
    Extract voter information using AI (Groq) Vision capabilities.
    
    Processing Flow:
    1. Collects all merged images from all pages into a unified queue.
    2. Sends AI_OCR_BATCH_SIZE concurrent API requests (1 image per request).
    3. Waits for all concurrent requests to complete, then processes results in order.
    4. Aggregates results page-wise and handles saving via callback.
    
    This approach maximizes throughput while maintaining correct page ordering.
    """
    
    name = "AIOCRProcessor"
    
    # Legacy: Separator used by _call_ai_api for multi-image batches (kept for reference)
    IMAGE_SEPARATOR = "---NEXT_IMAGE---"
    
    def __init__(
        self,
        context: ProcessingContext,
        on_page_complete: Optional[callable] = None,
    ):
        super().__init__(context)
        self.on_page_complete = on_page_complete
        self.merged_dir = self.context.extracted_dir / "merged" if self.context.extracted_dir else None
        self.client = None
        self.model = self.config.ai.model or "meta-llama/llama-4-maverick-17b-128e-instruct" 
        
        # Concurrency level for AI requests (number of parallel API calls)
        # Using environment variable AI_OCR_BATCH_SIZE (default 5)
        self.ai_batch_size = self.config.ai.batch_size
        
        # Retry configuration
        self.max_retries = self.config.ai.max_retries
        self.retry_delay = self.config.ai.retry_delay_sec
        
        # Accumulator for final list return
        self._all_voters: List[Voter] = []
        
        # Page tracking state
        self.page_voters = defaultdict(list)          # page_id -> List[Voter]
        self.page_image_counts = {}                   # page_id -> total_images_count
        self.page_images_processed = defaultdict(int) # page_id -> processed_count
        self.page_start_times = {}                    # page_id -> start_time
        
        # Failed image tracking (thread-safe)
        self._failed_images: List[FailedImage] = []
        self._failed_images_lock = threading.Lock()

    def validate(self) -> bool:
        if not GROQ_AVAILABLE:
            self.log_error("Groq library not installed. Install with: pip install groq")
            return False
            
        if not self.config.ai.api_key:
            self.log_error("AI_API_KEY not set. Please set in .env or environment variables.")
            return False
            
        if not self.merged_dir or not self.merged_dir.exists():
            self.log_error(f"Merged directory not found: {self.merged_dir}. Run 'merge' step first.")
            return False
            
        return True

    def _initialize_client(self):
        if not self.client:
            self.client = Groq(api_key=self.config.ai.api_key)
            # Set AI usage metadata for proper ai_metadata_voter output
            self.context.stats.ai_usage.provider = "Groq"
            self.context.stats.ai_usage.model = self.model

    def process(self) -> bool:
        self._initialize_client()
        self._all_voters = []
        self.page_voters.clear()
        self.page_image_counts.clear()
        self.page_images_processed.clear()
        self.page_start_times.clear()
        
        # Clear failed images for fresh processing
        with self._failed_images_lock:
            self._failed_images.clear()
        
        # 1. Collect all images from all pages
        page_dirs = sorted([d for d in self.merged_dir.iterdir() if d.is_dir()])
        
        if not page_dirs:
            self.log_warning(f"No page directories found in {self.merged_dir}")
            return False
            
        # Structure to track task details: (image_path, page_id, image_index)
        all_tasks = [] 
        
        for i, page_dir in enumerate(page_dirs):
            page_id = page_dir.name
            self.page_start_times[page_id] = time.perf_counter()
            
            images = sorted([
                p for p in page_dir.iterdir() 
                if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg')
            ])
            
            self.page_image_counts[page_id] = len(images)
            
            for img_idx, img in enumerate(images):
                all_tasks.append((img, page_id, img_idx))
                
        total_images = len(all_tasks)
        self.log_info(f"Collected {total_images} images from {len(page_dirs)} pages.")
        self.log_info(f"Processing with {self.ai_batch_size} concurrent requests (Model: {self.model})")
        self.log_info(f"Retry config: max_retries={self.max_retries}, retry_delay={self.retry_delay}s")
        
        # Temporary storage for results before finalization: page_id -> {img_idx -> voters}
        page_results = defaultdict(dict)
        
        # Track which pages are ready to be finalized
        # We can only finalize page N if page N is done AND page N-1 is already finalized.
        next_page_idx_to_finalize = 0
        
        overall_start = time.perf_counter()
        
        # Use a single executor for all tasks to avoid "convoy effect"
        # This ensures that as soon as one request finishes, another takes its place, 
        # maintaining full utilization of the concurrency limit.
        with ThreadPoolExecutor(max_workers=self.ai_batch_size) as executor:
            future_to_info = {}
            
            # Submit all tasks
            for img, page_id, img_idx in all_tasks:
                future = executor.submit(self._process_single_image, img)
                future_to_info[future] = (page_id, img_idx, img)
            
            # Process as they complete
            for future in as_completed(future_to_info):
                page_id, img_idx, img_path = future_to_info[future]
                
                try:
                    voters = future.result()
                    page_results[page_id][img_idx] = voters
                except Exception as e:
                    self.log_error(f"Failed to process {img_path.name} after all retries: {e}")
                    page_results[page_id][img_idx] = []  # Empty list on failure
                    
                    # Track the failed image for potential later retry
                    self._track_failed_image(img_path, page_id, img_idx, str(e))
                
                # Increment processed count
                self.page_images_processed[page_id] += 1
                
                # Check for finalization (Sequential Logic)
                # We loop to check if the *next expected page* is ready.
                while next_page_idx_to_finalize < len(page_dirs):
                    target_page_dir = page_dirs[next_page_idx_to_finalize]
                    target_page_id = target_page_dir.name
                    
                    processed_count = self.page_images_processed[target_page_id]
                    total_count = self.page_image_counts[target_page_id]
                    
                    if processed_count >= total_count:
                        # Page is fully done. Finalize it.
                        self._finalize_page_results(target_page_id, page_results[target_page_id])
                        next_page_idx_to_finalize += 1
                    else:
                        # Next expected page is not ready yet. 
                        # Even if subsequent pages are ready, we wait to maintain global order.
                        break
                        
        overall_elapsed = time.perf_counter() - overall_start
        
        # Log summary
        failed_count = len(self._failed_images)
        success_count = total_images - failed_count
        
        self.log_info(f"AI OCR Completed. Processed {total_images} images in {overall_elapsed:.2f}s "
                      f"(Avg: {overall_elapsed/total_images:.2f}s/image)")
        
        if failed_count > 0:
            self.log_warning(f"⚠️ {failed_count} images failed after all retries. "
                           f"Use get_failed_images() or retry_failed_images() to handle them.")
            # Log affected pages
            affected_pages = set(fi.page_id for fi in self._failed_images)
            self.log_warning(f"Affected pages: {', '.join(sorted(affected_pages))}")
        else:
            self.log_info(f"✓ All {success_count} images processed successfully.")
            
        return True

    def _finalize_page_results(self, page_id: str, results_dict: Dict[int, List[Dict[str, Any]]]):
        """
        Reassemble voters for a page in correct order and finalize.
        """
        # Reconstruct list sorted by image index
        sorted_indices = sorted(results_dict.keys())
        all_page_voters = []
        
        for idx in sorted_indices:
            raw_voters = results_dict.get(idx, [])
            # Convert
            for v_data in raw_voters:
                voter = self._create_voter_from_dict(v_data, page_id)
                all_page_voters.append(voter)
                
        # Populate self.page_voters so _check_and_finalize_page uses the correct list
        self.page_voters[page_id] = all_page_voters
        
        # Perform finalization (logging, stats, callbacks)
        self._check_and_finalize_page(page_id)
    
    def _process_single_image(self, img_path: Path) -> List[Dict[str, Any]]:
        """
        Process a single image through the AI API.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            List of voter dictionaries extracted from the image
        """
        # Capture image name for logging (avoid closure issues with threads)
        image_name = img_path.name
        self.log_debug(f"Starting AI processing for: {image_name}")
        result = self._call_ai_api_single(img_path, image_name)
        return result

    def _check_and_finalize_page(self, page_id: str):
        """Check if all images for a page are processed, then finalize."""
        processed = self.page_images_processed[page_id]
        total = self.page_image_counts[page_id]
        
        if processed >= total:
            # Page Complete
            page_voters = self.page_voters[page_id]
            
            # Sort valid voters by serial no or sequence? 
            # Usually they are added in order of images, so should be correct.
            # Just fix sequence numbers.
            # Note: We need accurate global sequence? 
            # We can't know global sequence perfectly until ALL pages are done if we processed out of order.
            # But we processed strictly in order of `all_tasks` (sorted by page), so we are safe.
            
            # Correct local sequence and global sequence
            # Use serial_no from AI response if available, otherwise fall back to sequence
            current_global_base = self.context.total_voters_found
            for i, v in enumerate(page_voters, 1):
                v.sequence_in_page = i
                v.sequence_in_document = current_global_base + i
                # Keep the serial_no from AI response - don't overwrite it
                # Only set if empty or invalid
                if not v.serial_no or not v.serial_no.strip():
                    v.serial_no = str(v.sequence_in_document)
                
            self.context.total_voters_found += len(page_voters)
            self._all_voters.extend(page_voters)
            
            # Timing
            elapsed = time.perf_counter() - self.page_start_times[page_id]
            
            # Update stats
            try:
                page_num = 0
                try: 
                    page_num = int(page_id.split("-")[1])
                except:
                    pass
                
                pt = PageTiming(
                    page_id=page_id,
                    page_number=page_num,
                    voters_extracted=len(page_voters),
                    voters_valid=sum(1 for v in page_voters if v.epic_valid),
                    ocr_time_sec=elapsed,
                    total_time_sec=elapsed
                )
                self.context.stats.add_page_timing(pt)
            except Exception as e:
                self.log_warning(f"Failed to update stats for {page_id}: {e}")
            
            self.log_info(f"Finished {page_id}: {len(page_voters)} voters in {elapsed:.2f}s")

            
            if self.on_page_complete:
                self.on_page_complete(page_id, page_voters, elapsed)

            # Cleanup to save memory
            del self.page_voters[page_id]

    def _create_voter_from_dict(self, v_data: Dict[str, Any], page_id: str) -> Voter:
        """Map raw dictionary to Voter object."""
        voter = Voter(
            serial_no=str(v_data.get("serial_no", "")),
            epic_no=v_data.get("epic_no", "") or v_data.get("epic", ""),
            name=v_data.get("name", ""),
            relation_type=v_data.get("relation_type", "Father"), # Default
            relation_name=v_data.get("relation_name", "") or v_data.get("relation", ""),
            house_no=str(v_data.get("house_no", "") or v_data.get("house_number", "")),
            age=str(v_data.get("age", "")),
            gender=v_data.get("gender", ""),
            
            # Deleted status (empty string = not deleted, "true" = deleted)
            deleted=v_data.get("deleted", "") or "",
            
            # Metadata
            page_id=page_id,
            image_file=f"{page_id}-ai_batch", 
            processing_method="ai_groq"

        )

        
        # Heuristics for relation
        if not voter.relation_type or voter.relation_type == "Father":
            if "father_name" in v_data:
                voter.relation_type = "Father"
                voter.relation_name = v_data["father_name"]
            elif "husband_name" in v_data:
                voter.relation_type = "Husband"
                voter.relation_name = v_data["husband_name"]
            elif "mother_name" in v_data:
                voter.relation_type = "Mother"
                voter.relation_name = v_data["mother_name"]
        
        return voter

    def _call_ai_api(self, images: List[Path]) -> List[List[Dict[str, Any]]]:
        """
        Send batch of images to Groq API.
        
        Returns:
            List of Lists: One list of voters for each image in the batch.
        """
        
        # Prompt construction
        prompt_text = (
            "Extract voter information from the provided electoral roll images. "
            f"There are {len(images)} images in this request. Process them strictly in order.\n"
            f"Output the results for each image separated by exactly '{self.IMAGE_SEPARATOR}'.\n\n"
            "Format for EACH image:\n"
            "items[N]{serial_no,epic_no,name,relation_type,relation_name,house_no,age,gender}:\n"
            "  31,NHH3334638,kumar,father,arumugam,40NM,37,Male\n"
            "  ...\n\n"
            "Rules:\n"
            f"1. Precede each image's output block with '{self.IMAGE_SEPARATOR}'.\n"
            "2. Output field names map:\n"
            "   - serial_no \n"
            "   - epic_no (IMPORTANT: Extract exactly 3 CAPITAL LETTERS followed by 7 DIGITS. Example: ABC1234567.)\n"
            "   - name (Name / பெயர்)\n"
            "   - relation_type (infer: father/husband/mother/other)\n"
            "   - relation_name (Father/Husband Name / தந்தை/கணவர்/தாய் பெயர்)\n"
            "   - house_no (House No / வீட்டு எண்) - Look for 'வீட்டு எண்:'\n"
            "   - age (Age / வயது)\n"
            "   - gender (Gender / பாலினம்)\n"
            "3. Values can be in Tamil if the input text is in Tamil.\n"
            "4. Use comma (,) as delimiter. Replace commas inside values with space.\n"
            "5. NO Markdown code blocks. Just the raw text."
        )

        content = [
            {
                "type": "text", 
                "text": prompt_text
            }
        ]
        
        for img_path in images:
            b64_str = self._encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_str}"
                }
            })
            
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": content}],
                model=self.model,
                temperature=0, 
                max_tokens=6000, # Ensure enough room for multi-image response
            )
            
            response_text = completion.choices[0].message.content
            
            # Track usage
            if hasattr(completion, 'usage') and completion.usage:
                input_tokens = completion.usage.prompt_tokens
                output_tokens = completion.usage.completion_tokens
                cost = self.config.ai.estimate_cost(input_tokens, output_tokens)
                self.context.ai_usage.add_call(input_tokens, output_tokens, cost)
            
            # Split response by separator
            # Remove empty strings from split
            parts = [p.strip() for p in response_text.split(self.IMAGE_SEPARATOR) if p.strip()]
            
            # Parse each part
            results = []
            for part in parts:
                results.append(parse_ai_response(part))
                
            return results
            
        except Exception as e:
            self.log_error(f"Groq API call failed: {e}")
            raise # Re-raise to trigger batch failure handling

    def _call_ai_api_single(self, image_path: Path, image_name: str = "") -> List[Dict[str, Any]]:
        """
        Send a single image to Groq API with retry logic.
        
        Args:
            image_path: Path to the image file
            image_name: Name of the image for logging (captures name before thread execution)
            
        Returns:
            List of voter dictionaries extracted from the image
            
        Raises:
            Exception: If all retries are exhausted
        """
        # Use provided image_name or extract from path
        if not image_name:
            image_name = image_path.name
        
        # Simplified prompt for single image
        prompt_text = (
            "Extract voter information from this electoral roll image.\n\n"
            "Format:\n"
            "items[N]{serial_no,epic_no,name,relation_type,relation_name,house_no,age,gender}:\n"
            "  31,NHH3334638,kumar,father,arumugam,40NM,37,Male\n"
            "  ...\n\n"
            "Rules:\n"
            "1. Output field names map:\n"
            "   - serial_no (keep original serial number from the image)\n"
            "   - epic_no (IMPORTANT: Extract exactly 3 CAPITAL LETTERS followed by 7 DIGITS. Example: ABC1234567. If it's 4 letters, use only the last 3 letters + 7 digits.)\n"
            "   - name (Name / பெயர்)\n"
            "   - relation_type (infer: father/husband/mother/other)\n"
            "   - relation_name (Father/Husband Name / தந்தை/கணவர்/தாய் பெயர்)\n"
            "   - house_no (House No / வீட்டு எண்) - Look for 'வீட்டு எண்:'\n"
            "   - age (Age / வயது)\n"
            "   - gender (Gender / பாலினம்)\n"
            "2. Values can be in Tamil if the input text is in Tamil.\n"
            "3. Use comma (,) as delimiter. Replace commas inside values with space.\n"
            "4. NO Markdown code blocks. Just the raw text."
        )

        b64_str = self._encode_image(image_path)
        
        content = [
            {
                "type": "text", 
                "text": prompt_text
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_str}"
                }
            }
        ]
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": content}],
                    model=self.model,
                    temperature=0, 
                    max_tokens=2000,  # Single image needs less tokens
                )
                
                response_text = completion.choices[0].message.content
                
                # Debug: Log the raw response with correct image name
                self.log_debug(f"API response for {image_name}: {response_text[:500] if response_text else 'EMPTY'}...")
                
                # Track usage (thread-safe via context.ai_usage and context.stats.ai_usage)
                if hasattr(completion, 'usage') and completion.usage:
                    input_tokens = completion.usage.prompt_tokens
                    output_tokens = completion.usage.completion_tokens
                    cost = self.config.ai.estimate_cost(input_tokens, output_tokens)
                    # Update both usage trackers
                    self.context.ai_usage.add_call(input_tokens, output_tokens, cost)
                    self.context.stats.ai_usage.add_call(input_tokens, output_tokens, cost)
                
                # Parse response directly (no separator needed for single image)
                parsed = parse_ai_response(response_text)
                
                # Debug: Log parse result with voter details
                self.log_debug(f"Parsed {len(parsed)} voters from {image_name}")
                if parsed:
                    # Log first few voters for debugging
                    for idx, v in enumerate(parsed[:3]):
                        self.log_debug(f"  Voter {idx+1}: serial_no={v.get('serial_no', 'N/A')}, epic={v.get('epic_no', 'N/A')}, name={v.get('name', 'N/A')}")
                    if len(parsed) > 3:
                        self.log_debug(f"  ... and {len(parsed) - 3} more voters")
                
                return parsed
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if this is a retryable error
                is_rate_limit = "429" in str(e) or "rate" in error_str or "limit" in error_str
                is_server_error = "500" in str(e) or "502" in str(e) or "503" in str(e) or "504" in str(e)
                is_timeout = "timeout" in error_str or "timed out" in error_str
                is_connection_error = "connection" in error_str or "network" in error_str
                
                is_retryable = is_rate_limit or is_server_error or is_timeout or is_connection_error
                
                if attempt < self.max_retries and is_retryable:
                    # Calculate delay with exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    
                    # For rate limits, use longer delay
                    if is_rate_limit:
                        delay = max(delay, 5.0)  # Minimum 5 seconds for rate limits
                    
                    self.log_warning(f"Retryable error for {image_name} (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                elif attempt < self.max_retries:
                    # Non-retryable error, but we still have attempts - log and try again
                    delay = self.retry_delay * (2 ** attempt)
                    self.log_warning(f"Error for {image_name} (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    # All retries exhausted
                    self.log_error(f"Groq API call failed for {image_name} after {self.max_retries + 1} attempts: {e}")
                    raise  # Re-raise to trigger failure handling

    def _encode_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_all_voters(self) -> List[Voter]:
        """Return all extracted voters."""
        return self._all_voters

    def _track_failed_image(self, image_path: Path, page_id: str, image_index: int, error_message: str):
        """Thread-safe tracking of failed images."""
        failed_image = FailedImage(
            image_path=image_path,
            page_id=page_id,
            image_index=image_index,
            error_message=error_message,
            retry_count=self.max_retries + 1,  # Already exhausted all retries
            last_error_time=time.time()
        )
        
        with self._failed_images_lock:
            self._failed_images.append(failed_image)
    
    def get_failed_images(self) -> List[FailedImage]:
        """Get list of all failed images from the last processing run.
        
        Returns:
            List of FailedImage objects containing details about each failure
        """
        with self._failed_images_lock:
            return list(self._failed_images)
    
    def get_failed_pages(self) -> List[str]:
        """Get list of page IDs that have at least one failed image.
        
        Returns:
            Sorted list of unique page IDs with failures
        """
        with self._failed_images_lock:
            return sorted(set(fi.page_id for fi in self._failed_images))
    
    def has_failures(self) -> bool:
        """Check if there are any failed images from the last run.
        
        Returns:
            True if there are failed images, False otherwise
        """
        with self._failed_images_lock:
            return len(self._failed_images) > 0
    
    def clear_failed_images(self):
        """Clear the list of tracked failed images."""
        with self._failed_images_lock:
            self._failed_images.clear()
    
    def retry_failed_images(self, max_additional_retries: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """Retry processing only the failed images from the last run.
        
        This method allows you to reprocess only images that failed, without
        reprocessing the entire document. Useful for transient failures.
        
        Args:
            max_additional_retries: Number of additional retry attempts for each image.
                                   If None, uses the configured max_retries value.
        
        Returns:
            Dictionary mapping page_id to list of voters extracted from retried images.
            Empty list for images that still failed.
        """
        with self._failed_images_lock:
            failed_list = list(self._failed_images)
        
        if not failed_list:
            self.log_info("No failed images to retry.")
            return {}
        
        self._initialize_client()
        
        if max_additional_retries is None:
            max_additional_retries = self.max_retries
        
        # Store original max_retries and temporarily override
        original_max_retries = self.max_retries
        self.max_retries = max_additional_retries
        
        self.log_info(f"Retrying {len(failed_list)} failed images with {max_additional_retries} additional attempts each...")
        
        results = defaultdict(list)
        still_failed = []
        success_count = 0
        
        overall_start = time.perf_counter()
        
        # Process failed images with limited concurrency
        with ThreadPoolExecutor(max_workers=min(self.ai_batch_size, len(failed_list))) as executor:
            future_to_failed = {}
            
            for failed_image in failed_list:
                future = executor.submit(self._process_single_image, failed_image.image_path)
                future_to_failed[future] = failed_image
            
            for future in as_completed(future_to_failed):
                failed_image = future_to_failed[future]
                
                try:
                    voters = future.result()
                    results[failed_image.page_id].extend(voters)
                    success_count += 1
                    self.log_info(f"✓ Retry successful for {failed_image.image_path.name}: {len(voters)} voters")
                except Exception as e:
                    self.log_error(f"✗ Retry failed for {failed_image.image_path.name}: {e}")
                    failed_image.retry_count += max_additional_retries + 1
                    failed_image.error_message = str(e)
                    failed_image.last_error_time = time.time()
                    still_failed.append(failed_image)
        
        # Restore original max_retries
        self.max_retries = original_max_retries
        
        overall_elapsed = time.perf_counter() - overall_start
        
        # Update failed images list
        with self._failed_images_lock:
            self._failed_images = still_failed
        
        self.log_info(f"Retry completed in {overall_elapsed:.2f}s: "
                      f"{success_count}/{len(failed_list)} recovered, "
                      f"{len(still_failed)} still failing")
        
        return dict(results)
    
    def save_failed_images_report(self, output_path: Path = None) -> Path:
        """Save a JSON report of failed images for debugging or later retry.
        
        Args:
            output_path: Path to save the report. If None, saves to output directory.
            
        Returns:
            Path to the saved report file
        """
        import json
        
        with self._failed_images_lock:
            failed_list = list(self._failed_images)
        
        if not failed_list:
            self.log_info("No failed images to report.")
            return None
        
        if output_path is None:
            output_dir = self.context.extracted_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "failed_images.json"
        
        report = {
            "total_failed": len(failed_list),
            "affected_pages": sorted(set(fi.page_id for fi in failed_list)),
            "failed_images": [
                {
                    "image_path": str(fi.image_path),
                    "page_id": fi.page_id,
                    "image_index": fi.image_index,
                    "error_message": fi.error_message,
                    "retry_count": fi.retry_count,
                    "last_error_time": fi.last_error_time
                }
                for fi in failed_list
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.log_info(f"Failed images report saved to: {output_path}")
        return output_path
    
    def load_failed_images(self, report_path: Path) -> int:
        """Load failed images from a previously saved report.
        
        This allows retrying failed images from a previous session.
        
        Args:
            report_path: Path to the failed images JSON report
            
        Returns:
            Number of failed images loaded
        """
        import json
        
        if not report_path.exists():
            self.log_error(f"Report file not found: {report_path}")
            return 0
        
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        loaded_count = 0
        with self._failed_images_lock:
            for item in report.get("failed_images", []):
                image_path = Path(item["image_path"])
                if image_path.exists():
                    failed_image = FailedImage(
                        image_path=image_path,
                        page_id=item["page_id"],
                        image_index=item["image_index"],
                        error_message=item.get("error_message", "Unknown"),
                        retry_count=item.get("retry_count", 0),
                        last_error_time=item.get("last_error_time", time.time())
                    )
                    self._failed_images.append(failed_image)
                    loaded_count += 1
                else:
                    self.log_warning(f"Image file not found, skipping: {image_path}")
        
        self.log_info(f"Loaded {loaded_count} failed images from report")
        return loaded_count
