"""
Metadata Extractor processor.

Extracts document metadata from Electoral Roll PDFs using AI vision.
Analyzes front page to extract:
- Assembly/Parliamentary constituency info
- Revision information
- Languages
"""

from __future__ import annotations

import base64
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any, Tuple

import cv2
import numpy as np
from openai import OpenAI

from .base import BaseProcessor, ProcessingContext
from ..config import Config
from ..models import DocumentMetadata, AIUsage
from ..exceptions import MetadataExtractionError

# Add project root to path for ROI imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rois import PINCODE_ROI, VOTERS_END_ROI


# Supported image extensions
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageScore:
    """Scoring for back page selection."""
    path: Path
    ink_ratio: float
    table_line_ratio: float


    

class MetadataExtractor(BaseProcessor):
    """
    Extract metadata from Electoral Roll PDFs using AI.
    
    Sends front page image to a multimodal AI model to extract
    structured metadata.
    """
    
    name = "MetadataExtractor"
    
    def __init__(
        self,
        context: ProcessingContext,
        prompt_path: Optional[Path] = None,
        force: bool = False,
        output_identifier: Optional[str] = None,
    ):
        """
        Initialize extractor.
        
        Args:
            context: Processing context
            prompt_path: Path to prompt file (default: prompt_flat.md in project root)
            force: Overwrite existing metadata
            output_identifier: Optional identifier to include in metadata
        """
        super().__init__(context)
        self.prompt_path = prompt_path or self.config.base_dir / "prompt_flat.md"
        self.force = force
        self.output_identifier = output_identifier
        self.result: Optional[DocumentMetadata] = None
    
    def validate(self) -> bool:
        """Validate prerequisites."""
        if not self.context.images_dir:
            self.log_error("Images directory not set")
            return False
        
        if not self.context.images_dir.exists():
            self.log_error(f"Images directory not found: {self.context.images_dir}")
            return False
        
        if not self.prompt_path.exists():
            self.log_error(f"Prompt file not found: {self.prompt_path}")
            return False
        
        # Check for existing output
        if not self.force:
            output_path = self._get_output_path()
            if output_path.exists():
                self.log_info("Metadata already exists, skipping (use --force to override)")
                return False
        
        return True
    
    def _get_output_path(self) -> Path:
        """Get output path for metadata JSON."""
        return self.context.output_dir / f"{self.context.pdf_name}-metadata.json"
    
    def process(self) -> bool:
        """
        Extract metadata from document.
        
        Returns:
            True if extraction succeeded
        """
        images_dir = self.context.images_dir
        
        # Get sorted images
        images = self._get_sorted_images(images_dir)
        
        if len(images) < 1:
            self.log_warning(f"Need at least 1 image, found {len(images)}")
            return False
        
        # Select front page only
        front_page = images[0]
        
        self.log_info(f"Selected page: front={front_page.name}")
        
        # Crop and stitch ROIs from front page (pincode + voters_end regions only)
        cropped_image = self._crop_and_stitch_rois(front_page)
        if cropped_image is None:
            self.log_error("Failed to crop ROIs from front page")
            return False
        
        self.log_info("Created cropped ROI image for AI extraction")
        
        # Load prompt
        prompt_text = self.prompt_path.read_text(encoding="utf-8")
        
        # Call AI with timing and retry
        start_time = time.time()
        max_retries = self.config.ai.max_retries
        retry_delay = self.config.ai.retry_delay_sec
        
        content = None
        ai_meta = None
        parsed_result = None
        
        for attempt in range(max_retries + 1):
            try:
                content, ai_meta = self._call_ai(
                    prompt_text=prompt_text,
                    cropped_image=cropped_image,
                )
                
                # Parse to validate
                try:
                    current_parsed = self._extract_json(content)
                    
                    # Check completeness
                    if self._is_complete(current_parsed):
                        parsed_result = current_parsed
                        break
                    
                    # If not complete, treat as failure unless it's the last attempt
                    if attempt < max_retries:
                        self.log_warning(f"Metadata incomplete (attempt {attempt + 1}), retrying...")
                        # Save result anyway in case next attempts fail worse or crash
                        parsed_result = current_parsed
                        
                        wait = retry_delay * (2 ** attempt)
                        time.sleep(wait)
                        continue
                    else:
                        parsed_result = current_parsed
                        
                except Exception as parse_e:
                    # If parsing fails, that's an error
                    if attempt < max_retries:
                        self.log_warning(f"Failed to parse AI response (attempt {attempt + 1}), retrying...", error=parse_e)
                        wait = retry_delay * (2 ** attempt)
                        time.sleep(wait)
                        continue
                    raise parse_e

            except Exception as e:
                is_last = attempt == max_retries
                if is_last:
                    self.log_error(f"AI call failed after {max_retries + 1} attempts", error=e)
                    self._save_error(str(e), front_page, None)
                    return False
                
                wait = retry_delay * (2 ** attempt)
                self.log_warning(
                    f"AI call failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait:.1f}s...",
                    error=e
                )
                time.sleep(wait)
        
        extraction_time_sec = time.time() - start_time
        if ai_meta:
            ai_meta["extraction_time_sec"] = round(extraction_time_sec, 2)
        
        parsed = parsed_result

        
        # IMPORTANT: Always overwrite ai_metadata field from AI response
        # The AI might return an ai_metadata field in its JSON, but we need our tracked data
        if isinstance(parsed, dict):
            parsed["ai_metadata"] = self._flatten_ai_metadata(ai_meta)
            
            # Add output identifier if present
            if self.output_identifier:
                parsed["output_identifier"] = self.output_identifier
        
        # Update context AI usage
        if ai_meta.get("usage"):
            usage = ai_meta["usage"]
            self.context.ai_usage = AIUsage(
                model=ai_meta.get("model", ""),
                provider=ai_meta.get("provider", ""),
            )
            # Add the call with token counts
            self.context.ai_usage.add_call(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                cost_usd=ai_meta.get("estimated_cost"),
            )
        
        # Save output
        output_path = self._get_output_path()
        output_path.write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        # Create DocumentMetadata from parsed with AI metadata
        flattened_ai_meta = self._flatten_ai_metadata(ai_meta)
        self.result = DocumentMetadata.from_ai_response(
            parsed,
            ai_provider=flattened_ai_meta.get("provider", ""),
            ai_model=flattened_ai_meta.get("model", ""),
            input_tokens=flattened_ai_meta.get("input_tokens", 0),
            output_tokens=flattened_ai_meta.get("output_tokens", 0),
            cost_usd=flattened_ai_meta.get("cost_usd"),
            extraction_time_sec=flattened_ai_meta.get("extraction_time_sec", 0.0),
        )
        
        self.log_info(f"Saved metadata to {output_path.name}")
        
        return True
    
    def _get_sorted_images(self, images_dir: Path) -> List[Path]:
        """Get sorted list of images."""
        images = [
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        return sorted(images, key=lambda p: p.name.lower())
    
    def _pick_back_page(self, images: List[Path], max_lookback: int = 6) -> Path:
        """
        Select the best back page from image list.
        
        Sometimes the last image is blank or lacks the summary table.
        This method scores candidates by ink density and table line presence.
        
        Args:
            images: List of page images
            max_lookback: Maximum pages to scan from end
        
        Returns:
            Best back page image path
        """
        if not images:
            raise ValueError("No images provided")
        if len(images) == 1:
            return images[0]
        
        # Scoring thresholds
        ink_min = 0.006  # ~0.6% pixels are ink
        table_min = 0.002  # ~0.2% pixels are line structures
        
        candidates = list(reversed(images[-max_lookback:]))
        scored: List[ImageScore] = []
        
        for img_path in candidates:
            try:
                score = self._score_back_page(img_path)
            except Exception:
                continue
            
            scored.append(score)
            
            # Prefer first strong match scanning backwards
            if score.ink_ratio >= ink_min and score.table_line_ratio >= table_min:
                self.log_debug(
                    f"Selected back page: {img_path.name}",
                    ink=f"{score.ink_ratio:.4f}",
                    table=f"{score.table_line_ratio:.4f}"
                )
                return img_path
        
        # Fallback: choose best by table_line_ratio, then ink
        if scored:
            scored_sorted = sorted(
                scored,
                key=lambda s: (s.table_line_ratio, s.ink_ratio),
                reverse=True
            )
            best = scored_sorted[0]
            
            # If best is basically blank, use second-to-last
            if best.ink_ratio < ink_min and len(images) >= 2:
                return images[-2]
            return best.path
        
        # If scoring fails entirely, use second-to-last
        return images[-2]

    def _is_complete(self, data: dict) -> bool:
        """
        Check if critical metadata fields are present.
        
        Supports both NEW FLAT and OLD NESTED structures.
        """
        if not data:
            return False
        
        # Detect if this is the NEW FLAT structure
        is_flat = (
            "language_detected" in data and
            "total" in data and
            "pin_code" in data and
            "document_metadata" not in data
        )
        
        if is_flat:
            # NEW FLAT STRUCTURE validation (only 3 required fields - state removed since cropped ROIs don't contain it)
            langs = data.get("language_detected")
            if not langs or len(langs) == 0:
                self.log_warning("Language detected field is empty or missing")
                return False
            
            # Only 3 fields required (state is nullable since cropped ROIs don't contain it)
            required_flat_fields = ["language_detected", "pin_code", "total"]
            for field in required_flat_fields:
                value = data.get(field)
                # Allow 0 as valid, but not None or empty string
                if value is None or (isinstance(value, str) and not value):
                    self.log_warning(f"Flat structure missing or empty field: {field}")
                    return False
            
            # Validate total is a number
            total = data.get("total")
            if not isinstance(total, (int, float)) or total < 0:
                self.log_warning(f"Total must be a non-negative number, got: {total}")
                return False
            
            return True
        
        # OLD NESTED STRUCTURE validation (backward compatibility)
        constituency = data.get("constituency_details", {})
        if not constituency:
            return False

        # Validate Language Detection (Critical)
        doc_meta = data.get("document_metadata", {})
        if not doc_meta:
            doc_meta = data
        
        langs = doc_meta.get("language_detected")
        if not langs or len(langs) == 0:
            self.log_warning("Language detected field is empty or missing")
            return False
            
        # Critical fields that should not be empty
        critical_fields = [
            constituency.get("assembly_constituency_name"),
            constituency.get("assembly_constituency_number"),
            constituency.get("part_number")
        ]

        # Check for detailed elector summary (Crucial)
        summary = data.get("detailed_elector_summary", {})
        
        # Validate Serial Number Range
        serial_range = summary.get("serial_number_range", {})
        if serial_range.get("start") is None or serial_range.get("end") is None:
             self.log_warning(f"Detailed elector summary range incomplete: {serial_range}")
             critical_fields.append(None)

        # Validate Net Total
        net_total = summary.get("net_total", {})
        required_stats = ["male", "female", "third_gender", "total"]
        for f in required_stats:
             if net_total.get(f) is None:
                 self.log_warning(f"Detailed elector summary missing field: net_total.{f}")
                 critical_fields.append(None)

        
        for field in critical_fields:
            if not field and field != 0: # 0 is a valid number, but usually these are strings or >0 ints
                return False
                
        return True

    
    def _score_back_page(self, image_path: Path) -> ImageScore:
        """
        Score a candidate back page.
        
        Args:
            image_path: Path to image
        
        Returns:
            Score with ink and table line ratios
        """
        # Read and downscale
        img = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Downscale for speed
        h, w = img.shape[:2]
        target_w = 900
        if w > target_w:
            scale = target_w / float(w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        # Binarize
        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Ink ratio
        ink = (bw < 250).astype(np.uint8)
        ink_ratio = float(ink.mean())
        
        # Detect table lines
        inv = 255 - bw
        h, w = inv.shape[:2]
        
        hk = max(20, w // 30)
        vk = max(20, h // 30)
        
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
        
        horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
        vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)
        lines = cv2.bitwise_or(horiz, vert)
        
        table_line_ratio = float((lines > 0).mean())
        
        return ImageScore(
            path=image_path,
            ink_ratio=ink_ratio,
            table_line_ratio=table_line_ratio
        )
    
    def _denormalize_roi(self, roi: Tuple[float, float, float, float], h: int, w: int) -> Tuple[int, int, int, int]:
        """Convert normalized ROI coordinates to pixel coordinates."""
        x1, y1, x2, y2 = roi
        return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    
    def _remove_table_lines_inpaint(self, bgr_img: np.ndarray) -> np.ndarray:
        """Remove table/grid lines via masking + inpainting; tends to preserve digits better."""
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        bw = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            10,
        )

        horiz_len = max(25, min(120, bgr_img.shape[1] // 2))
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
        horizontal = cv2.erode(bw, horiz_kernel, iterations=1)
        horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=1)

        vert_len = max(25, min(120, bgr_img.shape[0] // 2))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
        vertical = cv2.erode(bw, vert_kernel, iterations=1)
        vertical = cv2.dilate(vertical, vert_kernel, iterations=1)

        line_mask = cv2.bitwise_or(horizontal, vertical)
        line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)

        cleaned = cv2.inpaint(bgr_img, line_mask, 3, cv2.INPAINT_TELEA)
        return cleaned
    
    def _crop_and_stitch_rois(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Crop pincode and voters_end ROIs from image and stitch them together.
        
        Args:
            image_path: Path to the front page image
            
        Returns:
            Stitched BGR image containing both ROIs, or None if failed
        """
        try:
            # Read image
            img = cv2.imdecode(
                np.fromfile(str(image_path), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is None:
                self.log_error(f"Failed to read image: {image_path}")
                return None
            
            h, w = img.shape[:2]
            
            # Denormalize ROI coordinates
            pincode_roi_px = self._denormalize_roi(PINCODE_ROI, h, w)
            voters_roi_px = self._denormalize_roi(VOTERS_END_ROI, h, w)
            
            # Crop ROIs
            x1, y1, x2, y2 = pincode_roi_px
            pincode_roi = img[y1:y2, x1:x2].copy()
            
            x1, y1, x2, y2 = voters_roi_px
            voters_roi = img[y1:y2, x1:x2].copy()
            # Clean table lines from voters ROI
            voters_roi = self._remove_table_lines_inpaint(voters_roi)
            
            # Pad to same height for horizontal concatenation
            target_h = max(pincode_roi.shape[0], voters_roi.shape[0])
            
            def pad_to_h(bgr: np.ndarray, new_h: int) -> np.ndarray:
                if bgr.shape[0] == new_h:
                    return bgr
                pad = new_h - bgr.shape[0]
                return cv2.copyMakeBorder(bgr, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            
            pincode_roi = pad_to_h(pincode_roi, target_h)
            voters_roi = pad_to_h(voters_roi, target_h)
            
            # Create separator and stitch
            sep_w = 28
            separator = np.ones((target_h, sep_w, 3), dtype=np.uint8) * 255
            stitched = cv2.hconcat([pincode_roi, separator, voters_roi])
            
            # Upscale for better AI readability
            stitched = cv2.resize(stitched, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Optionally save debug image
            debug_path = self.context.output_dir / f"{self.context.pdf_name}-roi-crop.png"
            cv2.imwrite(str(debug_path), stitched)
            self.log_debug(f"Saved ROI crop debug image: {debug_path.name}")
            
            return stitched
            
        except Exception as e:
            self.log_error(f"Failed to crop and stitch ROIs: {e}")
            return None
    
    def _encode_image_array(self, bgr_array: np.ndarray) -> str:
        """Encode BGR numpy array as data URL."""
        ok, buf = cv2.imencode(".png", bgr_array)
        if not ok:
            raise RuntimeError("Failed to PNG-encode image array")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    
    def _call_ai(
        self,
        prompt_text: str,
        cropped_image: np.ndarray,
    ) -> Tuple[str, dict[str, Any]]:
        """
        Call AI model with cropped ROI image.
        
        Args:
            prompt_text: System prompt
            cropped_image: Cropped and stitched ROI image (BGR numpy array)
        
        Returns:
            Tuple of (response content, AI metadata)
        """
        ai_config = self.config.ai
        
        if not ai_config.api_key:
            raise RuntimeError("Missing AI_API_KEY environment variable")
        
        self.log_debug(
            f"Calling AI",
            model=ai_config.model,
            provider=ai_config.provider or "default"
        )
        
        # Get normalized base URL for OpenAI SDK
        base_url = ai_config.get_normalized_base_url() or None
        
        client = OpenAI(
            api_key=ai_config.api_key,
            base_url=base_url,
            timeout=ai_config.timeout_sec,
        )
        
        payload: dict[str, Any] = {
            "model": ai_config.model,
            "messages": [
                {"role": "system", "content": prompt_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Cropped ROI image containing pincode and voters_end regions."},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._encode_image_array(cropped_image)}
                        },
                    ],
                },
            ],
        }
        
        if ai_config.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        
        resp = client.chat.completions.create(**payload)
        
        # Extract content
        try:
            content = str(resp.choices[0].message.content)
        except Exception as e:
            raise RuntimeError(f"Unexpected response shape: {e}")
        
        # Extract usage
        usage: Optional[dict[str, int]] = None
        try:
            u = getattr(resp, "usage", None)
            if u is not None:
                usage = {
                    "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(u, "total_tokens", 0) or 0),
                }
                self.log_debug(
                    "Usage extracted",
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"]
                )
            else:
                self.log_warning("No usage data in AI response")
        except Exception as e:
            self.log_warning(f"Failed to extract usage data: {e}")
            usage = None
        
        # Estimate cost
        estimated_cost = self._estimate_cost(usage, ai_config)
        
        # Store full metadata (will be flattened later)
        ai_meta = {
            "provider": ai_config.provider or "",
            "model": ai_config.model or "",
            "usage": usage,
            "estimated_cost": estimated_cost,
            "base_url": ai_config.base_url,
            "pricing": {
                "currency": ai_config.cost_currency,
                "input_cost_per_1m_tokens": ai_config.input_cost_per_1m,
                "output_cost_per_1m_tokens": ai_config.output_cost_per_1m,
            },
        }
        
        return content, ai_meta
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image as data URL."""
        raw = image_path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        ext = image_path.suffix.lower().lstrip(".")
        
        if ext == "png":
            mime = "image/png"
        elif ext in {"jpg", "jpeg"}:
            mime = "image/jpeg"
        else:
            mime = f"image/{ext}"
        
        return f"data:{mime};base64,{b64}"
    
    def _estimate_cost(
        self,
        usage: Optional[dict[str, int]],
        ai_config,
    ) -> Optional[float]:
        """Estimate AI call cost."""
        if not usage:
            return None
        
        in_cost = ai_config.input_cost_per_1m
        out_cost = ai_config.output_cost_per_1m
        
        if in_cost is None and out_cost is None:
            return None
        
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        total = 0.0
        if in_cost is not None:
            total += (prompt_tokens / 1_000_000.0) * in_cost
        if out_cost is not None:
            total += (completion_tokens / 1_000_000.0) * out_cost
        
        return total
    
    def _extract_json(self, text: str) -> Any:
        """
        Extract first JSON object/array from text.
        
        Handles common patterns like ```json ... ``` wrappers.
        """
        if text is None:
            raise ValueError("Empty response")
        
        t = text.strip()
        
        # Strip fenced code blocks
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
            t = re.sub(r"\s*```$", "", t).strip()
        
        # Fast path: full JSON
        try:
            return json.loads(t)
        except Exception:
            pass
        
        # Scan for balanced braces/brackets
        start_candidates = [i for i, ch in enumerate(t) if ch in "[{"]
        
        for start in start_candidates:
            stack: List[str] = []
            in_str = False
            escape = False
            
            for i in range(start, len(t)):
                ch = t[i]
                
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                
                if ch == '"':
                    in_str = True
                    continue
                
                if ch in "[{":
                    stack.append(ch)
                elif ch in "]}":
                    if not stack:
                        break
                    opener = stack.pop()
                    if (opener == "[" and ch != "]") or (opener == "{" and ch != "}"):
                        break
                    if not stack:
                        snippet = t[start:i + 1]
                        try:
                            return json.loads(snippet)
                        except Exception:
                            break
        
        raise ValueError("Could not parse JSON from model response")
    
    def _flatten_ai_metadata(self, ai_meta: dict[str, Any]) -> dict[str, Any]:
        """Flatten AI metadata to simplified structure."""
        usage = ai_meta.get("usage") or {}
        
        return {
            "provider": ai_meta.get("provider") or "",
            "model": ai_meta.get("model") or "",
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cost_usd": ai_meta.get("estimated_cost"),
            "extraction_time_sec": ai_meta.get("extraction_time_sec", 0.0),
        }
    
    def _save_error(
        self,
        error: str,
        front: Path,
        content: Optional[str]
    ) -> None:
        """Save error information for debugging."""
        raw_path = self.context.output_dir / f"{self.context.pdf_name}.raw.txt"
        
        try:
            with raw_path.open("w", encoding="utf-8") as f:
                f.write(f"ERROR: {error}\n")
                f.write(f"front={front}\n")
                if content:
                    f.write("\n--- MODEL OUTPUT (raw) ---\n")
                    f.write(content)
        except Exception:
            pass


def extract_metadata(
    extracted_dir: Path,
    prompt_path: Optional[Path] = None,
    force: bool = False,
    output_identifier: Optional[str] = None,
) -> Optional[DocumentMetadata]:
    """
    Convenience function to extract metadata from an extracted folder.
    
    Args:
        extracted_dir: Path to extracted folder
        prompt_path: Path to prompt file
        force: Overwrite existing metadata
    
    Returns:
        Extracted metadata, or None if failed
    """
    from ..config import Config
    
    config = Config()
    context = ProcessingContext(config=config)
    context.setup_paths_from_extracted(extracted_dir)
    
    extractor = MetadataExtractor(context, prompt_path=prompt_path, force=force, output_identifier=output_identifier)
    
    if not extractor.run():
        return None
    
    return extractor.result
