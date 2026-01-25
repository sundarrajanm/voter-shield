"""
AI-based field extraction and DELETED mark detector.

Sends a voter crop image to AI to:
1. Extract the age if readable
2. Detect if it contains a "DELETED" mark
"""

import base64
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("ai_deleted_detector")

# Try to import OpenAI SDK
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI SDK not available - AI deleted detection disabled")


@dataclass
class AIFieldResult:
    """Result from AI field extraction."""
    age: str = ""  # Extracted age, empty if not readable
    deleted: bool = False  # True if DELETED mark detected
    epic_no: str = ""  # Extracted EPIC number, empty if not readable


def extract_fields_with_ai(
    image_path: Optional[Path] = None,
    image_array: Optional[np.ndarray] = None,
    api_key: str = "",
    base_url: str = "",
    model: str = "gemini-2.0-flash",
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> AIFieldResult:
    """
    Send voter crop image to AI to extract age and detect DELETED mark.
    
    Args:
        image_path: Path to the crop image file
        image_array: Alternatively, numpy array of the image (BGR format)
        api_key: AI API key
        base_url: AI API base URL
        model: Model name to use
        max_retries: Maximum number of retry attempts on failure
        retry_delay: Delay between retries in seconds
        
    Returns:
        AIFieldResult with age and deleted status
    """
    result = AIFieldResult()
    
    if not OPENAI_AVAILABLE:
        logger.debug("OpenAI SDK not available, skipping AI extraction")
        return result
    
    if not api_key:
        logger.debug("No API key provided, skipping AI extraction")
        return result
    
    # Get image data
    if image_path is not None:
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
        except Exception as e:
            logger.debug(f"Failed to read image file: {e}")
            return result
    elif image_array is not None:
        try:
            # Encode numpy array to PNG bytes
            success, buffer = cv2.imencode('.png', image_array)
            if not success:
                logger.debug("Failed to encode image array")
                return result
            image_data = buffer.tobytes()
        except Exception as e:
            logger.debug(f"Failed to encode image array: {e}")
            return result
    else:
        logger.debug("No image provided")
        return result
    
    # Encode to base64
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    # Initialize client
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None,
        )
    except Exception as e:
        logger.debug(f"Failed to initialize OpenAI client: {e}")
        return result
    
    # Prompt asking for age, epic number, and deleted status
    prompt = """Look at this voter information image from an Indian electoral roll.

1. Find the EPIC NUMBER field - it's typically a combination of 3 letters followed by 7 digits (e.g., ABC1234567)
2. Find the AGE (வயது) field and extract the age number
3. Check if there is a "DELETED" stamp/mark covering the voter information

Reply in this exact format (nothing else):
epic: <epic number or empty>
age: <number or empty>
deleted: <true or false>

Example responses:
epic: ABC1234567
age: 45
deleted: false

epic: 
age: 
deleted: true"""

    # Retry loop for network/server errors
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
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
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=50,
                temperature=0.0,
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip().lower()
            
            # Extract EPIC number
            epic_match = re.search(r'epic:\s*([A-Z]{3}\d{7,10})?', response_text, re.IGNORECASE)
            if epic_match and epic_match.group(1):
                result.epic_no = epic_match.group(1).upper()
                logger.debug(f"AI extracted EPIC: {result.epic_no}")
            
            # Extract age
            age_match = re.search(r'age:\s*(\d+)?', response_text)
            if age_match and age_match.group(1):
                result.age = age_match.group(1)
                logger.debug(f"AI extracted age: {result.age}")
            
            # Extract deleted status
            deleted_match = re.search(r'deleted:\s*(true|false)', response_text)
            if deleted_match:
                result.deleted = deleted_match.group(1) == "true"
                if result.deleted:
                    logger.debug("AI detected DELETED mark")
            
            # Success - return result
            return result
            
        except Exception as e:
            last_error = e
            attempt_num = attempt + 1
            
            # Check if it's a retryable error (network, server, rate limit)
            error_str = str(e).lower()
            is_retryable = any(term in error_str for term in [
                'timeout', 'connection', 'network', 'server', 
                '500', '502', '503', '504', '429', 'rate limit',
                'overloaded', 'temporarily unavailable'
            ])
            
            if is_retryable and attempt_num < max_retries:
                logger.warning(f"AI API call failed (attempt {attempt_num}/{max_retries}): {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Exponential backoff
                retry_delay *= 1.5
            else:
                # Non-retryable error or max retries reached
                logger.debug(f"AI API call failed: {e}")
                break
    
    # All retries exhausted
    if last_error:
        logger.warning(f"AI extraction failed after {max_retries} attempts: {last_error}")
    
    return result


class AIDeletedDetector:
    """
    Reusable AI-based field extractor and DELETED mark detector.
    """
    
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "gemini-2.0-flash",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize detector with API configuration.
        
        Args:
            api_key: AI API key
            base_url: AI API base URL  
            model: Model name to use
            max_retries: Maximum retry attempts on failure
            retry_delay: Initial delay between retries (seconds)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None
        
    @property
    def client(self):
        """Lazy-initialize OpenAI client."""
        if self._client is None and OPENAI_AVAILABLE and self.api_key:
            try:
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url if self.base_url else None,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        return self._client
    
    def is_available(self) -> bool:
        """Check if AI detection is available."""
        return OPENAI_AVAILABLE and bool(self.api_key)
    
    def extract_fields(
        self,
        image_path: Optional[Path] = None,
        image_array: Optional[np.ndarray] = None,
    ) -> AIFieldResult:
        """
        Extract age and detect if image contains DELETED mark.
        
        Args:
            image_path: Path to image file
            image_array: Alternatively, numpy array (BGR)
            
        Returns:
            AIFieldResult with age and deleted status
        """
        if not self.is_available():
            return AIFieldResult()
            
        return extract_fields_with_ai(
            image_path=image_path,
            image_array=image_array,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )
