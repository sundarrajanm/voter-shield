"""
Base processor class and processing context.

Provides common functionality for all document processors including
logging, timing, error handling, and configuration access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

from ..config import Config
from ..logger import get_logger
from ..models import ProcessingStats, AIUsage
from ..utils.timing import Timer


@dataclass
class ProcessingContext:
    """
    Shared context passed between processors.
    
    Contains:
    - Configuration
    - Paths
    - Accumulated statistics
    - Shared state
    """
    
    config: Config
    pdf_path: Optional[Path] = None
    pdf_name: Optional[str] = None
    extracted_dir: Optional[Path] = None
    images_dir: Optional[Path] = None
    crops_dir: Optional[Path] = None
    crop_top_dir: Optional[Path] = None  # Directory for header/top section crops
    id_crops_dir: Optional[Path] = None
    id_merged_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    
    # Processing statistics
    stats: ProcessingStats = field(default_factory=ProcessingStats)
    
    # Page tracking
    current_page: Optional[str] = None
    total_pages: int = 0
    pages_processed: int = 0
    
    # Voter tracking
    total_voters_found: int = 0
    
    # AI usage tracking
    ai_usage: AIUsage = field(default_factory=AIUsage)
    
    def setup_paths_from_pdf(self, pdf_path: Path) -> None:
        """
        Initialize all paths from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = pdf_path
        self.pdf_name = pdf_path.stem
        self.extracted_dir = self.config.extracted_dir / self.pdf_name
        self.images_dir = self.extracted_dir / "images"
        self.crops_dir = self.extracted_dir / "crops"
        self.crop_top_dir = self.extracted_dir / "crop-top"  # Header crops
        self.id_crops_dir = self.extracted_dir / "id-crops"
        self.id_merged_dir = self.extracted_dir / "id-merged"
        self.output_dir = self.extracted_dir / "output"
        
        # Ensure directories exist
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.crops_dir.mkdir(exist_ok=True)
        self.crop_top_dir.mkdir(exist_ok=True)
        self.id_crops_dir.mkdir(exist_ok=True)
        self.id_merged_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
    
    def setup_paths_from_extracted(self, extracted_dir: Path) -> None:
        """
        Initialize paths from an extracted directory.
        
        Args:
            extracted_dir: Path to extracted folder
        """
        self.extracted_dir = extracted_dir
        self.pdf_name = extracted_dir.name
        self.images_dir = extracted_dir / "images"
        self.crops_dir = extracted_dir / "crops"
        self.crop_top_dir = extracted_dir / "crop-top"  # Header crops
        self.id_crops_dir = extracted_dir / "id-crops"
        self.id_merged_dir = extracted_dir / "id-merged"
        self.output_dir = extracted_dir / "output"
        
        # Ensure directories exist
        self.crop_top_dir.mkdir(exist_ok=True)
        self.id_crops_dir.mkdir(exist_ok=True)
        self.id_merged_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)


class BaseProcessor(ABC):
    """
    Abstract base class for all document processors.
    
    Provides:
    - Consistent logging
    - Timing instrumentation
    - Error handling
    - Configuration access
    """
    
    # Processor name for logging (override in subclass)
    name: str = "BaseProcessor"
    
    def __init__(self, context: ProcessingContext):
        """
        Initialize processor.
        
        Args:
            context: Shared processing context
        """
        self.context = context
        self.config = context.config
        self.logger = get_logger(self.name)
        self._timer = Timer()
    
    @property
    def debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.config.debug
    
    def log_debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message (only in debug mode)."""
        if self.debug_mode:
            extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
            self.logger.debug(f"{message} {extra}".strip())
    
    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"{message} {extra}".strip())
    
    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.warning(f"{message} {extra}".strip())
    
    def log_error(self, message: str, error: Optional[Exception] = None) -> None:
        """Log error message."""
        if error:
            self.logger.error(f"{message}: {error}", exc_info=self.debug_mode)
        else:
            self.logger.error(message)
    
    @abstractmethod
    def process(self) -> bool:
        """
        Execute the processor's main task.
        
        Returns:
            True if processing succeeded, False otherwise
        """
        pass
    
    def validate(self) -> bool:
        """
        Validate that processor can run.
        
        Override in subclass to check prerequisites.
        
        Returns:
            True if validation passes
        """
        return True
    
    def run(self) -> bool:
        """
        Run processor with timing and error handling.
        
        Returns:
            True if processing succeeded
        """
        self.log_info(f"Starting {self.name}")
        self._timer = Timer()  # Create fresh timer to track from start
        
        try:
            if not self.validate():
                self.log_error("Validation failed")
                return False
            
            result = self.process()
            
            elapsed = self._timer.elapsed
            self.log_info(f"Completed {self.name}", duration=f"{elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            elapsed = self._timer.elapsed
            self.log_error(f"Failed after {elapsed:.2f}s", error=e)
            return False
    
    def save_debug_info(self, name: str, data: Any) -> Optional[Path]:
        """
        Save debug information to file.
        
        Only saves if debug mode is enabled.
        
        Args:
            name: Debug file name
            data: Data to save
        
        Returns:
            Path to saved file, or None if not saved
        """
        if not self.debug_mode or not self.context.output_dir:
            return None
        
        import json
        
        debug_dir = self.context.output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        
        debug_path = debug_dir / f"{name}.json"
        
        if isinstance(data, (dict, list)):
            debug_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
                errors="replace"
            )
        else:
            debug_path.write_text(str(data), encoding="utf-8", errors="replace")
        
        self.log_debug(f"Saved debug info to {debug_path}")
        return debug_path
