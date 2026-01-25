"""
Processing statistics and timing models.

Tracks performance metrics for processing operations.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Any
from datetime import datetime


@dataclass
class AIUsage:
    """AI API usage tracking (thread-safe)."""
    provider: str = ""
    model: str = ""
    calls_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    
    def add_call(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_usd: Optional[float] = None
    ) -> None:
        """Add a single API call to the usage stats (thread-safe)."""
        with self._lock:
            self.calls_count += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            if cost_usd is not None:
                self.total_cost_usd += cost_usd
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "calls_count": self.calls_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
        }


@dataclass
class PageTiming:
    """Timing information for a single page."""
    page_id: str = ""
    page_number: int = 0
    
    # Crop counts
    crops_detected: int = 0
    crops_saved: int = 0
    crops_skipped: int = 0
    
    # Voter counts
    voters_extracted: int = 0
    voters_valid: int = 0  # With valid EPIC
    
    # Timing breakdown (seconds)
    cropping_time_sec: float = 0.0
    ocr_time_sec: float = 0.0
    total_time_sec: float = 0.0
    
    # Per-image timing
    per_image_times_sec: List[float] = field(default_factory=list)
    
    @property
    def avg_time_per_image_sec(self) -> float:
        """Calculate average time per image."""
        if not self.per_image_times_sec:
            return 0.0
        return sum(self.per_image_times_sec) / len(self.per_image_times_sec)
    
    @property
    def avg_time_per_image_ms(self) -> float:
        """Calculate average time per image in milliseconds."""
        return self.avg_time_per_image_sec * 1000
    
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["avg_time_per_image_sec"] = round(self.avg_time_per_image_sec, 4)
        return data


@dataclass
class ProcessingStats:
    """
    Complete processing statistics for a document.
    
    Tracks timing, costs, and counts for all processing stages.
    """
    
    # Document identification
    document_id: str = ""
    pdf_name: str = ""
    
    # Timestamps
    started_at: str = ""  # ISO format
    completed_at: str = ""  # ISO format
    
    # Status
    status: str = "pending"  # pending, processing, completed, failed
    error_message: str = ""
    
    # Counts
    total_pages: int = 0
    pages_processed: int = 0
    total_crops: int = 0
    total_voters: int = 0
    valid_voters: int = 0  # With valid EPIC
    
    # Overall timing (seconds)
    extraction_time_sec: float = 0.0
    metadata_time_sec: float = 0.0
    cropping_time_sec: float = 0.0
    ocr_time_sec: float = 0.0
    total_time_sec: float = 0.0
    
    # AI usage
    ai_usage: AIUsage = field(default_factory=AIUsage)
    
    # Per-page details
    page_timings: List[PageTiming] = field(default_factory=list)
    
    def start(self) -> None:
        """Mark processing as started."""
        self.started_at = datetime.utcnow().isoformat() + "Z"
        self.status = "processing"
    
    def complete(self) -> None:
        """Mark processing as completed."""
        self.completed_at = datetime.utcnow().isoformat() + "Z"
        self.status = "completed"
        self._calculate_totals()
    
    def fail(self, error: str) -> None:
        """Mark processing as failed."""
        self.completed_at = datetime.utcnow().isoformat() + "Z"
        self.status = "failed"
        self.error_message = error
    
    def _calculate_totals(self) -> None:
        """Calculate aggregate totals from page timings."""
        self.pages_processed = len(self.page_timings)
        self.total_crops = sum(pt.crops_saved for pt in self.page_timings)
        self.total_voters = sum(pt.voters_extracted for pt in self.page_timings)
        self.valid_voters = sum(pt.voters_valid for pt in self.page_timings)
        
        self.cropping_time_sec = sum(pt.cropping_time_sec for pt in self.page_timings)
        self.ocr_time_sec = sum(pt.ocr_time_sec for pt in self.page_timings)
    
    def add_page_timing(self, page_timing: PageTiming) -> None:
        """Add timing for a processed page."""
        self.page_timings.append(page_timing)
    
    @property
    def avg_time_per_voter_sec(self) -> float:
        """Calculate average time per voter extraction."""
        if self.total_voters == 0:
            return 0.0
        return self.ocr_time_sec / self.total_voters
    
    @property
    def avg_time_per_voter_ms(self) -> float:
        """Calculate average time per voter in milliseconds."""
        return self.avg_time_per_voter_sec * 1000
    
    @property
    def avg_time_per_page_sec(self) -> float:
        """Calculate average time per page."""
        if self.pages_processed == 0:
            return 0.0
        return (self.cropping_time_sec + self.ocr_time_sec) / self.pages_processed
    
    @property
    def epic_valid_rate(self) -> float:
        """Calculate percentage of valid EPIC numbers."""
        if self.total_voters == 0:
            return 0.0
        return (self.valid_voters / self.total_voters) * 100
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "document_id": self.document_id,
            "pdf_name": self.pdf_name,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "error_message": self.error_message,
            
            "counts": {
                "total_pages": self.total_pages,
                "pages_processed": self.pages_processed,
                "total_crops": self.total_crops,
                "total_voters": self.total_voters,
                "valid_voters": self.valid_voters,
                "epic_valid_rate_percent": round(self.epic_valid_rate, 2),
            },
            
            "timing": {
                "extraction_time_sec": round(self.extraction_time_sec, 4),
                "metadata_time_sec": round(self.metadata_time_sec, 4),
                "cropping_time_sec": round(self.cropping_time_sec, 4),
                "ocr_time_sec": round(self.ocr_time_sec, 4),
                "total_time_sec": round(self.total_time_sec, 4),
                "avg_time_per_page_sec": round(self.avg_time_per_page_sec, 4),
                "avg_time_per_voter_ms": round(self.avg_time_per_voter_ms, 2),
            },
            
            "ai_usage": self.ai_usage.to_dict(),
            
            "page_timings": [pt.to_dict() for pt in self.page_timings],
        }
    
    def summary_str(self) -> str:
        """Generate a human-readable summary string."""
        lines = [
            f"Processing Summary for: {self.pdf_name}",
            f"  Status: {self.status}",
            f"  Pages: {self.pages_processed}/{self.total_pages}",
            f"  Voters: {self.total_voters} (EPIC valid: {self.valid_voters}, {self.epic_valid_rate:.1f}%)",
            f"  Total time: {self.total_time_sec:.2f}s",
            f"  Avg per voter: {self.avg_time_per_voter_ms:.1f}ms",
        ]
        
        if self.ai_usage.calls_count > 0:
            lines.append(
                f"  AI cost: ${self.ai_usage.total_cost_usd:.4f} "
                f"({self.ai_usage.calls_count} calls)"
            )
        
        if self.error_message:
            lines.append(f"  Error: {self.error_message}")
        
        return "\n".join(lines)
