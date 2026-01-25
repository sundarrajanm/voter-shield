"""
Document models.

Represents the complete processed document with all associated data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Any, Dict
from datetime import datetime
import uuid

from .voter import Voter
from .metadata import DocumentMetadata
from .processing_stats import ProcessingStats


@dataclass
class PageData:
    """
    Data for a single page of the electoral roll.
    
    Contains all voters extracted from this page.
    """
    
    # Page identification
    page_id: str = ""  # e.g., "page-004"
    page_number: int = 0  # 1-based
    
    # Source files
    image_path: str = ""
    
    # Page header metadata (extracted from top section)
    assembly_constituency_number_and_name: str = ""  # e.g., "116-SULUR"
    street_name_and_number: str = ""  # e.g., "1-Karupparayan Kovil Street Ward No-9"
    part_number: Optional[int] = None  # e.g., 244
    
    # Voters on this page (in order)
    voters: List[Voter] = field(default_factory=list)
    
    # Processing info
    crops_count: int = 0
    processing_time_sec: float = 0.0
    
    @property
    def voters_count(self) -> int:
        return len(self.voters)
    
    @property
    def valid_voters_count(self) -> int:
        return sum(1 for v in self.voters if v.epic_valid)
    
    def to_dict(self) -> dict[str, Any]:
        result = {
            "page_id": self.page_id,
            "page_number": self.page_number,
            "image_path": self.image_path,
            "assembly_constituency_number_and_name": self.assembly_constituency_number_and_name,
            "street_name_and_number": self.street_name_and_number,
            "part_number": self.part_number,
            "crops_count": self.crops_count,
            "voters_count": self.voters_count,
            "valid_voters_count": self.valid_voters_count,
            "processing_time_sec": round(self.processing_time_sec, 4),
            "voters": [v.to_dict() for v in self.voters],
        }
        return result


@dataclass
class ProcessedDocument:
    """
    Complete processed document with all data.
    
    This is the main data structure that combines:
    - Document identification
    - Metadata from AI extraction
    - All voter data from OCR
    - Processing statistics
    
    Designed to be database-ready with clear relationships.
    """
    
    # Primary identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pdf_name: str = ""
    pdf_path: str = ""
    
    # Status
    status: str = "pending"  # pending, processing, completed, failed
    
    # Timestamps
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    processed_at: str = ""
    
    # Metadata (from AI extraction)
    metadata: Optional[DocumentMetadata] = None
    
    # Pages with voter data
    pages: List[PageData] = field(default_factory=list)
    
    # Processing statistics
    stats: ProcessingStats = field(default_factory=ProcessingStats)
    
    @property
    def total_voters(self) -> int:
        """Total number of voters across all pages."""
        return sum(p.voters_count for p in self.pages)
    
    @property
    def valid_voters(self) -> int:
        """Number of voters with valid EPIC."""
        return sum(p.valid_voters_count for p in self.pages)
    
    @property
    def all_voters(self) -> List[Voter]:
        """Get all voters in document order."""
        voters = []
        for page in self.pages:
            voters.extend(page.voters)
        return voters
    
    def add_page(self, page_data: PageData) -> None:
        """Add a processed page with voters."""
        # Update sequence numbers
        current_doc_sequence = self.total_voters
        for i, voter in enumerate(page_data.voters):
            voter.sequence_in_page = i + 1
            voter.sequence_in_document = current_doc_sequence + i + 1
        
        self.pages.append(page_data)
    
    def add_voters(self, voters: List[Voter], header_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add voters from a flat list, grouping by image_file.
        
        This is a convenience method for adding voters when page data
        is not already structured.
        
        Args:
            voters: List of Voter objects
            header_data: Optional dict mapping page_id to header info with keys:
                - assembly_constituency_number_and_name
                - street_name_and_number
                - part_number
        """
        from collections import defaultdict
        
        # Group voters by page (derived from image_file)
        by_page = defaultdict(list)
        for voter in voters:
            # Prioritize explicit page_id
            if voter.page_id:
                page_id = voter.page_id
            # Extract page_id from image_file (e.g., "page-004-001.png" -> "page-004")
            elif voter.image_file:
                parts = voter.image_file.split("-")
                if len(parts) >= 2:
                    page_id = f"{parts[0]}-{parts[1]}"
                else:
                    page_id = voter.image_file
            else:
                page_id = "unknown"
            by_page[page_id].append(voter)

        
        # Sort and add each page
        for page_id in sorted(by_page.keys()):
            page_voters = by_page[page_id]
            # Try to get page number from page_id
            page_num = 0
            try:
                page_num = int(page_id.split("-")[1])
            except (IndexError, ValueError):
                pass
            
            # Get header info for this page if available
            assembly_info = ""
            section_info = ""
            part_num = None
            
            if header_data and page_id in header_data:
                page_header = header_data[page_id]
                # Support both dict and dataclass-like objects
                if hasattr(page_header, 'assembly_constituency_number_and_name'):
                    assembly_info = page_header.assembly_constituency_number_and_name or ""
                    section_info = getattr(page_header, 'street_name_and_number', getattr(page_header, 'section_number_and_name', "")) or ""
                    part_num = page_header.part_number
                elif isinstance(page_header, dict):
                    assembly_info = page_header.get('assembly_constituency_number_and_name', "")
                    section_info = page_header.get('street_name_and_number', page_header.get('section_number_and_name', ""))
                    part_num = page_header.get('part_number')
            
            # Get timing from stats
            proc_time = 0.0
            if self.stats and self.stats.page_timings:
                proc_time = sum(pt.total_time_sec for pt in self.stats.page_timings if pt.page_id == page_id)

            page = PageData(
                page_id=page_id,
                page_number=page_num,
                voters=page_voters,
                crops_count=len(page_voters),
                assembly_constituency_number_and_name=assembly_info,
                street_name_and_number=section_info,
                part_number=part_num,
                processing_time_sec=proc_time,
            )
            self.add_page(page)

    
    def complete(self) -> None:
        """Mark document as completed."""
        self.status = "completed"
        self.processed_at = datetime.utcnow().isoformat() + "Z"
        self.stats.complete()
    
    def fail(self, error: str) -> None:
        """Mark document as failed."""
        self.status = "failed"
        self.processed_at = datetime.utcnow().isoformat() + "Z"
        self.stats.fail(error)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        This is the unified output format that can be:
        1. Saved as JSON
        2. Mapped to SQL tables
        """
        return {
            "document": {
                "id": self.id,
                "pdf_name": self.pdf_name,
                "pdf_path": self.pdf_path,
                "status": self.status,
                "created_at": self.created_at,
                "processed_at": self.processed_at,
                "total_pages": len(self.pages),
                "total_voters": self.total_voters,
                "valid_voters": self.valid_voters,
            },
            
            "metadata": self.metadata.to_dict() if self.metadata else None,
            
            "pages": [p.to_dict() for p in self.pages],
            
            "voters": [v.to_dict() for v in self.all_voters],
            
            "stats": self.stats.to_dict(),
        }
    
    def _get_metadata_dict(self) -> Optional[dict[str, Any]]:
        """
        Get metadata dictionary with synchronized calculated fields.
        """
        if not self.metadata:
            return None
            
        meta_dict = self.metadata.to_dict()
        
        # Ensure total_voters_extracted matches the actual extracted count
        # This overrides any None value or stale count in the metadata
        if self.pages:  # Only update if we have pages (meaning we processed voters)
            meta_dict["total_voters_extracted"] = self.total_voters
            
        return meta_dict
    
    def to_combined_json(self) -> dict[str, Any]:
        """
        Generate combined JSON matching original output format.
        
        This maintains backward compatibility with the existing
        output structure while adding new tracking fields.
        """
        # Combine voters across all pages
        all_records = []
        for page in self.pages:
            for voter in page.voters:
                voter_dict = voter.to_dict()
                voter_dict["assembly_constituency_number_and_name"] = page.assembly_constituency_number_and_name
                voter_dict["street_name_and_number"] = page.street_name_and_number
                voter_dict["part_number"] = page.part_number
                all_records.append(voter_dict)
        
        return {
            "folder": self.pdf_name,
            "document_id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "processed_at": self.processed_at,
            
            # Metadata if available
            "metadata": self._get_metadata_dict(),
            
            # Summary counts
            "pages_count": len(self.pages),
            "images_count": sum(p.crops_count for p in self.pages),
            "total_voters": self.total_voters,
            "valid_voters": self.valid_voters,
            
            # Timing
            "timing": {
                "total_time_sec": round(self.stats.total_time_sec, 4),
                "avg_time_per_voter_ms": round(self.stats.avg_time_per_voter_ms, 2),
            },
            
            # AI Metadata (if applicable)
            "ai_metadata_voter": {
                "provider": self.stats.ai_usage.provider or "Groq", 
                "model": self.stats.ai_usage.model,
                "input_tokens": self.stats.ai_usage.total_input_tokens,
                "output_tokens": self.stats.ai_usage.total_output_tokens,
                "cost_usd": self.stats.ai_usage.total_cost_usd, 
            } if self.stats.ai_usage.calls_count > 0 else None,


            
            # Page summaries
            "pages": [
                {
                    "page": p.page_id,
                    "page_number": p.page_number,
                    "images_processed": p.crops_count,
                    "voters_count": p.voters_count,
                    "processing_time_sec": round(p.processing_time_sec, 4),
                    "assembly_constituency_number_and_name": p.assembly_constituency_number_and_name,
                    "street_name_and_number": p.street_name_and_number,
                    "part_number": p.part_number,
                }
                for p in self.pages
            ],
            
            # All voter records
            "records": all_records,
        }


@dataclass
class ProcessingResult:
    """
    Result of a processing operation.
    
    Used to communicate success/failure between processors.
    """
    success: bool = True
    message: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    data: Any = None
    timing_sec: float = 0.0
    
    @classmethod
    def ok(cls, data: Any = None, message: str = "", timing_sec: float = 0.0) -> "ProcessingResult":
        """Create successful result."""
        return cls(success=True, data=data, message=message, timing_sec=timing_sec)
    
    @classmethod
    def error(
        cls,
        error: str,
        error_type: str = "UnknownError",
        timing_sec: float = 0.0
    ) -> "ProcessingResult":
        """Create error result."""
        return cls(
            success=False,
            error=error,
            error_type=error_type,
            timing_sec=timing_sec
        )
