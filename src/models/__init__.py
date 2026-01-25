"""
Data models for the Electoral Roll Processing application.

These models represent the core data structures and are designed
to be easily serializable to JSON and mappable to SQL database tables.
"""

from .voter import Voter, VoterRecord
from .metadata import DocumentMetadata, ConstituencyDetails, AdministrativeAddress
from .processing_stats import ProcessingStats, PageTiming, AIUsage
from .document import ProcessedDocument, PageData, ProcessingResult

__all__ = [
    # Voter models
    "Voter",
    "VoterRecord",
    
    # Metadata models
    "DocumentMetadata",
    "ConstituencyDetails",
    "AdministrativeAddress",
    
    # Processing stats
    "ProcessingStats",
    "PageTiming",
    "AIUsage",
    
    # Document models
    "ProcessedDocument",
    "PageData",
    "ProcessingResult",
]
