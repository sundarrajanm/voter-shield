"""
Voter data models.

Represents individual voter records extracted from electoral roll pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import re


@dataclass
class Voter:
    """
    Core voter information extracted from electoral roll.
    
    This is the primary data model for voter records, designed to be
    database-ready with explicit foreign key references.
    """
    
    # Unique identifiers
    serial_no: str = ""
    epic_no: str = ""
    epic_valid: bool = False
    
    # Personal information
    name: str = ""
    relation_type: str = ""  # father, mother, husband
    relation_name: str = ""
    house_no: str = ""
    age: str = ""
    gender: str = ""  # Male, Female, Other
    
    # Extraction metadata (for tracking and debugging)
    image_file: str = ""
    page_id: str = ""
    sequence_in_page: int = 0  # 1-based order within page
    sequence_in_document: int = 0  # 1-based order within entire document
    
    # Processing metrics
    processing_time_ms: float = 0.0
    extraction_confidence: float = 0.0
    processing_method: str = ""
    
    # Deletion status (empty string = not deleted, "true" = deleted)
    deleted: str = ""
    
    def __post_init__(self):

        """Validate and clean data after initialization."""
        # Clean whitespace
        self.serial_no = self.serial_no.strip()
        self.epic_no = self.epic_no.strip().upper()
        self.name = self.name.strip()
        self.relation_type = self.relation_type.strip().lower()
        self.relation_name = self.relation_name.strip()
        self.house_no = self.house_no.strip()
        self.age = self.age.strip()
        self.gender = self.gender.strip()
        
        # Validate EPIC format if present
        if self.epic_no and not self.epic_valid:
            self.epic_valid = self.validate_epic(self.epic_no)
    
    @staticmethod
    def validate_epic(epic: str) -> bool:
        """
        Validate EPIC number format.
        
        Indian EPIC format: 3 letters followed by 7 digits (e.g., ABC1234567)
        """
        if not epic:
            return False
        # Standard format: XXX0000000 (3 letters + 7 digits)
        return bool(re.fullmatch(r"[A-Z]{3}\d{7}", epic.upper()))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Voter":
        """Create Voter from dictionary."""
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })
    
    @property
    def is_complete(self) -> bool:
        """Check if all essential fields are populated."""
        return bool(
            self.epic_no and
            self.epic_valid and
            self.name and
            (self.age or self.gender)
        )
    
    @property
    def completeness_score(self) -> float:
        """
        Calculate completeness score (0-1).
        
        Higher scores indicate more complete records.
        """
        fields = [
            (self.serial_no, 0.05),
            (self.epic_no and self.epic_valid, 0.25),
            (self.name, 0.20),
            (self.relation_type and self.relation_name, 0.15),
            (self.house_no, 0.10),
            (self.age, 0.15),
            (self.gender, 0.10),
        ]
        return sum(weight for value, weight in fields if value)


@dataclass
class VoterRecord:
    """
    Extended voter record with document context.
    
    Used for full database representation including foreign keys.
    """
    
    # Primary key
    id: str = ""  # UUID or auto-generated
    
    # Foreign keys
    document_id: str = ""
    page_id: str = ""
    
    # Voter data
    voter: Voter = field(default_factory=Voter)
    
    # Audit fields
    created_at: str = ""  # ISO timestamp
    updated_at: str = ""  # ISO timestamp
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for database insertion."""
        data = {
            "id": self.id,
            "document_id": self.document_id,
            "page_id": self.page_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        data.update(self.voter.to_dict())
        return data
