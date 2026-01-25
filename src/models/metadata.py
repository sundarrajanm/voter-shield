"""
Document metadata models.

Represents metadata extracted from electoral roll front and back pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Any


@dataclass
class Section:
    """A section within a polling area."""
    street_number: str = ""
    street_name: str = ""


@dataclass
class ElectorSummary:
    """Summary of electors by gender."""
    male: Optional[int] = None
    female: Optional[int] = None
    third_gender: Optional[int] = None
    total: Optional[int] = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SerialRange:
    """Range of serial numbers."""
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class ConstituencyDetails:
    """Assembly and Parliamentary constituency information."""
    assembly_constituency_number: Optional[int] = None
    assembly_constituency_name: str = ""
    assembly_reservation_status: Optional[str] = None
    parliamentary_constituency_number: Optional[int] = None
    parliamentary_constituency_name: str = ""
    parliamentary_reservation_status: Optional[str] = None
    part_number: Optional[int] = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdministrativeAddress:
    """Administrative address information."""
    town_or_village: str = ""
    ward_number: str = ""
    post_office: str = ""
    police_station: str = ""
    taluk_or_block: str = ""
    subdivision: str = ""
    district: str = ""
    pin_code: str = ""
    panchayat_name: str = ""
    main_town_or_village: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)





@dataclass
class PollingDetails:
    """Polling station and section details."""
    sections: List[Section] = field(default_factory=list)
    polling_station_number: Optional[int] = None
    polling_station_name: str = ""
    polling_station_address: str = ""
    polling_station_type: Optional[str] = None
    auxiliary_polling_station_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sections"] = [asdict(s) for s in self.sections]
        return data


@dataclass
class DetailedElectorSummary:
    """Detailed summary of electors with mother roll and modifications."""
    serial_number_range: SerialRange = field(default_factory=SerialRange)
    net_total: ElectorSummary = field(default_factory=ElectorSummary)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "serial_number_range": asdict(self.serial_number_range),
            "net_total": self.net_total.to_dict(),
        }


@dataclass
class AuthorityVerification:
    """Authority/signature verification details."""
    designation: str = ""
    signature_present: bool = False


@dataclass
class DocumentMetadata:
    """
    Complete metadata for an electoral roll document.
    
    This combines all information extracted from the front page
    (administrative info) and back page (elector summary).
    """
    
    # Document identification
    document_id: str = ""  # Foreign key to parent document
    
    # Languages
    language_detected: List[str] = field(default_factory=list)
    
    # Basic document info
    state: str = ""
    electoral_roll_year: Optional[int] = None
    revision_type: str = ""
    qualifying_date: str = ""
    publication_date: str = ""
    roll_type: str = ""
    roll_identification: str = ""
    total_pages: Optional[int] = None
    total_voters_extracted: Optional[int] = None
    page_number_current: Optional[int] = None
    output_identifier: Optional[str] = None
    
    # Nested structures
    constituency_details: ConstituencyDetails = field(default_factory=ConstituencyDetails)
    administrative_address: AdministrativeAddress = field(default_factory=AdministrativeAddress)
    polling_details: PollingDetails = field(default_factory=PollingDetails)
    detailed_elector_summary: DetailedElectorSummary = field(default_factory=DetailedElectorSummary)
    authority_verification: AuthorityVerification = field(default_factory=AuthorityVerification)
    
    # AI extraction tracking
    ai_provider: str = ""
    ai_model: str = ""
    ai_input_tokens: int = 0
    ai_output_tokens: int = 0
    ai_cost_usd: Optional[float] = None
    ai_extraction_time_sec: float = 0.0
    
    # Raw response (for debugging)
    raw_response: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for JSON serialization (NEW FLAT STRUCTURE).
        
        Returns only the 5 essential fields:
        - language_detected
        - state
        - electoral_roll_year
        - pin_code (from administrative_address)
        - total (from detailed_elector_summary.net_total.total)
        """
        return {
            "language_detected": self.language_detected,
            "state": self.state,
            "electoral_roll_year": self.electoral_roll_year,
            "pin_code": self.administrative_address.pin_code,
            "total": self.detailed_elector_summary.net_total.total,
        }
    
    def to_dict_legacy(self) -> dict[str, Any]:
        """
        Convert to dictionary with full nested structure (LEGACY).
        
        Use this method if you need the old nested structure for backward compatibility.
        """
        return {
            "document_id": self.document_id,
            "language_detected": self.language_detected,
            "state": self.state,
            "electoral_roll_year": self.electoral_roll_year,
            "revision_type": self.revision_type,
            "qualifying_date": self.qualifying_date,
            "publication_date": self.publication_date,
            "roll_type": self.roll_type,
            "roll_identification": self.roll_identification,
            "total_pages": self.total_pages,
            "total_voters_extracted": self.total_voters_extracted,
            "page_number_current": self.page_number_current,
            "output_identifier": self.output_identifier,
            "constituency_details": self.constituency_details.to_dict(),
            "administrative_address": self.administrative_address.to_dict(),
            "polling_details": self.polling_details.to_dict(),
            "detailed_elector_summary": self.detailed_elector_summary.to_dict(),
            "authority_verification": asdict(self.authority_verification),
            "ai_metadata": {
                "provider": self.ai_provider,
                "model": self.ai_model,
                "input_tokens": self.ai_input_tokens,
                "output_tokens": self.ai_output_tokens,
                "cost_usd": self.ai_cost_usd,
                "extraction_time_sec": self.ai_extraction_time_sec,
            }
        }
    
    @classmethod
    def from_ai_response(
        cls,
        response_data: dict[str, Any],
        ai_provider: str = "",
        ai_model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: Optional[float] = None,
        extraction_time_sec: float = 0.0
    ) -> "DocumentMetadata":
        """
        Create DocumentMetadata from AI response.
        
        Maps the AI response structure to our data model.
        Handles THREE formats with backward compatibility:
        1. NEW FLAT: {language_detected, state, electoral_roll_year, pin_code, total}
        2. OLD NESTED: {document_metadata: {...}, administrative_address: {...}, ...}
        3. SAVED FILES: Top-level fields without document_metadata wrapper
        """
        metadata = cls()
        
        # Detect structure type
        is_flat_structure = (
            "language_detected" in response_data and
            "total" in response_data and
            "pin_code" in response_data and
            "document_metadata" not in response_data and
            "administrative_address" not in response_data
        )
        
        if is_flat_structure:
            # NEW FLAT STRUCTURE: Only 5 fields at top-level
            metadata.language_detected = response_data.get("language_detected", [])
            metadata.state = response_data.get("state", "")
            metadata.electoral_roll_year = response_data.get("electoral_roll_year")
            
            # Map flat pin_code to administrative_address
            pin_code = response_data.get("pin_code", "")
            metadata.administrative_address = AdministrativeAddress(pin_code=pin_code)
            
            # Map flat total to detailed_elector_summary.net_total.total
            total = response_data.get("total")
            metadata.detailed_elector_summary = DetailedElectorSummary(
                net_total=ElectorSummary(total=total)
            )
            
            # AI tracking
            metadata.ai_provider = ai_provider
            metadata.ai_model = ai_model
            metadata.ai_input_tokens = input_tokens
            metadata.ai_output_tokens = output_tokens
            metadata.ai_cost_usd = cost_usd
            metadata.ai_extraction_time_sec = extraction_time_sec
            
            return metadata
        
        # Document metadata - check nested or top-level
        doc_meta = response_data.get("document_metadata", {})
        # If document_metadata is empty, try top-level fields (for saved metadata files)
        if not doc_meta:
            doc_meta = response_data
        
        metadata.language_detected = doc_meta.get("language_detected", [])
        metadata.state = doc_meta.get("state", "")
        metadata.electoral_roll_year = doc_meta.get("electoral_roll_year")
        metadata.revision_type = doc_meta.get("revision_type", "")
        metadata.qualifying_date = doc_meta.get("qualifying_date", "")
        metadata.publication_date = doc_meta.get("publication_date", "")
        metadata.roll_type = doc_meta.get("roll_type", "")
        metadata.roll_identification = doc_meta.get("roll_identification", "")
        metadata.total_pages = doc_meta.get("total_pages")
        metadata.total_voters_extracted = doc_meta.get("total_voters_extracted")
        metadata.page_number_current = doc_meta.get("page_number_current")
        metadata.output_identifier = response_data.get("output_identifier")
        
        # Constituency details
        const_data = response_data.get("constituency_details") or {}
        metadata.constituency_details = ConstituencyDetails(
            assembly_constituency_number=const_data.get("assembly_constituency_number"),
            assembly_constituency_name=const_data.get("assembly_constituency_name", ""),
            assembly_reservation_status=const_data.get("assembly_reservation_status"),
            parliamentary_constituency_number=const_data.get("parliamentary_constituency_number"),
            parliamentary_constituency_name=const_data.get("parliamentary_constituency_name", ""),
            parliamentary_reservation_status=const_data.get("parliamentary_reservation_status"),
            part_number=const_data.get("part_number"),
        )
        
        # Administrative address (with fallback for flat structure)
        addr_data = response_data.get("administrative_address") or {}
        # Fallback: Check if pin_code exists at top-level (flat structure from DB)
        if not addr_data and "pin_code" in response_data:
            addr_data = {"pin_code": response_data.get("pin_code", "")}
        
        metadata.administrative_address = AdministrativeAddress(
            town_or_village=addr_data.get("town_or_village", ""),
            ward_number=addr_data.get("ward_number", ""),
            post_office=addr_data.get("post_office", ""),
            police_station=addr_data.get("police_station", ""),
            taluk_or_block=addr_data.get("taluk_or_block", ""),
            subdivision=addr_data.get("subdivision", ""),
            district=addr_data.get("district", ""),
            pin_code=addr_data.get("pin_code", ""),
            panchayat_name=addr_data.get("panchayat_name", ""),
            main_town_or_village=addr_data.get("main_town_or_village", ""),
        )
        
        # Polling details - check both field names
        poll_data = response_data.get("part_and_polling_details") or response_data.get("polling_details") or {}
        
        sections = [
            Section(
                street_number=str(s.get("street_number", s.get("section_number", ""))),
                street_name=s.get("street_name", s.get("section_name", ""))
            )
            for s in poll_data.get("sections") or [] if s
        ]
        metadata.polling_details = PollingDetails(
            sections=sections,
            polling_station_number=poll_data.get("polling_station_number"),
            polling_station_name=poll_data.get("polling_station_name", ""),
            polling_station_address=poll_data.get("polling_station_address", ""),
            polling_station_type=poll_data.get("polling_station_type"),
            auxiliary_polling_station_count=poll_data.get("auxiliary_polling_station_count", 0) or 0,
        )
        
        # Detailed elector summary (with fallback for flat structure)
        summary_data = response_data.get("detailed_elector_summary") or {}
        range_data = summary_data.get("serial_number_range") or {}
        
        def make_elector_summary(data: Optional[dict]) -> ElectorSummary:
            if not data:
                # Fallback: Check for flat 'total' at top level
                if "total" in response_data and not summary_data:
                    return ElectorSummary(total=response_data.get("total"))
                return ElectorSummary()
            return ElectorSummary(
                male=data.get("male"),
                female=data.get("female"),
                third_gender=data.get("third_gender"),
                total=data.get("total"),
            )
        
        metadata.detailed_elector_summary = DetailedElectorSummary(
            serial_number_range=SerialRange(
                start=range_data.get("start"),
                end=range_data.get("end"),
            ),
            net_total=make_elector_summary(summary_data.get("net_total")),
        )
        
        # Authority verification
        auth_data = response_data.get("authority_verification", {})
        metadata.authority_verification = AuthorityVerification(
            designation=auth_data.get("designation", ""),
            signature_present=bool(auth_data.get("signature_present", False)),
        )
        
        # AI tracking
        metadata.ai_provider = ai_provider
        metadata.ai_model = ai_model
        metadata.ai_input_tokens = input_tokens
        metadata.ai_output_tokens = output_tokens
        metadata.ai_cost_usd = cost_usd
        metadata.ai_extraction_time_sec = extraction_time_sec
        
        return metadata
