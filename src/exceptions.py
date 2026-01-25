"""
Custom exceptions for the Electoral Roll Processing application.

All application-specific exceptions inherit from ElectorialsError.
"""

from __future__ import annotations

from typing import Optional, Any


class ElectorialsError(Exception):
    """
    Base exception for all application errors.
    
    Attributes:
        message: Human-readable error message
        details: Additional error details (for debugging)
        recoverable: Whether the error can potentially be recovered from
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        recoverable: bool = False
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.recoverable = recoverable
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(ElectorialsError):
    """
    Invalid or missing configuration.
    
    Examples:
        - Missing required environment variable
        - Invalid value for configuration option
    """
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        details = {"config_key": config_key} if config_key else None
        super().__init__(message, details=details, recoverable=False)


class PDFExtractionError(ElectorialsError):
    """
    Failed to extract images from PDF.
    
    Examples:
        - Corrupted PDF file
        - Password-protected PDF
        - Unsupported PDF format
    """
    
    def __init__(
        self,
        message: str,
        pdf_path: Optional[str] = None,
        page_number: Optional[int] = None
    ):
        details = {}
        if pdf_path:
            details["pdf_path"] = pdf_path
        if page_number is not None:
            details["page_number"] = page_number
        super().__init__(message, details=details, recoverable=False)


class MetadataExtractionError(ElectorialsError):
    """
    Failed to extract metadata via AI.
    
    Examples:
        - AI API error
        - Invalid response format
        - Timeout
    """
    
    def __init__(
        self,
        message: str,
        folder_name: Optional[str] = None,
        ai_provider: Optional[str] = None,
        response_text: Optional[str] = None
    ):
        details = {}
        if folder_name:
            details["folder_name"] = folder_name
        if ai_provider:
            details["ai_provider"] = ai_provider
        if response_text:
            # Truncate long responses
            details["response_preview"] = response_text[:500] if len(response_text) > 500 else response_text
        super().__init__(message, details=details, recoverable=True)


class CroppingError(ElectorialsError):
    """
    Failed to crop voter boxes from image.
    
    Examples:
        - Image file unreadable
        - No boxes detected
        - Invalid image dimensions
    """
    
    def __init__(
        self,
        message: str,
        image_path: Optional[str] = None,
        page_id: Optional[str] = None
    ):
        details = {}
        if image_path:
            details["image_path"] = image_path
        if page_id:
            details["page_id"] = page_id
        super().__init__(message, details=details, recoverable=False)


class OCRError(ElectorialsError):
    """
    OCR processing failed.
    
    Examples:
        - Tesseract not installed
        - Invalid image format
        - Language pack missing
    """
    
    def __init__(
        self,
        message: str,
        image_path: Optional[str] = None,
        languages: Optional[str] = None
    ):
        details = {}
        if image_path:
            details["image_path"] = image_path
        if languages:
            details["languages"] = languages
        super().__init__(message, details=details, recoverable=False)


class TesseractNotFoundError(OCRError):
    """Tesseract OCR is not installed or not accessible."""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        message = (
            "Tesseract OCR not found. Please install Tesseract:\n"
            "  - Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  - macOS: brew install tesseract\n"
            "  - Ubuntu: sudo apt install tesseract-ocr"
        )
        super().__init__(message)
        if tesseract_path:
            self.details["tesseract_path_tried"] = tesseract_path


class DataPersistenceError(ElectorialsError):
    """
    Failed to save or load data.
    
    Examples:
        - File write permission denied
        - Invalid JSON format
        - Disk full
    """
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None  # "save" or "load"
    ):
        details = {}
        if file_path:
            details["file_path"] = file_path
        if operation:
            details["operation"] = operation
        super().__init__(message, details=details, recoverable=False)


class ProcessingAbortedError(ElectorialsError):
    """
    Processing was aborted (e.g., by user interrupt).
    """
    
    def __init__(
        self,
        message: str = "Processing aborted by user",
        items_processed: int = 0,
        items_total: int = 0
    ):
        details = {
            "items_processed": items_processed,
            "items_total": items_total
        }
        super().__init__(message, details=details, recoverable=True)


class ValidationError(ElectorialsError):
    """
    Data validation failed.
    
    Examples:
        - Invalid EPIC number format
        - Missing required field
        - Out of range value
    """
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        expected: Optional[str] = None
    ):
        details = {}
        if field_name:
            details["field_name"] = field_name
        if field_value is not None:
            details["field_value"] = str(field_value)[:100]
        if expected:
            details["expected"] = expected
        super().__init__(message, details=details, recoverable=False)


# Aliases for backward compatibility
CropExtractionError = CroppingError
OCRProcessingError = OCRError
ProcessingError = ElectorialsError  # Generic processing error
