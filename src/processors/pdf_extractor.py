"""
PDF Extractor processor.

Extracts page images from PDF files using PyMuPDF.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Any

import fitz  # PyMuPDF

from .base import BaseProcessor, ProcessingContext
from ..exceptions import PDFExtractionError


@dataclass(frozen=True)
class ExtractedPage:
    """Information about an extracted page."""
    page_number: int
    path: str
    width: int
    height: int


@dataclass
class ExtractionResult:
    """Result of PDF extraction."""
    pdf_name: str
    pdf_path: str
    total_pages: int
    extracted_pages: List[ExtractedPage]
    metadata: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "pdf_name": self.pdf_name,
            "pdf_path": self.pdf_path,
            "total_pages": self.total_pages,
            "extracted_pages": [asdict(p) for p in self.extracted_pages],
            "metadata": self.metadata,
        }


class PDFExtractor(BaseProcessor):
    """
    Extract page images from PDF files.
    
    Uses PyMuPDF to render each page as a high-resolution PNG image.
    Creates manifest.json with extraction metadata.
    """
    
    name = "PDFExtractor"
    
    def __init__(
        self,
        context: ProcessingContext,
        dpi: int = 200,
        extract_text: bool = False,
    ):
        """
        Initialize extractor.
        
        Args:
            context: Processing context
            dpi: DPI for page rendering (default: 200)
            extract_text: Whether to also extract text (default: False)
        """
        super().__init__(context)
        self.dpi = dpi
        self.extract_text = extract_text
        self.result: Optional[ExtractionResult] = None
    
    def validate(self) -> bool:
        """Check PDF path exists."""
        if not self.context.pdf_path:
            self.log_error("No PDF path specified")
            return False
        
        if not self.context.pdf_path.exists():
            self.log_error(f"PDF not found: {self.context.pdf_path}")
            return False
        
        return True
    
    def process(self) -> bool:
        """
        Extract images from PDF.
        
        Returns:
            True if extraction succeeded
        """
        pdf_path = self.context.pdf_path
        images_dir = self.context.images_dir
        
        self.log_info(f"Opening PDF: {pdf_path.name}")
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise PDFExtractionError(f"Failed to open PDF: {e}", str(pdf_path))
        
        try:
            total_pages = doc.page_count
            self.log_info(f"PDF has {total_pages} pages")
            
            # Update context
            self.context.total_pages = total_pages
            
            # Extract PDF metadata
            pdf_metadata = {k: v for k, v in (doc.metadata or {}).items() if v}
            
            # Render pages
            extracted_pages = self._render_pages(doc, images_dir)
            
            # Optionally extract text
            text_info = None
            if self.extract_text:
                text_info = self._extract_text(doc)
            
            # Create manifest
            manifest = self._create_manifest(
                pdf_path=pdf_path,
                total_pages=total_pages,
                pdf_metadata=pdf_metadata,
                extracted_pages=extracted_pages,
                text_info=text_info,
            )
            
            # Save manifest
            manifest_path = self.context.extracted_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            
            # Store result
            self.result = ExtractionResult(
                pdf_name=self.context.pdf_name,
                pdf_path=str(pdf_path),
                total_pages=total_pages,
                extracted_pages=extracted_pages,
                metadata=pdf_metadata,
            )
            
            self.log_info(
                f"Extracted {len(extracted_pages)} pages",
                output_dir=str(images_dir)
            )
            
            return True
            
        finally:
            doc.close()
    
    def _render_pages(
        self,
        doc: fitz.Document,
        out_dir: Path
    ) -> List[ExtractedPage]:
        """
        Render PDF pages as PNG images.
        
        Args:
            doc: PyMuPDF document
            out_dir: Output directory
        
        Returns:
            List of extracted page info
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        extracted: List[ExtractedPage] = []
        
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            
            output_path = out_dir / f"page-{page_index + 1:03d}.png"
            pix.save(str(output_path))
            
            extracted.append(ExtractedPage(
                page_number=page_index + 1,
                path=str(output_path.as_posix()),
                width=pix.width,
                height=pix.height,
            ))
            
            self.log_debug(
                f"Rendered page {page_index + 1}",
                size=f"{pix.width}x{pix.height}"
            )
        
        return extracted
    
    def _extract_text(self, doc: fitz.Document) -> dict[str, Any]:
        """
        Extract text from PDF pages.
        
        Args:
            doc: PyMuPDF document
        
        Returns:
            Text extraction info
        """
        text_dir = self.context.extracted_dir / "text"
        text_dir.mkdir(exist_ok=True)
        
        combined_lines: List[str] = []
        per_page: List[dict[str, Any]] = []
        
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            
            # Save individual page text
            page_path = text_dir / f"page-{page_index + 1:03d}.txt"
            page_path.write_text(text, encoding="utf-8")
            
            combined_lines.append(text)
            per_page.append({
                "page": page_index + 1,
                "path": str(page_path.as_posix()),
                "chars": len(text),
            })
        
        # Save combined text
        combined_path = text_dir / "text.txt"
        combined_path.write_text("\n".join(combined_lines), encoding="utf-8")
        
        return {
            "combined": str(combined_path.as_posix()),
            "pages": per_page,
        }
    
    def _create_manifest(
        self,
        pdf_path: Path,
        total_pages: int,
        pdf_metadata: dict[str, Any],
        extracted_pages: List[ExtractedPage],
        text_info: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create extraction manifest."""
        manifest = {
            "input_pdf": str(pdf_path.as_posix()),
            "pages": total_pages,
            "metadata": pdf_metadata,
            "dpi": self.dpi,
            "rendered_pages": [p.path for p in extracted_pages],
        }
        
        if text_info:
            manifest["text"] = text_info
        
        return manifest


def extract_pdf(
    pdf_path: Path,
    output_dir: Optional[Path] = None,
    dpi: int = 200,
    extract_text: bool = False,
) -> ExtractionResult:
    """
    Convenience function to extract images from a PDF.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory (default: extracted/<pdf_name>)
        dpi: DPI for rendering
        extract_text: Whether to extract text
    
    Returns:
        Extraction result
    """
    from ..config import Config
    
    config = Config()
    
    if output_dir:
        config.extracted_dir = output_dir.parent
    
    context = ProcessingContext(config=config)
    context.setup_paths_from_pdf(pdf_path)
    
    extractor = PDFExtractor(context, dpi=dpi, extract_text=extract_text)
    
    if not extractor.run():
        raise PDFExtractionError("Extraction failed", str(pdf_path))
    
    return extractor.result
