"""
Repository pattern for document persistence.

Defines abstract interface and concrete implementations for
document storage that can be swapped between JSON files and
database backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Any

from ..models import ProcessedDocument


class DocumentRepository(ABC):
    """
    Abstract repository for document storage.
    
    Implementations can store to JSON files, SQLite, PostgreSQL, etc.
    The interface is designed to be database-agnostic.
    """
    
    @abstractmethod
    def save(self, document: ProcessedDocument) -> str:
        """
        Save a processed document.
        
        Args:
            document: Document to save
        
        Returns:
            Document ID
        """
        pass
    
    @abstractmethod
    def get(self, document_id: str) -> Optional[ProcessedDocument]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: Document identifier
        
        Returns:
            Document if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_by_name(self, pdf_name: str) -> Optional[ProcessedDocument]:
        """
        Retrieve a document by PDF name.
        
        Args:
            pdf_name: PDF file name
        
        Returns:
            Document if found, None otherwise
        """
        pass
    
    @abstractmethod
    def exists(self, pdf_name: str) -> bool:
        """
        Check if a document exists.
        
        Args:
            pdf_name: PDF file name
        
        Returns:
            True if document exists
        """
        pass
    
    @abstractmethod
    def list_all(self) -> List[str]:
        """
        List all document IDs.
        
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def delete(self, document_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            True if deleted, False if not found
        """
        pass


class JSONRepository(DocumentRepository):
    """
    JSON file-based repository implementation.
    
    Stores documents as JSON files, suitable for small to medium
    datasets. Can be replaced with SQL implementation later.
    """
    
    def __init__(self, base_dir: Path):
        """
        Initialize repository.
        
        Args:
            base_dir: Base directory for storage
        """
        from .json_store import JSONStore
        self.store = JSONStore(base_dir)
    
    def save(self, document: ProcessedDocument) -> str:
        """Save document and return its ID."""
        self.store.save_document(document)
        return document.id
    
    def get(self, document_id: str) -> Optional[ProcessedDocument]:
        """Get document by ID (searches by name for JSON store)."""
        # For JSON store, ID is same as name
        return self.get_by_name(document_id)
    
    def get_by_name(self, pdf_name: str) -> Optional[ProcessedDocument]:
        """Get document by PDF name."""
        data = self.store.load_document(pdf_name)
        if not data:
            return None
        
        # Note: Full reconstruction from JSON to ProcessedDocument
        # would require implementing from_dict methods
        # For now, return raw data wrapped in a minimal document
        doc = ProcessedDocument(
            id=data.get("document_id", pdf_name),
            pdf_name=pdf_name,
            status=data.get("status", "unknown"),
        )
        return doc
    
    def exists(self, pdf_name: str) -> bool:
        """Check if document exists."""
        return self.store.document_exists(pdf_name)
    
    def list_all(self) -> List[str]:
        """List all processed documents."""
        return self.store.list_processed()
    
    def delete(self, document_id: str) -> bool:
        """Delete is not implemented for JSON store."""
        # Would need to implement file deletion
        return False


# Future: SQL Repository implementation
# class SQLRepository(DocumentRepository):
#     """
#     SQL database repository implementation.
#     
#     Uses SQLAlchemy for database operations.
#     Supports SQLite, PostgreSQL, MySQL, etc.
#     """
#     
#     def __init__(self, connection_string: str):
#         self.engine = create_engine(connection_string)
#         self.Session = sessionmaker(bind=self.engine)
#     
#     def save(self, document: ProcessedDocument) -> str:
#         # Insert into documents table
#         # Insert metadata into metadata table
#         # Insert voters into voters table
#         # Insert stats into stats table
#         pass
#     
#     def get(self, document_id: str) -> Optional[ProcessedDocument]:
#         # Query and reconstruct document
#         pass
#     
#     # ... etc
