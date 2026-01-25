"""
Data persistence layer.

Provides abstract repository pattern and concrete implementations
for storing processed document data.
"""

from .json_store import JSONStore
from .repository import DocumentRepository

__all__ = [
    "JSONStore",
    "DocumentRepository",
]
