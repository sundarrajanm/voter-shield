"""
ID Field Merger.

Merges ID field strips (created by IdFieldCropper) into vertical batches.
"""
from __future__ import annotations

from .base import ProcessingContext
from .image_merger import ImageMerger

class IdFieldMerger(ImageMerger):
    """
    Merges ID field crops into vertical batches.
    Inherits logical from ImageMerger but uses different input/output directories
    and batch size configuration.
    """
    
    name = "IdFieldMerger"
    
    def __init__(self, context: ProcessingContext):
        super().__init__(context)
        
        # Override directories to use ID crop paths
        self.crops_dir = self.context.id_crops_dir
        self.merged_dir = self.context.id_merged_dir
        
        # Override batch size from ID crop config
        self.max_voters_per_batch = self.config.id_crop.batch_size

    def validate(self) -> bool:
        """Validate prerequisites."""
        if not self.merged_dir:
            self.log_error("ID merged directory not set")
            return False
            
        if not self.crops_dir:
            self.log_error("ID crops directory not set")
            return False
            
        if not self.crops_dir.exists():
            self.log_error(f"ID Crops directory not found: {self.crops_dir}")
            return False
            
        # We still use the voter_end.jpg from base config
        if not self.voter_end_path.exists():
            self.log_error(f"voter_end.jpg not found: {self.voter_end_path}")
            return False
            
        return True
