
import json
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("ai_parser")

def parse_ai_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse AI response text to extract structured voter data.
    
    Supports:
    1. TOON (Token Oriented Object Notation) format:
       items[N]{key1,key2,...}:
         val1,val2,...
         val1,val2,...
    
    2. Simple CSV format (when AI returns plain comma-separated lines)
    
    3. Fallback to JSON if above parsing fails.
    
    Args:
        response_text: Raw text response from AI
        
    Returns:
        List of dictionaries containing voter data
    """
    if not response_text:
        return []

    # Try TOON parsing first (since it's the primary requested format now)
    toon_data = parse_toon(response_text)
    if toon_data:
        return toon_data
    
    # Try simple CSV parsing (no header, just data lines)
    csv_data = parse_simple_csv(response_text)
    if csv_data:
        return csv_data

    # Fallback to JSON parsing
    return parse_json(response_text)


def parse_toon(text: str) -> List[Dict[str, Any]]:
    """
    Parse TOON format.
    
    Format example:
    items[2]{serial_no,epic_no,name,relation_type,relation_name,house_no,age,gender}:
      31,NHH3334638,kumar,father,arumugam,40NM,37,Male
      32,NHH3670387,MAHESHWARI,husband,SATHIYA SIVAN,MAHALAKSHMISIZING,25,Female
    """
    try:
        # Regex to find the header pattern: items[N]{columns}:
        header_pattern = r'items\[(\d+)\]\{([^}]+)\}:'
        match = re.search(header_pattern, text)
        
        if not match:
            return []
            
        # count = int(match.group(1))
        columns_str = match.group(2)
        columns = [c.strip() for c in columns_str.split(',')]
        
        # Get the content after the header
        start_pos = match.end()
        content = text[start_pos:].strip()
        
        lines = content.splitlines()
        records = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse CSV-like line
            # Careful with values containing commas. 
            # Simple split for now, assuming AI follows simple CSV (no quotes nesting unless specifically asked)
            # Use specific CSV reader if complex, but simple split is robust for simple AI outputs
            
            # NOTE: If house_no contains comma, this breaks. 
            # We asked AI for "TOON", usually implied simple comma msg.
            values = [v.strip() for v in line.split(',')]
            
            # Align with columns
            if len(values) == len(columns):
                record = dict(zip(columns, values))
                records.append(record)
            else:
                # Handle mismatch (maybe empty trailing commas or merged fields)
                # Try to salvage partial data
                # Extend or truncate
                if len(values) > len(columns):
                     values = values[:len(columns)] # truncate
                elif len(values) < len(columns):
                     values.extend([""] * (len(columns) - len(values)))
                
                record = dict(zip(columns, values))
                records.append(record)
                
        return records
        
    except Exception as e:
        logger.warning(f"TOON parsing failed: {e}")
        return []


def parse_simple_csv(text: str) -> List[Dict[str, Any]]:
    """
    Parse simple CSV format where AI returns plain comma-separated lines.
    
    Format example (no header, columns are assumed in order):
    1,WDZ1264647,பிரியங்கா,father,லட்சுமணன்,283,33,Male
    2,WDZ1264654,பிரியதர்சினி,father,லட்சுமணன்,283,29,Female
    
    Expected column order: serial_no, epic_no, name, relation_type, relation_name, house_no, age, gender
    """
    # Standard columns in expected order
    columns = ["serial_no", "epic_no", "name", "relation_type", "relation_name", "house_no", "age", "gender"]
    
    try:
        lines = text.strip().splitlines()
        records = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Split by comma
            values = [v.strip() for v in line.split(',')]
            
            # Check if this looks like voter data (at least 6 columns, first could be a number)
            if len(values) >= 6:
                # Try to verify it looks like voter data
                # First column should be serial number (numeric or could be missing)
                # Second column should look like EPIC (starts with letters + numbers)
                
                # Handle case where columns might be shifted
                if len(values) >= 8:
                    # Full match - use as-is
                    pass
                elif len(values) < 8:
                    # Pad with empty strings
                    values.extend([""] * (8 - len(values)))
                
                # Truncate if too many values
                values = values[:8]
                
                record = dict(zip(columns, values))
                records.append(record)
        
        return records
        
    except Exception as e:
        logger.warning(f"Simple CSV parsing failed: {e}")
        return []

def parse_json(response_text: str) -> List[Dict[str, Any]]:
    """Legacy JSON parsing."""
    try:
        clean_text = response_text.strip()
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', clean_text, re.IGNORECASE)
        if json_match:
            clean_text = json_match.group(1)
        
        data = json.loads(clean_text)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            for key in ["voters", "data", "records", "result"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        return []
    except Exception:
        return []
