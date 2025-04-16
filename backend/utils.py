import json
from datetime import datetime
from bson import ObjectId
import logging
from config import DEBUG

# Configure logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MongoDB documents"""
    
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

def json_response(data):
    """
    Convert MongoDB documents to JSON-serializable format
    
    Args:
        data: Data to convert
        
    Returns:
        dict: JSON-serializable data
    """
    return json.loads(JSONEncoder().encode(data))

def sanitize_document(doc):
    """
    Sanitize a MongoDB document for safe JSON serialization
    
    Args:
        doc (dict): MongoDB document
        
    Returns:
        dict: Sanitized document
    """
    if not doc:
        return doc
        
    # Create a copy to avoid modifying the original
    sanitized = {}
    
    for key, value in doc.items():
        # Convert ObjectId to string
        if isinstance(value, ObjectId):
            sanitized[key] = str(value)
        # Convert datetime to ISO format
        elif isinstance(value, datetime):
            sanitized[key] = value.isoformat()
        # Recursively sanitize nested dictionaries
        elif isinstance(value, dict):
            sanitized[key] = sanitize_document(value)
        # Recursively sanitize items in lists
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_document(item) if isinstance(item, dict) 
                else str(item) if isinstance(item, ObjectId)
                else item.isoformat() if isinstance(item, datetime)
                else item
                for item in value
            ]
        # Keep other values as is
        else:
            sanitized[key] = value
    
    return sanitized

def parse_int(value, default=None):
    """
    Safely parse a string to int
    
    Args:
        value (str): String to parse
        default (int, optional): Default value if parsing fails
        
    Returns:
        int: Parsed integer or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def parse_float(value, default=None):
    """
    Safely parse a string to float
    
    Args:
        value (str): String to parse
        default (float, optional): Default value if parsing fails
        
    Returns:
        float: Parsed float or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default 