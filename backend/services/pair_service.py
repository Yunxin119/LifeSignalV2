import random
import string
import logging
import time
from datetime import datetime, timedelta
from database import get_collection

logger = logging.getLogger(__name__)

class PairingService:
    """Service for handling device pairing"""
    
    # Collection name in MongoDB
    PAIRING_COLLECTION = "pairing_codes"
    USER_COLLECTION = "users"
    
    # Pairing code expiration time
    EXPIRATION_MINUTES = 10
    
    @staticmethod
    def generate_pairing_code(user_id):
        """Generate a 6-digit pairing code for a user
        
        Args:
            user_id (str): The ID of the user to generate a code for
            
        Returns:
            str: The generated pairing code
        """
        # Validate user_id exists
        user_collection = get_collection(PairingService.USER_COLLECTION)
        user = user_collection.find_one({"_id": user_id})
        
        if not user:
            logger.warning(f"Attempt to generate pairing code for non-existent user: {user_id}")
            return None
        
        # Delete any existing pairing codes for this user
        pairing_collection = get_collection(PairingService.PAIRING_COLLECTION)
        pairing_collection.delete_many({"user_id": user_id})
        
        # Generate a new 6-digit code
        pairing_code = ''.join(random.choices(string.digits, k=6))
        
        # Save the pairing code to the database with expiration time
        expiration_time = datetime.now() + timedelta(minutes=PairingService.EXPIRATION_MINUTES)
        
        pairing_data = {
            "user_id": user_id,
            "code": pairing_code,
            "expiration_time": expiration_time,
            "is_paired": False,
            "created_at": datetime.now()
        }
        
        pairing_collection.insert_one(pairing_data)
        
        logger.info(f"Generated pairing code for user {user_id}")
        return pairing_code
    
    @staticmethod
    def validate_pairing_code(pairing_code, device_type="apple_watch"):
        """Validate a pairing code and pair a device
        
        Args:
            pairing_code (str): The pairing code to validate
            device_type (str): The type of device being paired
            
        Returns:
            dict: Information about the paired user, or None if invalid
        """
        pairing_collection = get_collection(PairingService.PAIRING_COLLECTION)
        user_collection = get_collection(PairingService.USER_COLLECTION)
        
        # Find the pairing code in the database
        pairing_data = pairing_collection.find_one({
            "code": pairing_code,
            "is_paired": False
        })
        
        if not pairing_data:
            logger.warning(f"Invalid or already used pairing code: {pairing_code}")
            return None
        
        # Check if the code has expired
        if datetime.now() > pairing_data["expiration_time"]:
            logger.warning(f"Expired pairing code: {pairing_code}")
            return None
        
        # Get the user associated with this code
        user_id = pairing_data["user_id"]
        user = user_collection.find_one({"_id": user_id})
        
        if not user:
            logger.warning(f"User associated with pairing code no longer exists: {user_id}")
            return None
        
        # Mark the code as paired
        pairing_collection.update_one(
            {"_id": pairing_data["_id"]},
            {"$set": {"is_paired": True, "paired_at": datetime.now(), "device_type": device_type}}
        )
        
        # Update the user record to indicate they have a paired device
        device_field = "is_watch_connected" if device_type == "apple_watch" else "is_mobile_connected"
        user_collection.update_one(
            {"_id": user_id},
            {"$set": {device_field: True, "last_paired_at": datetime.now()}}
        )
        
        logger.info(f"Successfully paired device {device_type} for user {user_id}")
        
        # Return user information (excluding sensitive fields)
        user_info = {
            "_id": str(user["_id"]),
            "username": user["username"],
            "email": user["email"]
        }
        
        if "full_name" in user:
            user_info["full_name"] = user["full_name"]
            
        return user_info
    
    @staticmethod
    def unpair_device(user_id, device_type="apple_watch"):
        """Unpair a device from a user
        
        Args:
            user_id (str): The ID of the user to unpair
            device_type (str): The type of device to unpair
            
        Returns:
            bool: True if successful, False otherwise
        """
        user_collection = get_collection(PairingService.USER_COLLECTION)
        
        # Update the user record to indicate they no longer have a paired device
        device_field = "is_watch_connected" if device_type == "apple_watch" else "is_mobile_connected"
        
        result = user_collection.update_one(
            {"_id": user_id},
            {"$set": {device_field: False}}
        )
        
        if result.modified_count > 0:
            logger.info(f"Successfully unpaired device {device_type} for user {user_id}")
            return True
        else:
            logger.warning(f"Failed to unpair device {device_type} for user {user_id}")
            return False
    
    @staticmethod
    def check_pairing_status(user_id, device_type="apple_watch"):
        """Check if a user has a paired device
        
        Args:
            user_id (str): The ID of the user to check
            device_type (str): The type of device to check
            
        Returns:
            bool: True if paired, False otherwise
        """
        user_collection = get_collection(PairingService.USER_COLLECTION)
        
        # Get the user record
        user = user_collection.find_one({"_id": user_id})
        
        if not user:
            logger.warning(f"User not found when checking pairing status: {user_id}")
            return False
        
        # Check if the specified device is paired
        device_field = "is_watch_connected" if device_type == "apple_watch" else "is_mobile_connected"
        
        return user.get(device_field, False)
