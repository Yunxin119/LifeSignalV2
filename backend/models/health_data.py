from datetime import datetime
from bson import ObjectId
from database import get_collection

class HealthData:
    """Model for handling health data operations in MongoDB"""
    
    COLLECTION_NAME = 'health_data'
    
    @classmethod
    def get_collection(cls):
        """Get the health data collection"""
        return get_collection(cls.COLLECTION_NAME)
    
    @classmethod
    def create(cls, user_id, heart_rate, blood_oxygen, additional_metrics=None):
        """
        Create a new health data entry
        
        Args:
            user_id (str): ID of the user this data belongs to
            heart_rate (float): Heart rate measurement
            blood_oxygen (float): Blood oxygen level measurement
            additional_metrics (dict, optional): Additional health metrics
            
        Returns:
            str: ID of the created document
        """
        collection = cls.get_collection()
        
        # Create base document
        document = {
            'user_id': user_id,
            'heart_rate': heart_rate,
            'blood_oxygen': blood_oxygen,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        # Add any additional metrics
        if additional_metrics and isinstance(additional_metrics, dict):
            document.update(additional_metrics)
        
        # Insert document and return ID
        result = collection.insert_one(document)
        return str(result.inserted_id)
    
    @classmethod
    def get_by_id(cls, document_id):
        """Get health data by ID"""
        collection = cls.get_collection()
        document = collection.find_one({'_id': ObjectId(document_id)})
        return document
    
    @classmethod
    def get_by_user_id(cls, user_id, limit=10, sort_by='created_at', sort_order=-1):
        """
        Get health data for a specific user
        
        Args:
            user_id (str): ID of the user
            limit (int): Maximum number of records to return
            sort_by (str): Field to sort by
            sort_order (int): Sort order (1 for ascending, -1 for descending)
            
        Returns:
            list: List of health data documents
        """
        collection = cls.get_collection()
        cursor = collection.find({'user_id': user_id})
        
        # Apply sorting
        cursor = cursor.sort(sort_by, sort_order)
        
        # Apply limit
        if limit:
            cursor = cursor.limit(limit)
        
        # Convert to list and return
        return list(cursor)
    
    @classmethod
    def update(cls, document_id, updates):
        """
        Update health data document
        
        Args:
            document_id (str): ID of the document to update
            updates (dict): Fields to update
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        collection = cls.get_collection()
        
        # Add updated_at timestamp
        updates['updated_at'] = datetime.now()
        
        result = collection.update_one(
            {'_id': ObjectId(document_id)},
            {'$set': updates}
        )
        
        return result.modified_count > 0
    
    @classmethod
    def delete(cls, document_id):
        """Delete health data document"""
        collection = cls.get_collection()
        result = collection.delete_one({'_id': ObjectId(document_id)})
        return result.deleted_count > 0 