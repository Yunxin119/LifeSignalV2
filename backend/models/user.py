from datetime import datetime
import hashlib
import uuid
from bson import ObjectId
from database import get_collection

class User:
    """Model for handling user operations in MongoDB"""
    
    COLLECTION_NAME = 'users'
    
    @classmethod
    def get_collection(cls):
        """Get the users collection"""
        return get_collection(cls.COLLECTION_NAME)
    
    @classmethod
    def create(cls, username, email, password, full_name=None, age=None, medical_history=None):
        """
        Create a new user
        
        Args:
            username (str): User's username
            email (str): User's email
            password (str): User's password (will be hashed)
            full_name (str, optional): User's full name
            age (int, optional): User's age
            medical_history (dict, optional): User's medical history
            
        Returns:
            str: ID of the created user document
        """
        collection = cls.get_collection()
        
        # Check if user already exists
        existing_user = collection.find_one({'$or': [{'username': username}, {'email': email}]})
        if existing_user:
            raise ValueError("Username or email already exists")
        
        # Hash the password
        salt = uuid.uuid4().hex
        hashed_password = cls._hash_password(password, salt)
        
        # Create user document
        document = {
            'username': username,
            'email': email,
            'password': hashed_password,
            'salt': salt,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        # Add optional fields
        if full_name:
            document['full_name'] = full_name
        if age:
            document['age'] = age
        if medical_history and isinstance(medical_history, dict):
            document['medical_history'] = medical_history
        
        # Insert document and return ID
        result = collection.insert_one(document)
        return str(result.inserted_id)
    
    @classmethod
    def get_by_id(cls, user_id):
        """Get user by ID"""
        collection = cls.get_collection()
        user = collection.find_one({'_id': ObjectId(user_id)})
        
        # Remove sensitive information
        if user is not None:
            user.pop('password', None)
            user.pop('salt', None)
        
        return user
    
    @classmethod
    def get_by_email(cls, email):
        """Get user by email"""
        collection = cls.get_collection()
        return collection.find_one({'email': email})
    
    @classmethod
    def get_by_username(cls, username):
        """Get user by username"""
        collection = cls.get_collection()
        return collection.find_one({'username': username})
    
    @classmethod
    def authenticate(cls, username_or_email, password):
        """
        Authenticate a user
        
        Args:
            username_or_email (str): Username or email
            password (str): Password to verify
            
        Returns:
            dict or None: User document if authentication succeeded, None otherwise
        """
        collection = cls.get_collection()
        
        # Find user by username or email
        user = collection.find_one({
            '$or': [
                {'username': username_or_email},
                {'email': username_or_email}
            ]
        })
        
        if user is None:
            return None
        
        # Verify password
        hashed_password = cls._hash_password(password, user['salt'])
        if hashed_password != user['password']:
            return None
        
        # Remove sensitive information
        user.pop('password', None)
        user.pop('salt', None)
        
        return user
    
    @classmethod
    def update(cls, user_id, updates):
        """Update user"""
        collection = cls.get_collection()
        
        # Handle password updates separately
        if 'password' in updates:
            salt = uuid.uuid4().hex
            updates['password'] = cls._hash_password(updates['password'], salt)
            updates['salt'] = salt
        
        # Add updated_at timestamp
        updates['updated_at'] = datetime.now()
        
        result = collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': updates}
        )
        
        return result.modified_count > 0
    
    @classmethod
    def delete(cls, user_id):
        """Delete user"""
        collection = cls.get_collection()
        result = collection.delete_one({'_id': ObjectId(user_id)})
        return result.deleted_count > 0
    
    @staticmethod
    def _hash_password(password, salt):
        """Hash a password with the given salt"""
        return hashlib.sha256(f"{password}{salt}".encode()).hexdigest() 