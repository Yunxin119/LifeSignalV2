import jwt
from datetime import datetime, timedelta
import logging
from config import JWT_SECRET, JWT_TOKEN_EXPIRY, DEBUG
from models.user import User

# Configure logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AuthService:
    """Service for handling authentication"""
    
    @staticmethod
    def generate_token(user):
        """
        Generate JWT token for a user
        
        Args:
            user (dict): User document
            
        Returns:
            str: JWT token
        """
        try:
            # Create payload with user ID and expiry
            payload = {
                'user_id': str(user['_id']),
                'username': user['username'],
                'exp': datetime.utcnow() + timedelta(seconds=JWT_TOKEN_EXPIRY)
            }
            
            # Generate token
            token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
            
            return token
        except Exception as e:
            logger.error(f"Error generating token: {e}")
            raise
    
    @staticmethod
    def validate_token(token):
        """
        Validate JWT token
        
        Args:
            token (str): JWT token
            
        Returns:
            dict: Token payload if valid, None otherwise
        """
        try:
            # Decode token
            payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return None
    
    @staticmethod
    def authenticate(username_or_email, password):
        """
        Authenticate a user and generate token
        
        Args:
            username_or_email (str): Username or email
            password (str): Password
            
        Returns:
            dict: Authentication result with token and user info
        """
        try:
            # Authenticate user
            user = User.authenticate(username_or_email, password)
            
            if user is None:
                return {
                    'success': False,
                    'message': 'Invalid credentials'
                }
            
            # Generate token
            token = AuthService.generate_token(user)
            
            return {
                'success': True,
                'token': token,
                'user': {
                    'id': str(user['_id']),
                    'username': user['username'],
                    'email': user['email']
                }
            }
        except Exception as e:
            logger.error(f"Error in authentication: {e}")
            return {
                'success': False,
                'message': str(e)
            }
    
    @staticmethod
    def register(username, email, password, **kwargs):
        """
        Register a new user
        
        Args:
            username (str): Username
            email (str): Email
            password (str): Password
            **kwargs: Additional user information
            
        Returns:
            dict: Registration result with token and user info
        """
        try:
            # Create user
            user_id = User.create(username, email, password, **kwargs)
            
            # Get created user
            user = User.get_by_id(user_id)
            
            if user is None:
                return {
                    'success': False,
                    'message': 'Failed to create user'
                }
            
            # Generate token
            token = AuthService.generate_token(user)
            
            return {
                'success': True,
                'token': token,
                'user': {
                    'id': str(user['_id']),
                    'username': user['username'],
                    'email': user['email']
                }
            }
        except ValueError as e:
            logger.warning(f"Validation error in registration: {e}")
            return {
                'success': False,
                'message': str(e)
            }
        except Exception as e:
            logger.error(f"Error in registration: {e}")
            return {
                'success': False,
                'message': str(e)
            } 