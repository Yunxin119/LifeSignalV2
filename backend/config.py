import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv('MONGODB_URI')
DATABASE_NAME = 'lifesignal'

# JWT configuration
JWT_SECRET = os.getenv('JWT_SECRET')
JWT_TOKEN_EXPIRY = 24 * 60 * 60  # 24 hours in seconds

# Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# App configuration
DEBUG = os.getenv('DEBUG', '0') == '1'

# Flask configuration
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
FLASK_PORT = int(os.getenv('FLASK_PORT', '5100'))
FLASK_HOST = os.getenv('FLASK_HOST', '::') 