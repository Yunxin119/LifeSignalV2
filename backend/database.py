from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging
from config import MONGODB_URI, DATABASE_NAME, DEBUG

# Configure logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoDB:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDB, cls).__new__(cls)
            cls._instance.client = None
            cls._instance.db = None
            cls._instance.connect()
        return cls._instance
    
    def connect(self):
        """Establish connection to MongoDB"""
        try:
            if not MONGODB_URI:
                raise ValueError("MongoDB URI is not configured")
            
            self.client = MongoClient(MONGODB_URI)
            # Ping the server to verify connection
            self.client.admin.command('ping')
            logger.info("Connected to MongoDB successfully")
            
            # Get database
            self.db = self.client[DATABASE_NAME]
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise
    
    def get_collection(self, collection_name):
        """Get collection from database"""
        if self.db is None:
            raise ConnectionError("Database connection not established")
        return self.db[collection_name]
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# Create a singleton instance
db = MongoDB()

# Function to get a collection
def get_collection(collection_name):
    return db.get_collection(collection_name) 