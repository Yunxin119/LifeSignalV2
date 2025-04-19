"""
Script to reset and regenerate classification models.

Usage:
python reset_classification_models.py
"""

import os
import sys
import glob
import logging
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Directory where classification models are stored
MODEL_DIR = "classification_models"

def reset_models():
    """
    Reset all classification models and create fresh directory structure.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create backup directory if models exist
    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        backup_dir = f"model_backups_{current_time}"
        os.makedirs(backup_dir, exist_ok=True)
        logger.info(f"Created backup directory: {backup_dir}")
        
        # Create subdirectory in backup
        backup_subdir = os.path.join(backup_dir, MODEL_DIR)
        os.makedirs(backup_subdir, exist_ok=True)
        
        # Copy all files
        for filename in os.listdir(MODEL_DIR):
            src_path = os.path.join(MODEL_DIR, filename)
            dst_path = os.path.join(backup_subdir, filename)
            
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                logger.info(f"Backed up: {src_path} → {dst_path}")
        
        # Remove directory
        shutil.rmtree(MODEL_DIR)
        logger.info(f"Removed directory: {MODEL_DIR}")
    
    # Create fresh directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Created fresh directory: {MODEL_DIR}")

if __name__ == "__main__":
    logger.info("Starting classification model reset process...")
    reset_models()
    logger.info("Classification model reset complete!")
    
    # Instructions for next steps
    print("\n======= NEXT STEPS =======")
    print("1. Classification models have been reset and the directory is now empty.")
    print("2. New models will be created automatically when the service runs.")
    print("3. Restart the API service to begin using the new models.")
    print("===========================\n")