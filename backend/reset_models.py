"""
Script to reset and regenerate ML models after code changes.

Usage:
python reset_models.py [--regenerate]
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

# Directory where models are stored
MODEL_DIR = "models"

def reset_models():
    """
    Reset all machine learning models and create fresh directory structure.
    This will:
    1. Back up current models to a timestamped directory
    2. Remove old models directory
    3. Create new empty model directories
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Directories to manage
    model_dirs = [
        "models",
        "user_models",
        "improved_models"
    ]
    
    # Create backup directory
    backup_dir = f"model_backups_{current_time}"
    os.makedirs(backup_dir, exist_ok=True)
    logger.info(f"Created backup directory: {backup_dir}")
    
    # Backup and reset each model directory
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            # Create subdirectory in backup
            backup_subdir = os.path.join(backup_dir, model_dir)
            os.makedirs(backup_subdir, exist_ok=True)
            
            # Copy all files
            for filename in os.listdir(model_dir):
                src_path = os.path.join(model_dir, filename)
                dst_path = os.path.join(backup_subdir, filename)
                
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Backed up: {src_path} → {dst_path}")
            
            # Remove directory
            shutil.rmtree(model_dir)
            logger.info(f"Removed directory: {model_dir}")
        
        # Create fresh directory
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Created fresh directory: {model_dir}")

def cleanup_test_results():
    """Clean up test result directories"""
    test_dirs = [
        "model_test_results",
        "dashboard_results"
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            # Create a backup
            backup_name = f"{test_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(test_dir, backup_name)
            logger.info(f"Backed up {test_dir} to {backup_name}")
            
            # Clean the directory
            shutil.rmtree(test_dir)
            os.makedirs(test_dir, exist_ok=True)
            logger.info(f"Reset test directory: {test_dir}")
        else:
            os.makedirs(test_dir, exist_ok=True)
            logger.info(f"Created test directory: {test_dir}")

if __name__ == "__main__":
    logger.info("Starting model reset process...")
    reset_models()
    cleanup_test_results()
    logger.info("Model reset complete!")
    
    # Instructions for next steps
    print("\n======= NEXT STEPS =======")
    print("1. The models have been reset and the directories are now empty.")
    print("2. New models will be created automatically when the service runs.")
    print("3. Restart the API service to begin using the new models.")
    print("4. Old models have been backed up to a timestamped directory.")
    print("===========================\n") 