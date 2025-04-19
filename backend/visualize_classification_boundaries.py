"""
Script to visualize classification boundaries for health risk classification.

This script:
1. Generates a dense grid of vital sign values
2. Classifies each point with various methods
3. Visualizes decision boundaries
4. Creates comparative visualizations

Usage:
python visualize_classification_boundaries.py --user_id YOUR_USER_ID
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import seaborn as sns
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('boundaries_visualization.log')
    ]
)

logger = logging.getLogger(__name__)

# Make sure the services modules are in the path
sys.path.append('.')

# Import necessary modules
from services.risk_classification import RiskClassification
from services.classification_model import ClassificationModel
from services.health_service import HealthService
from models.user import User

# Directory for saving results
RESULTS_DIR = "boundary_visualizations"
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_grid_data():
    """
    Generate a grid of heart rate and blood oxygen values
    
    Returns:
        tuple: (xx, yy, X) grid data
    """
    # Create a mesh grid to visualize decision boundaries
    hr_min, hr_max = 30, 180
    bo_min, bo_max = 80, 100
    
    # Step size in the mesh
    hr_step = 2
    bo_step = 0.2
    
    # Create meshgrid
    hr_range = np.arange(hr_min, hr_max, hr_step)
    bo_range = np.arange(bo_min, bo_max, bo_step)
    xx, yy = np.meshgrid(hr_range, bo_range)
    
    # Reshape for classification
    X = np.c_[xx.ravel(), yy.ravel()]
    
    return xx, yy, X

def classify_grid_points(X, user_id, user_context=None):
    """
    Classify grid points using different methods
    
    Args:
        X (numpy.ndarray): Grid points as [heart_rate, blood_oxygen] pairs
        user_id (str): User ID
        user_context (dict, optional): User context information
        
    Returns:
        dict: Classification results
    """
    n_samples = len(X)
    logger.info(f"Classifying {n_samples} grid points...")
    
    # Initialize classification results
    results = {
        'rule': np.zeros(n_samples, dtype=int),
        'ml': np.zeros(n_samples, dtype=int),
        'hybrid': np.zeros(n_samples, dtype=int),
        'rule_proba': np.zeros((n_samples, 3)),
        'ml_proba': np.zeros((n_samples, 3)),
        'hybrid_proba': np.zeros((n_samples, 3))
    }
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch = X[i:end]
        
        logger.info(f"Processing batch {i//batch_size + 1}/{(n_samples-1)//batch_size + 1}...")
        
        # Classify each point in the batch
        for j, (hr, bo) in enumerate(batch):
            idx = i + j
            
            # Rule-based classification
            risk_score = HealthService.calculate_risk_score(hr, bo, user_context)
            rule_probs = RiskClassification.score_to_probabilities(risk_score)
            rule_class = RiskClassification.score_to_class(risk_score)
            
            results['rule'][idx] = rule_class
            results['rule_proba'][idx] = rule_probs
            
            # ML classification
            try:
                ml_result = ClassificationModel.predict_risk_class(user_id, [hr, bo], user_context)
                ml_class = ml_result['risk_class']
                ml_probs = [ml_result['probabilities']['low'], 
                           ml_result['probabilities']['medium'], 
                           ml_result['probabilities']['high']]
                
                results['ml'][idx] = ml_class
                results['ml_proba'][idx] = ml_probs
                results['hybrid'][idx] = ml_class  # Hybrid result already computed in predict_risk_class
                results['hybrid_proba'][idx] = ml_probs
            except Exception as e:
                logger.warning(f"Error in ML classification: {e}")
                results['ml'][idx] = rule_class
                results['ml_proba'][idx] = rule_probs
                results['hybrid'][idx] = rule_class
                results['hybrid_proba'][idx] = rule_probs
    
    return results

def visualize_boundaries(xx, yy, results, user_context=None):
    """
    Visualize classification boundaries
    
    Args:
        xx (numpy.ndarray): Meshgrid x-values
        yy (numpy.ndarray): Meshgrid y-values
        results (dict): Classification results
        user_context (dict, optional): User context information
    """
    # Reshape results back to grid
    rule_Z = results['rule'].reshape(xx.shape)
    ml_Z = results['ml'].reshape(xx.shape)
    hybrid_Z = results['hybrid'].reshape(xx.shape)
    
    # Create color maps
    colors = ['green', 'orange', 'red']
    cmap = ListedColormap(colors)
    
    # Create figure
    plt.figure(figsize=(18, 6))
    
    # Plot rule-based boundaries
    plt.subplot(1, 3, 1)
    plt.contourf(xx, yy, rule_Z, alpha=0.8, cmap=cmap)
    plt.title('Rule-based Classification Boundaries')
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Blood Oxygen (%)')
    
    # Add color bar
    norm = mcolors.BoundaryNorm([0, 1, 2, 3], cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=[0.33, 1, 1.67], label='Risk Class')
    plt.clim(-0.5, 2.5)
    plt.colorbar(sm, ticks=[0.33, 1, 1.67])
    plt.clim(-0.5, 2.5)
    plt.gca().set_yticks([85, 90, 95, 100])
    
    # Plot ML boundaries
    plt.subplot(1, 3, 2)
    plt.contourf(xx, yy, ml_Z, alpha=0.8, cmap=cmap)
    plt.title('ML Model Classification Boundaries')
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Blood Oxygen (%)')
    
    # Add color bar
    norm = mcolors.BoundaryNorm([0, 1, 2, 3], cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=[0.33, 1, 1.67], label='Risk Class')
    plt.clim(-0.5, 2.5)
    plt.gca().set_yticks([85, 90, 95, 100])
    
    # Plot hybrid boundaries
    plt.subplot(1, 3, 3)
    plt.contourf(xx, yy, hybrid_Z, alpha=0.8, cmap=cmap)
    plt.title('Hybrid Classification Boundaries')
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Blood Oxygen (%)')
    
    # Add color bar
    norm = mcolors.BoundaryNorm([0, 1, 2, 3], cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=[0.33, 1, 1.67], label='Risk Class')
    plt.clim(-0.5, 2.5)
    plt.gca().set_yticks([85, 90, 95, 100])
    
    # Add context information in title if available
    conditions = []
    if user_context and 'health_conditions' in user_context and user_context['health_conditions']:
        conditions = user_context['health_conditions']
        
    if conditions:
        plt.suptitle(f"Classification Boundaries\nHealth Conditions: {', '.join(conditions)}", fontsize=16)
    else:
        plt.suptitle("Classification Boundaries", fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure
    plt.savefig(os.path.join(RESULTS_DIR, 'classification_boundaries.png'), dpi=300)
    
    # Visualize probability distributions
    visualize_probability_distributions(xx, yy, results)

def visualize_probability_distributions(xx, yy, results):
    """
    Visualize probability distributions for each risk class
    
    Args:
        xx (numpy.ndarray): Meshgrid x-values
        yy (numpy.ndarray): Meshgrid y-values
        results (dict): Classification results
    """
    # Reshape probability results back to grid
    for risk_class in range(3):
        rule_proba = results['rule_proba'][:, risk_class].reshape(xx.shape)
        ml_proba = results['ml_proba'][:, risk_class].reshape(xx.shape)
        hybrid_proba = results['hybrid_proba'][:, risk_class].reshape(xx.shape)
        
        # Create figure
        plt.figure(figsize=(18, 6))
        
        # Plot rule-based probabilities
        plt.subplot(1, 3, 1)
        plt.contourf(xx, yy, rule_proba, alpha=0.8, cmap='Blues')
        plt.colorbar(label='Probability')
        plt.title(f'Rule-based: {["Low", "Medium", "High"][risk_class]} Risk Probability')
        plt.xlabel('Heart Rate (BPM)')
        plt.ylabel('Blood Oxygen (%)')
        plt.gca().set_yticks([85, 90, 95, 100])
        
        # Plot ML probabilities
        plt.subplot(1, 3, 2)
        plt.contourf(xx, yy, ml_proba, alpha=0.8, cmap='Blues')
        plt.colorbar(label='Probability')
        plt.title(f'ML Model: {["Low", "Medium", "High"][risk_class]} Risk Probability')
        plt.xlabel('Heart Rate (BPM)')
        plt.ylabel('Blood Oxygen (%)')
        plt.gca().set_yticks([85, 90, 95, 100])
        
        # Plot hybrid probabilities
        plt.subplot(1, 3, 3)
        plt.contourf(xx, yy, hybrid_proba, alpha=0.8, cmap='Blues')
        plt.colorbar(label='Probability')
        plt.title(f'Hybrid: {["Low", "Medium", "High"][risk_class]} Risk Probability')
        plt.xlabel('Heart Rate (BPM)')
        plt.ylabel('Blood Oxygen (%)')
        plt.gca().set_yticks([85, 90, 95, 100])
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{["low", "medium", "high"][risk_class]}_risk_probability.png'), dpi=300)

def main():
    """Main function to run visualization"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize classification boundaries')
    parser.add_argument('--user_id', required=True, help='User ID for ML model')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info(f"Starting boundary visualization for user {args.user_id}...")
    
    # Get user context
    user = User.get_by_id(args.user_id)
    user_context = {}
    if user:
        if 'age' in user:
            user_context['age'] = user['age']
        if 'health_conditions' in user:
            user_context['health_conditions'] = user['health_conditions']
    
    # Generate grid data
    xx, yy, X = generate_grid_data()
    
    # Classify grid points
    results = classify_grid_points(X, args.user_id, user_context)
    
    # Visualize boundaries
    visualize_boundaries(xx, yy, results, user_context)
    
    # Calculate total runtime
    total_time = (datetime.now() - start_time).total_seconds() / 60
    logger.info(f"Visualization complete in {total_time:.2f} minutes")
    
    # Print summary
    print("\n======= VISUALIZATION COMPLETE =======")
    print(f"User ID: {args.user_id}")
    if user_context and 'health_conditions' in user_context and user_context['health_conditions']:
        print(f"Health conditions: {', '.join(user_context['health_conditions'])}")
    print(f"Visualizations saved to: {RESULTS_DIR}")
    print("=======================================\n")

if __name__ == "__main__":
    main()