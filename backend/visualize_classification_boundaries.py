"""
Script to visualize classification boundaries for health risk classification.

This script creates detailed visualizations of how the model classifies different 
regions of vital signs, including condition-specific boundaries and probability landscapes.

Usage:
python visualize_classification_boundaries.py [--user_id USER_ID] [--model MODEL_PATH]
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
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('boundary_visualization.log')
    ]
)

logger = logging.getLogger(__name__)

# Make sure modules are in the path
sys.path.append('.')

# Import necessary modules
from services.risk_classification import RiskClassification
from services.classification_model import ClassificationModel
from services.health_service import HealthService
from models.user import User

# Directory for saving results
RESULTS_DIR = "boundary_visualizations"
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_grid_data(resolution=0.5):
    """
    Generate a grid of heart rate and blood oxygen values
    
    Args:
        resolution (float): Step size for the grid
        
    Returns:
        tuple: (xx, yy, X) mesh grid data
    """
    # Create a mesh grid to visualize decision boundaries
    hr_min, hr_max = 25, 190
    bo_min, bo_max = 80, 100
    
    # Create meshgrid with specified resolution
    hr_range = np.arange(hr_min, hr_max, resolution)
    bo_range = np.arange(bo_min, bo_max, resolution)
    xx, yy = np.meshgrid(hr_range, bo_range)
    
    # Reshape for classification
    X = np.c_[xx.ravel(), yy.ravel()]
    
    return xx, yy, X

def classify_grid_points(X, model=None, user_id=None, user_context=None):
    """
    Classify grid points using different methods
    
    Args:
        X (numpy.ndarray): Grid points as [heart_rate, blood_oxygen] pairs
        model (object, optional): Pre-trained model
        user_id (str, optional): User ID for the classification service
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
            if model is not None:
                # Use provided model
                ml_class = model.predict([[hr, bo]])[0]
                ml_probs = model.predict_proba([[hr, bo]])[0]
                
                # Simple 50/50 blend for hybrid
                hybrid_probs = []
                for k in range(3):
                    hybrid_probs.append(0.5 * ml_probs[k] + 0.5 * rule_probs[k])
                hybrid_class = np.argmax(hybrid_probs)
                
                results['ml'][idx] = ml_class
                results['ml_proba'][idx] = ml_probs
                results['hybrid'][idx] = hybrid_class
                results['hybrid_proba'][idx] = hybrid_probs
            elif user_id is not None:
                # Use classification service
                try:
                    ml_result = ClassificationModel.predict_risk_class(user_id, [hr, bo], user_context)
                    ml_class = ml_result['risk_class']
                    ml_probs = [
                        ml_result['probabilities']['low'], 
                        ml_result['probabilities']['medium'], 
                        ml_result['probabilities']['high']
                    ]
                    
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
            else:
                # No model or user ID, just use rule-based
                results['ml'][idx] = rule_class
                results['ml_proba'][idx] = rule_probs
                results['hybrid'][idx] = rule_class
                results['hybrid_proba'][idx] = rule_probs
    
    return results

def visualize_boundaries(xx, yy, results, user_context=None, health_condition=None):
    """
    Visualize classification boundaries
    
    Args:
        xx (numpy.ndarray): Meshgrid x-values
        yy (numpy.ndarray): Meshgrid y-values
        results (dict): Classification results
        user_context (dict, optional): User context information
        health_condition (str, optional): Health condition being visualized
    """
    # Reshape results back to grid
    rule_Z = results['rule'].reshape(xx.shape)
    ml_Z = results['ml'].reshape(xx.shape)
    hybrid_Z = results['hybrid'].reshape(xx.shape)
    
    # Create color maps
    colors = ['green', 'orange', 'red']
    cmap = ListedColormap(colors)
    risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
    
    # Create figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot rule-based boundaries
    contourf1 = axs[0].contourf(xx, yy, rule_Z, alpha=0.8, cmap=cmap, levels=[0, 1, 2, 3])
    axs[0].set_title('Rule-based Classification')
    axs[0].set_xlabel('Heart Rate (BPM)')
    axs[0].set_ylabel('Blood Oxygen (%)')
    axs[0].set_yticks([85, 90, 95, 100])
    
    # Plot ML boundaries
    contourf2 = axs[1].contourf(xx, yy, ml_Z, alpha=0.8, cmap=cmap, levels=[0, 1, 2, 3])
    axs[1].set_title('ML Model Classification')
    axs[1].set_xlabel('Heart Rate (BPM)')
    axs[1].set_ylabel('Blood Oxygen (%)')
    axs[1].set_yticks([85, 90, 95, 100])
    
    # Plot hybrid boundaries
    contourf3 = axs[2].contourf(xx, yy, hybrid_Z, alpha=0.8, cmap=cmap, levels=[0, 1, 2, 3])
    axs[2].set_title('Hybrid Classification')
    axs[2].set_xlabel('Heart Rate (BPM)')
    axs[2].set_ylabel('Blood Oxygen (%)')
    axs[2].set_yticks([85, 90, 95, 100])
    
    # Add condition information in title if available
    if health_condition:
        fig.suptitle(f"Classification Boundaries for {health_condition.title()} Condition", fontsize=16)
    else:
        fig.suptitle("Classification Boundaries", fontsize=16)
    
    # Add a single colorbar for all subplots
    cbar = fig.colorbar(contourf3, ax=axs, orientation='horizontal', pad=0.1)
    cbar.set_ticks([0.33, 1.0, 1.67])
    cbar.set_ticklabels(risk_labels)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    # Save figure
    filename = f"classification_boundaries_{health_condition or 'default'}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    
    # Visualize probability distributions for each risk class
    for risk_class in range(3):
        risk_names = ['Low Risk', 'Medium Risk', 'High Risk']
        
        # Reshape probability results
        rule_proba = results['rule_proba'][:, risk_class].reshape(xx.shape)
        ml_proba = results['ml_proba'][:, risk_class].reshape(xx.shape)
        hybrid_proba = results['hybrid_proba'][:, risk_class].reshape(xx.shape)
        
        # Create figure with three subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot rule-based probabilities
        im1 = axs[0].contourf(xx, yy, rule_proba, alpha=0.8, cmap='Blues', levels=np.linspace(0, 1, 11))
        axs[0].set_title(f'Rule-based: {risk_names[risk_class]} Probability')
        axs[0].set_xlabel('Heart Rate (BPM)')
        axs[0].set_ylabel('Blood Oxygen (%)')
        axs[0].set_yticks([85, 90, 95, 100])
        fig.colorbar(im1, ax=axs[0], label='Probability')
        
        # Plot ML probabilities
        im2 = axs[1].contourf(xx, yy, ml_proba, alpha=0.8, cmap='Blues', levels=np.linspace(0, 1, 11))
        axs[1].set_title(f'ML Model: {risk_names[risk_class]} Probability')
        axs[1].set_xlabel('Heart Rate (BPM)')
        axs[1].set_ylabel('Blood Oxygen (%)')
        axs[1].set_yticks([85, 90, 95, 100])
        fig.colorbar(im2, ax=axs[1], label='Probability')
        
        # Plot hybrid probabilities
        im3 = axs[2].contourf(xx, yy, hybrid_proba, alpha=0.8, cmap='Blues', levels=np.linspace(0, 1, 11))
        axs[2].set_title(f'Hybrid: {risk_names[risk_class]} Probability')
        axs[2].set_xlabel('Heart Rate (BPM)')
        axs[2].set_ylabel('Blood Oxygen (%)')
        axs[2].set_yticks([85, 90, 95, 100])
        fig.colorbar(im3, ax=axs[2], label='Probability')
        
        # Add condition information
        if health_condition:
            fig.suptitle(f"{risk_names[risk_class]} Probability Distribution for {health_condition.title()} Condition", fontsize=16)
        else:
            fig.suptitle(f"{risk_names[risk_class]} Probability Distribution", fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save figure
        filename = f"{risk_class}_{risk_names[risk_class].lower().replace(' ', '_')}_probability_{health_condition or 'default'}.png"
        plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    
    # Visualize regions where ML and rule-based models disagree
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create disagreement mask
    disagreement = (rule_Z != ml_Z).astype(int)
    
    # Plot disagreement regions
    im = ax.contourf(xx, yy, disagreement, alpha=0.5, cmap='Reds')
    fig.colorbar(im, ax=ax, label='Disagreement')
    ax.set_title('Regions where Rule-based and ML Model Disagree')
    ax.set_xlabel('Heart Rate (BPM)')
    ax.set_ylabel('Blood Oxygen (%)')
    
    # Add condition information
    if health_condition:
        fig.suptitle(f"Model Disagreement for {health_condition.title()} Condition", fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    filename = f"model_disagreement_{health_condition or 'default'}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    
def main():
    """Main function to run boundary visualization"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize classification boundaries')
    parser.add_argument('--user_id', type=str, help='User ID for ML model')
    parser.add_argument('--model', type=str, help='Path to pre-trained model file')
    parser.add_argument('--resolution', type=float, default=0.5, help='Grid resolution')
    parser.add_argument('--condition', type=str, default=None, help='Health condition to visualize')
    
    args = parser.parse_args()
    
    # Load model if specified
    model = None
    if args.model and os.path.exists(args.model):
        logger.info(f"Loading model from {args.model}")
        model = joblib.load(args.model)
    
    # Get user context if condition specified
    user_context = None
    if args.condition:
        user_context = {'health_conditions': [args.condition]}
        logger.info(f"Visualizing boundaries for condition: {args.condition}")
    elif args.user_id:
        # Get user context from database
        user = User.get_by_id(args.user_id)
        if user and 'health_conditions' in user and user['health_conditions']:
            user_context = {'health_conditions': user['health_conditions']}
            logger.info(f"Using context from user {args.user_id}: {user_context}")
    
    start_time = datetime.now()
    logger.info("Starting boundary visualization...")
    
    # Generate grid data
    xx, yy, X = generate_grid_data(resolution=args.resolution)
    logger.info(f"Generated grid with {len(X)} points at resolution {args.resolution}")
    
    # Classify grid points
    results = classify_grid_points(X, model, args.user_id, user_context)
    
    # Visualize boundaries
    visualize_boundaries(xx, yy, results, user_context, args.condition)
    
    # Calculate total runtime
    total_time = (datetime.now() - start_time).total_seconds() / 60
    logger.info(f"Boundary visualization complete in {total_time:.2f} minutes")
    
    # Print summary
    print("\n======= VISUALIZATION COMPLETE =======")
    if args.user_id:
        print(f"User ID: {args.user_id}")
    if args.model:
        print(f"Model: {args.model}")
    if args.condition:
        print(f"Health condition: {args.condition}")
    print(f"Grid resolution: {args.resolution}")
    print(f"Visualizations saved to: {RESULTS_DIR}")
    print("========================================\n")

if __name__ == "__main__":
    main()