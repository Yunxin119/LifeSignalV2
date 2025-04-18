"""
Health ML Model Performance Dashboard

This script creates an interactive dashboard for visualizing health risk prediction model performance.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from datetime import datetime
import argparse
import glob

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required project modules
from services.health_service import HealthService
from services.health_ml_service import HealthMLService
from services.feature_engineering import FeatureEngineering
from train.data_simulator import HealthDataSimulator

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelDashboard:
    """Interactive dashboard for visualizing model performance"""
    
    def __init__(self, results_dir="model_test_results", output_dir="dashboard_results"):
        """Initialize the dashboard with results directory"""
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_test_data(self):
        """Load all available test data"""
        logger.info(f"Loading test data from {self.results_dir}")
        
        # Find all CSV files with test results
        csv_files = glob.glob(os.path.join(self.results_dir, "*.csv"))
        
        # Initialize data containers
        self.condition_data = None
        self.age_group_data = None
        self.boundary_data = {}
        
        # Load data from files
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            
            if 'all_conditions' in file_name:
                # Load condition test data
                self.condition_data = pd.read_csv(file_path)
                logger.info(f"Loaded condition data: {len(self.condition_data)} records")
                
            elif 'age_groups' in file_name:
                # Load age group data
                self.age_group_data = pd.read_csv(file_path)
                logger.info(f"Loaded age group data: {len(self.age_group_data)} records")
                
            elif 'boundary' in file_name:
                # Load boundary test data
                data = pd.read_csv(file_path)
                
                # Extract range name from filename
                parts = file_name.split('_')
                range_index = parts.index('range')
                range_name = '_'.join(parts[range_index:]).split('.')[0]
                
                if range_name not in self.boundary_data:
                    self.boundary_data[range_name] = data
                    logger.info(f"Loaded boundary data for {range_name}: {len(data)} records")
        
        # Check what data was loaded
        if self.condition_data is None and self.age_group_data is None and not self.boundary_data:
            logger.warning("No test data found. You may need to run the model testing script first.")
            return False
        
        return True
    
    def create_dashboard(self):
        """Create comprehensive dashboard with visualizations"""
        if not hasattr(self, 'condition_data') or self.condition_data is None:
            if not self.load_test_data():
                logger.error("Failed to load test data. Exiting.")
                return False
        
        logger.info("Creating dashboard visualizations")
        
        # Create main dashboard figure
        plt.figure(figsize=(20, 12))
        plt.suptitle("Health Risk Prediction Model Performance Dashboard", fontsize=20)
        
        # Create grid for plots
        gs = plt.GridSpec(3, 3, figure=plt.gcf())
        
        # Visualize condition performance if data available
        if self.condition_data is not None:
            self._plot_condition_performance(gs[0, :2])
            self._plot_error_distribution(gs[1, 0])
            self._plot_error_by_vitals(gs[1, 1])
        
        # Visualize age group performance if data available
        if self.age_group_data is not None:
            self._plot_age_group_performance(gs[0, 2])
        
        # Visualize boundary performance if data available
        if self.boundary_data:
            self._plot_decision_boundaries(gs[1:, 2])
        
        # Visualize overall metrics
        self._plot_overall_metrics(gs[2, :2])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save dashboard
        dashboard_path = os.path.join(self.output_dir, f"model_dashboard_{self.timestamp}.png")
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved dashboard to {dashboard_path}")
        
        # Create additional detailed visualizations
        self._create_detailed_visualizations()
        
        return True
    
    def _plot_condition_performance(self, grid_pos):
        """Plot performance comparison across health conditions"""
        ax = plt.subplot(grid_pos)
        
        # Calculate metrics by condition
        condition_metrics = self.condition_data.groupby('condition').agg({
            'ml_risk': 'mean',
            'true_risk': 'mean',
            'diff': ['mean', 'std']
        })
        
        # Reshape for plotting
        plot_data = pd.DataFrame({
            'condition': condition_metrics.index,
            'avg_diff': condition_metrics[('diff', 'mean')],
            'std_diff': condition_metrics[('diff', 'std')]
        })
        
        # Sort by average difference
        plot_data = plot_data.sort_values('avg_diff')
        
        # Plot error by condition
        colors = ['green' if abs(val) < 5 else 'orange' if abs(val) < 10 else 'red' 
                  for val in plot_data['avg_diff']]
        
        bars = ax.barh(plot_data['condition'], plot_data['avg_diff'], 
                     xerr=plot_data['std_diff'], alpha=0.7, color=colors)
        
        # Add value labels
        for bar, value in zip(bars, plot_data['avg_diff']):
            width = bar.get_width()
            if width >= 0:
                x_pos = width + 1
                ha = 'left'
            else:
                x_pos = width - 1
                ha = 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{value:.2f}", 
                   va='center', ha=ha, fontsize=8)
        
        ax.set_xlabel('ML Risk - Rule Risk (Average Difference)')
        ax.set_title('Performance by Health Condition')
        ax.axvline(x=0, color='gray', linestyle='--')
        
        # Add zone indicators
        ax.axvspan(-5, 5, alpha=0.1, color='green')
        ax.axvspan(-10, -5, alpha=0.1, color='orange')
        ax.axvspan(5, 10, alpha=0.1, color='orange')
        ax.axvspan(-20, -10, alpha=0.1, color='red')
        ax.axvspan(10, 20, alpha=0.1, color='red')
        
        return ax
    
    def _plot_error_distribution(self, grid_pos):
        """Plot error distribution histogram"""
        ax = plt.subplot(grid_pos)
        
        # Plot error histogram
        sns.histplot(self.condition_data['diff'], bins=20, kde=True, ax=ax)
        ax.set_title('Distribution of ML-Rule Differences')
        ax.set_xlabel('Difference (ML Risk - Rule Risk)')
        
        # Calculate and show statistics
        mean_error = self.condition_data['diff'].mean()
        median_error = self.condition_data['diff'].median()
        std_error = self.condition_data['diff'].std()
        
        stats_text = f"Mean: {mean_error:.2f}\nMedian: {median_error:.2f}\nStd Dev: {std_error:.2f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8))
        
        return ax
    
    def _plot_error_by_vitals(self, grid_pos):
        """Plot error by vital signs"""
        ax = plt.subplot(grid_pos)
        
        # Create scatter plot of error by heart rate
        scatter = ax.scatter(
            self.condition_data['heart_rate'], 
            self.condition_data['blood_oxygen'],
            c=self.condition_data['diff'], 
            cmap='coolwarm', 
            alpha=0.6,
            vmin=-20,
            vmax=20
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('ML Risk - Rule Risk')
        
        # Format tick labels to 2 decimal places
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        
        ax.set_xlabel('Heart Rate (BPM)')
        ax.set_ylabel('Blood Oxygen (%)')
        ax.set_title('Model Error by Vital Signs')
        
        return ax
    
    def _plot_age_group_performance(self, grid_pos):
        """Plot performance across age groups"""
        ax = plt.subplot(grid_pos)
        
        if self.age_group_data is None:
            ax.text(0.5, 0.5, "No age group data available", 
                   ha='center', va='center', transform=ax.transAxes)
            return ax
        
        # Calculate metrics by age group
        age_metrics = self.age_group_data.groupby('age_group').agg({
            'ml_risk': 'mean',
            'true_risk': 'mean',
            'diff': ['mean', 'std', 'count']
        }).reset_index()
        
        # Ensure age groups are in the correct order
        if 'age_group' in age_metrics.columns:
            # Sort age groups
            age_metrics['min_age'] = age_metrics['age_group'].apply(
                lambda x: int(x.split('-')[0]) if '-' in str(x) else 0
            )
            age_metrics = age_metrics.sort_values('min_age')
        
        # Plot average risks by age group
        width = 0.35
        x = np.arange(len(age_metrics))
        
        # Create grouped bar chart
        bars1 = ax.bar(x - width/2, age_metrics['true_risk']['mean'], width, 
                      label='Rule Risk', color='steelblue')
        bars2 = ax.bar(x + width/2, age_metrics['ml_risk']['mean'], width,
                      label='ML Risk', color='darkorange')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f"{height:.2f}", ha='center', va='bottom', fontsize=8)
        
        # Add chart details
        ax.set_ylabel('Risk Score')
        ax.set_title('Risk Estimation by Age Group')
        ax.set_xticks(x)
        ax.set_xticklabels(age_metrics['age_group'])
        ax.legend()
        
        # Add error information
        for i, metrics in enumerate(age_metrics.itertuples()):
            if hasattr(metrics, 'diff') and hasattr(metrics.diff, 'mean'):
                diff = metrics.diff['mean']
                err_text = f"Diff: {diff:.2f}"
                ax.text(i, 0.1, err_text, ha='center', fontsize=7, 
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
        
        return ax
    
    def _plot_decision_boundaries(self, grid_pos):
        """Plot decision boundary visualizations"""
        ax = plt.subplot(grid_pos)
        
        if not self.boundary_data:
            ax.text(0.5, 0.5, "No boundary data available", 
                   ha='center', va='center', transform=ax.transAxes)
            return ax
        
        # Take the first boundary dataset for the dashboard
        range_name = list(self.boundary_data.keys())[0]
        boundary_df = self.boundary_data[range_name]
        
        # Filter to a single profile for the dashboard
        profiles = boundary_df['profile'].unique()
        if len(profiles) > 0:
            # Pick a profile with interesting differences if possible
            profile_diffs = {}
            for profile in profiles:
                profile_data = boundary_df[boundary_df['profile'] == profile]
                profile_diffs[profile] = profile_data['difference'].abs().mean()
            
            # Select profile with the most interesting differences
            selected_profile = max(profile_diffs, key=profile_diffs.get)
            boundary_df = boundary_df[boundary_df['profile'] == selected_profile]
        
        # Create a pivot table for heatmap
        if 'heart_rate' in boundary_df.columns and 'blood_oxygen' in boundary_df.columns:
            try:
                # Round values to reduce unique combinations if needed
                boundary_df['heart_rate_bin'] = boundary_df['heart_rate'].round(0)
                boundary_df['blood_oxygen_bin'] = boundary_df['blood_oxygen'].round(0)
                
                # Create pivot table
                heatmap_data = boundary_df.pivot_table(
                    values='difference', 
                    index='blood_oxygen_bin', 
                    columns='heart_rate_bin',
                    aggfunc='mean'
                )
                
                # Plot heatmap
                sns.heatmap(heatmap_data, cmap='coolwarm', center=0, 
                           vmin=-15, vmax=15, ax=ax)
                
                # Format tick labels to show only 2 decimal places
                ax.collections[0].colorbar.formatter = plt.FuncFormatter(lambda x, _: f"{x:.2f}")
                ax.collections[0].colorbar.update_ticks()
                
                ax.set_title(f'Risk Difference Heatmap\nProfile: {selected_profile if "selected_profile" in locals() else "Default"}')
                ax.set_xlabel('Heart Rate (BPM)')
                ax.set_ylabel('Blood Oxygen (%)')
            except Exception as e:
                logger.error(f"Failed to create heatmap: {e}")
                ax.text(0.5, 0.5, f"Heatmap error: {e}", 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Missing required columns for boundary visualization", 
                   ha='center', va='center', transform=ax.transAxes)
        
        return ax
    
    def _plot_overall_metrics(self, grid_pos):
        """Plot overall model performance metrics"""
        ax = plt.subplot(grid_pos)
        
        # Combine all available data
        all_data = []
        
        if self.condition_data is not None:
            all_data.append(self.condition_data)
        
        if self.age_group_data is not None:
            all_data.append(self.age_group_data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Calculate metrics
            metrics = {}
            
            # Calculate MAE and RMSE
            if 'true_risk' in combined_data.columns and 'ml_risk' in combined_data.columns:
                metrics['MAE'] = mean_absolute_error(
                    combined_data['true_risk'], combined_data['ml_risk'])
                metrics['RMSE'] = np.sqrt(mean_squared_error(
                    combined_data['true_risk'], combined_data['ml_risk']))
            
            # Calculate percentage of predictions within different error ranges
            if 'diff' in combined_data.columns:
                metrics['Within ±5'] = (combined_data['diff'].abs() <= 5).mean() * 100
                metrics['Within ±10'] = (combined_data['diff'].abs() <= 10).mean() * 100
                metrics['Within ±15'] = (combined_data['diff'].abs() <= 15).mean() * 100
            
            # Calculate correlation
            if 'true_risk' in combined_data.columns and 'ml_risk' in combined_data.columns:
                metrics['Correlation'] = combined_data['true_risk'].corr(combined_data['ml_risk'])
            
            # Create metrics display
            metrics_text = "\n".join([
                f"{metric}: {value:.2f}{' %' if 'Within' in metric else ''}"
                for metric, value in metrics.items()
            ])
            
            # Create scatter plot
            if 'true_risk' in combined_data.columns and 'ml_risk' in combined_data.columns:
                scatter = ax.scatter(
                    combined_data['true_risk'], 
                    combined_data['ml_risk'],
                    alpha=0.4, 
                    c=combined_data['diff'] if 'diff' in combined_data.columns else 'blue',
                    cmap='coolwarm' if 'diff' in combined_data.columns else None,
                    vmin=-20 if 'diff' in combined_data.columns else None,
                    vmax=20 if 'diff' in combined_data.columns else None
                )
                
                # Add colorbar if using difference for color
                if 'diff' in combined_data.columns:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('ML - Rule Difference')
                    # Format tick labels to 2 decimal places
                    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
                
                # Add perfect prediction line
                min_val = min(combined_data['true_risk'].min(), combined_data['ml_risk'].min())
                max_val = max(combined_data['true_risk'].max(), combined_data['ml_risk'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                
                ax.set_xlabel('Rule-Based Risk Score')
                ax.set_ylabel('ML-Based Risk Score')
                ax.set_title('ML vs Rule-Based Risk Prediction')
                
                # Add metrics box
                ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=9,
                       bbox=dict(facecolor='white', alpha=0.9))
            else:
                # If we don't have the required columns, just show metrics
                ax.text(0.5, 0.5, f"Overall Metrics:\n\n{metrics_text}",
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, bbox=dict(facecolor='white', alpha=0.9))
        else:
            ax.text(0.5, 0.5, "No data available for metrics calculation", 
                   ha='center', va='center', transform=ax.transAxes)
        
        return ax
    
    def _create_detailed_visualizations(self):
        """Create additional detailed visualizations"""
        logger.info("Creating detailed visualizations")
        
        if self.condition_data is not None:
            self._create_condition_detail_plots()
        
        if self.boundary_data:
            self._create_boundary_detail_plots()
    
    def _create_condition_detail_plots(self):
        """Create detailed plots for each condition"""
        logger.info("Creating condition-specific detail plots")
        
        conditions = self.condition_data['condition'].unique()
        
        for condition in conditions:
            condition_data = self.condition_data[self.condition_data['condition'] == condition]
            
            plt.figure(figsize=(15, 10))
            plt.suptitle(f"Performance Detail: {condition}", fontsize=16)
            
            # Create grid for plots
            gs = plt.GridSpec(2, 2, figure=plt.gcf())
            
            # Plot 1: Scatter plot of true vs ml risk
            ax1 = plt.subplot(gs[0, 0])
            ax1.scatter(condition_data['true_risk'], condition_data['ml_risk'], alpha=0.6)
            ax1.set_xlabel('Rule-Based Risk')
            ax1.set_ylabel('ML-Based Risk')
            ax1.set_title('Risk Prediction Comparison')
            
            # Add perfect prediction line
            min_val = min(condition_data['true_risk'].min(), condition_data['ml_risk'].min())
            max_val = max(condition_data['true_risk'].max(), condition_data['ml_risk'].max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # Calculate metrics
            mae = mean_absolute_error(condition_data['true_risk'], condition_data['ml_risk'])
            rmse = np.sqrt(mean_squared_error(condition_data['true_risk'], condition_data['ml_risk']))
            corr = condition_data['true_risk'].corr(condition_data['ml_risk'])
            
            metrics_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nCorr: {corr:.2f}"
            ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.9))
            
            # Plot 2: Error histogram
            ax2 = plt.subplot(gs[0, 1])
            sns.histplot(condition_data['diff'], bins=15, kde=True, ax=ax2)
            ax2.set_title('Distribution of Differences')
            ax2.set_xlabel('ML Risk - Rule Risk')
            
            mean_diff = condition_data['diff'].mean()
            median_diff = condition_data['diff'].median()
            std_diff = condition_data['diff'].std()
            
            diff_text = f"Mean Diff: {mean_diff:.2f}\nMedian Diff: {median_diff:.2f}\nStd Dev: {std_diff:.2f}"
            ax2.text(0.95, 0.95, diff_text, transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.9))
            
            # Plot 3: Error by heart rate
            ax3 = plt.subplot(gs[1, 0])
            scatter3 = ax3.scatter(condition_data['heart_rate'], condition_data['diff'], 
                                 alpha=0.6, c=condition_data['blood_oxygen'], cmap='viridis')
            ax3.set_xlabel('Heart Rate (BPM)')
            ax3.set_ylabel('ML Risk - Rule Risk')
            ax3.set_title('Error by Heart Rate')
            ax3.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            cbar3 = plt.colorbar(scatter3, ax=ax3)
            cbar3.set_label('Blood Oxygen (%)')
            # Format tick labels to 2 decimal places
            cbar3.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
            
            # Plot 4: Error by blood oxygen
            ax4 = plt.subplot(gs[1, 1])
            scatter4 = ax4.scatter(condition_data['blood_oxygen'], condition_data['diff'], 
                                 alpha=0.6, c=condition_data['heart_rate'], cmap='plasma')
            ax4.set_xlabel('Blood Oxygen (%)')
            ax4.set_ylabel('ML Risk - Rule Risk')
            ax4.set_title('Error by Blood Oxygen')
            ax4.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            cbar4 = plt.colorbar(scatter4, ax=ax4)
            cbar4.set_label('Heart Rate (BPM)')
            # Format tick labels to 2 decimal places
            cbar4.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # Save the figure
            condition_file = condition.replace(" ", "_").lower()
            detail_path = os.path.join(self.output_dir, f"detail_{condition_file}_{self.timestamp}.png")
            plt.savefig(detail_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved detail plot for {condition} to {detail_path}")
    
    def _create_boundary_detail_plots(self):
        """Create detailed boundary visualization plots"""
        logger.info("Creating detailed boundary plots")
        
        for range_name, boundary_df in self.boundary_data.items():
            profiles = boundary_df['profile'].unique()
            
            for profile in profiles:
                profile_data = boundary_df[boundary_df['profile'] == profile]
                
                # Create pivot tables for different visualizations
                try:
                    # Round values to reduce unique combinations
                    profile_data['heart_rate_bin'] = profile_data['heart_rate'].round(0)
                    profile_data['blood_oxygen_bin'] = profile_data['blood_oxygen'].round(0)
                    
                    # Create pivot tables
                    diff_data = profile_data.pivot_table(
                        values='difference', 
                        index='blood_oxygen_bin', 
                        columns='heart_rate_bin',
                        aggfunc='mean'
                    )
                    
                    true_data = profile_data.pivot_table(
                        values='true_risk', 
                        index='blood_oxygen_bin', 
                        columns='heart_rate_bin',
                        aggfunc='mean'
                    )
                    
                    ml_data = profile_data.pivot_table(
                        values='ml_risk', 
                        index='blood_oxygen_bin', 
                        columns='heart_rate_bin',
                        aggfunc='mean'
                    )
                    
                    # Create visualization
                    plt.figure(figsize=(18, 6))
                    plt.suptitle(f"Risk Model Comparison: {profile}\nRange: {range_name}", fontsize=16)
                    
                    # Plot 1: Rule-based risk
                    ax1 = plt.subplot(131)
                    im1 = sns.heatmap(true_data, cmap='viridis', ax=ax1)
                    ax1.set_title('Rule-Based Risk')
                    ax1.set_xlabel('Heart Rate (BPM)')
                    ax1.set_ylabel('Blood Oxygen (%)')
                    # Format colorbar to show 2 decimal places
                    im1.collections[0].colorbar.formatter = plt.FuncFormatter(lambda x, _: f"{x:.2f}")
                    im1.collections[0].colorbar.update_ticks()
                    
                    # Plot 2: ML-based risk
                    ax2 = plt.subplot(132)
                    im2 = sns.heatmap(ml_data, cmap='viridis', ax=ax2)
                    ax2.set_title('ML-Based Risk')
                    ax2.set_xlabel('Heart Rate (BPM)')
                    ax2.set_ylabel('Blood Oxygen (%)')
                    # Format colorbar to show 2 decimal places
                    im2.collections[0].colorbar.formatter = plt.FuncFormatter(lambda x, _: f"{x:.2f}")
                    im2.collections[0].colorbar.update_ticks()
                    
                    # Plot 3: Difference
                    ax3 = plt.subplot(133)
                    im3 = sns.heatmap(diff_data, cmap='coolwarm', center=0, 
                                    vmin=-15, vmax=15, ax=ax3)
                    ax3.set_title('ML - Rule Difference')
                    ax3.set_xlabel('Heart Rate (BPM)')
                    ax3.set_ylabel('Blood Oxygen (%)')
                    # Format colorbar to show 2 decimal places
                    im3.collections[0].colorbar.formatter = plt.FuncFormatter(lambda x, _: f"{x:.2f}")
                    im3.collections[0].colorbar.update_ticks()
                    
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.85)
                    
                    # Save the figure
                    profile_file = profile.replace(" ", "_").lower()
                    range_file = range_name.replace(" ", "_").lower()
                    detail_path = os.path.join(
                        self.output_dir, 
                        f"boundary_{range_file}_{profile_file}_{self.timestamp}.png"
                    )
                    plt.savefig(detail_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Saved boundary plot for {profile} to {detail_path}")
                except Exception as e:
                    logger.error(f"Failed to create boundary plot for {profile}: {e}")
        
def main():
    """Main function to run the dashboard"""
    parser = argparse.ArgumentParser(description='Generate model performance dashboard')
    parser.add_argument('--results-dir', default='model_test_results',
                        help='Directory containing model test results')
    parser.add_argument('--output-dir', default='dashboard_results',
                        help='Directory to save dashboard visualizations')
    
    args = parser.parse_args()
    
    dashboard = ModelDashboard(results_dir=args.results_dir, output_dir=args.output_dir)
    dashboard.create_dashboard()

if __name__ == "__main__":
    main()