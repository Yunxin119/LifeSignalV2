"""
Condition-Aware Health ML Model Dashboard Generator

This script creates a dashboard visualizing the performance of the condition-aware ML model.
"""

import os
import sys
import pandas as pd
import json
import logging
from datetime import datetime
from glob import glob
from sklearn.metrics import mean_absolute_error
import bisect
import shutil
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TEST_RESULTS_DIR = "model_test_results"
DASHBOARD_DIR = "dashboard_results"

class ConditionAwareDashboard:
    """Creates a dashboard to visualize the performance of condition-aware health ML models"""
    
    def __init__(self, test_results_dir=TEST_RESULTS_DIR, dashboard_dir=DASHBOARD_DIR):
        """Initialize the dashboard creator"""
        self.test_results_dir = test_results_dir
        self.dashboard_dir = dashboard_dir
        os.makedirs(dashboard_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
        # Dashboard components
        self.sections = []
        self.condition_results = {}
        self.health_conditions = []
        
    def load_test_data(self, test_results_file=None):
        """Load test results data from CSV"""
        # If specific file not provided, find the most recent one
        if not test_results_file:
            csv_files = glob(f"{self.test_results_dir}/*_comparison_*.csv")
            if not csv_files:
                logger.error(f"No test result files found in {self.test_results_dir}")
                return False
            
            # Sort by modification time and get the latest
            test_results_file = max(csv_files, key=os.path.getmtime)
        
        logger.info(f"Loading test results from {test_results_file}")
        
        try:
            # Load the data
            self.df = pd.read_csv(test_results_file)
            
            # Ensure numeric columns are properly converted
            numeric_columns = ['heart_rate', 'blood_oxygen', 'rule_risk', 'hybrid_risk', 'pure_ml_risk', 'hybrid_diff', 'pure_ml_diff']
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Handle any missing values
            self.df = self.df.dropna(subset=['condition', 'rule_risk', 'hybrid_risk', 'pure_ml_risk'])
            
            logger.info(f"Loaded data with {len(self.df)} samples and columns: {', '.join(self.df.columns)}")
            
            # Extract unique conditions
            self.health_conditions = sorted(self.df['condition'].unique())
            logger.info(f"Found {len(self.health_conditions)} conditions: {', '.join(self.health_conditions)}")
            
            # Group results by condition
            for condition in self.health_conditions:
                condition_df = self.df[self.df['condition'] == condition]
                
                # Calculate metrics for this condition
                metrics = {
                    'samples': len(condition_df),
                    'avg_rule_risk': condition_df['rule_risk'].mean(),
                    'avg_hybrid_risk': condition_df['hybrid_risk'].mean(),
                    'avg_pure_ml_risk': condition_df['pure_ml_risk'].mean(),
                    'hybrid_mae': mean_absolute_error(condition_df['rule_risk'], condition_df['hybrid_risk']),
                    'pure_ml_mae': mean_absolute_error(condition_df['rule_risk'], condition_df['pure_ml_risk']),
                    'avg_hybrid_diff': (condition_df['hybrid_risk'] - condition_df['rule_risk']).mean(),
                    'avg_pure_ml_diff': (condition_df['pure_ml_risk'] - condition_df['rule_risk']).mean(),
                    'max_hybrid_diff': abs(condition_df['hybrid_risk'] - condition_df['rule_risk']).max(),
                    'max_pure_ml_diff': abs(condition_df['pure_ml_risk'] - condition_df['rule_risk']).max()
                }
                
                self.condition_results[condition] = {
                    'df': condition_df,
                    'metrics': metrics
                }
                
                logger.info(f"Condition '{condition}': {metrics['samples']} samples, Hybrid MAE: {metrics['hybrid_mae']:.2f}, Pure ML MAE: {metrics['pure_ml_mae']:.2f}")
                
            logger.info(f"Loaded data for {len(self.health_conditions)} conditions with {len(self.df)} total samples")
            return True
            
        except Exception as e:
            logger.error(f"Error loading test results: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def create_dashboard(self, output_file=None):
        """Create the full dashboard report"""
        if not hasattr(self, 'df') or len(self.df) == 0:
            logger.error("No test data loaded. Run load_test_data first.")
            return False
        
        if not output_file:
            output_file = f"{self.dashboard_dir}/health_models_comparison_{self.timestamp}.html"
        
        logger.info(f"Creating dashboard at {output_file}")
        
        # Create dashboard sections
        self._create_title_page()
        self._create_performance_summary()
        self._create_model_comparison_overview()
        
        # Create detailed condition pages
        for condition in self.health_conditions:
            self._create_condition_detail_page(condition)
        
        # Create HTML report
        html_content = '\n'.join(self.sections)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard created at {output_file}")
        return output_file
    
    def _create_title_page(self):
        """Create the title page for the dashboard"""
        title_html = f"""
        <div class="title-page">
            <h1>Health Risk Prediction Models Comparison</h1>
            <h2>Performance Analysis Dashboard</h2>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="models-summary">
                <h3>Models Comparison:</h3>
                <ul>
                    <li><strong>Rule-based Model:</strong> Traditional threshold-based approach</li>
                    <li><strong>Hybrid Model:</strong> Condition-aware ML approach</li>
                    <li><strong>Pure ML Model:</strong> Machine learning without condition awareness</li>
                </ul>
            </div>
            
            <div class="conditions-tested">
                <h3>Health Conditions Tested:</h3>
                <ul>
                    {"".join([f'<li>{condition}</li>' for condition in self.health_conditions])}
                </ul>
            </div>
            
            <div class="navigation">
                <h3>Dashboard Sections:</h3>
                <ul>
                    <li><a href="#performance-summary">Performance Summary</a></li>
                    <li><a href="#model-comparison">Model Comparison Overview</a></li>
                    <li><a href="#condition-details">Condition-Specific Details</a></li>
                </ul>
            </div>
        </div>
        
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            .title-page {{
                text-align: center;
                margin-bottom: 50px;
                padding: 20px;
                background: #f5f5f5;
                border-radius: 8px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .models-summary, .conditions-tested, .navigation {{
                margin: 20px auto;
                max-width: 600px;
                text-align: left;
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            a {{
                color: #3498db;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            li {{
                margin: 5px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .section {{
                margin: 40px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .performance-table {{
                overflow-x: auto;
            }}
            .chart-container {{
                margin: 20px 0;
            }}
        </style>
        """
        
        self.sections.append(title_html)
    
    def _create_performance_summary(self):
        """Create performance summary section with overall metrics"""
        # Extract performance metrics by condition
        condition_metrics = []
        
        for condition in self.health_conditions:
            condition_data = self.condition_results[condition]
            metrics = condition_data['metrics']
            
            condition_metrics.append({
                'condition': condition,
                'samples': metrics['samples'],
                'hybrid_mae': metrics['hybrid_mae'],
                'pure_ml_mae': metrics['pure_ml_mae'],
                'avg_hybrid_diff': metrics['avg_hybrid_diff'],
                'avg_pure_ml_diff': metrics['avg_pure_ml_diff']
            })
        
        # Sort by sample count
        condition_metrics = sorted(condition_metrics, key=lambda x: x['samples'], reverse=True)
        
        # Prepare data for charts
        mae_chart_data = {
            'labels': [metrics['condition'] for metrics in condition_metrics],
            'datasets': [
                {
                    'label': 'Hybrid Model',
                    'data': [metrics['hybrid_mae'] for metrics in condition_metrics],
                    'backgroundColor': 'rgba(54, 162, 235, 0.5)',
                    'borderColor': 'rgba(54, 162, 235, 1)',
                    'borderWidth': 1
                },
                {
                    'label': 'Pure ML Model',
                    'data': [metrics['pure_ml_mae'] for metrics in condition_metrics],
                    'backgroundColor': 'rgba(255, 99, 132, 0.5)',
                    'borderColor': 'rgba(255, 99, 132, 1)',
                    'borderWidth': 1
                }
            ]
        }
        
        diff_chart_data = {
            'labels': [metrics['condition'] for metrics in condition_metrics],
            'datasets': [
                {
                    'label': 'Hybrid Model',
                    'data': [metrics['avg_hybrid_diff'] for metrics in condition_metrics],
                    'backgroundColor': 'rgba(54, 162, 235, 0.5)',
                    'borderColor': 'rgba(54, 162, 235, 1)',
                    'borderWidth': 1
                },
                {
                    'label': 'Pure ML Model',
                    'data': [metrics['avg_pure_ml_diff'] for metrics in condition_metrics],
                    'backgroundColor': 'rgba(255, 99, 132, 0.5)',
                    'borderColor': 'rgba(255, 99, 132, 1)',
                    'borderWidth': 1
                }
            ]
        }
        
        # Convert chart data to JSON strings
        mae_chart_json = json.dumps(mae_chart_data)
        diff_chart_json = json.dumps(diff_chart_data)
        
        # Create HTML table
        table_rows = ""
        for metrics in condition_metrics:
            row = f"""
            <tr>
                <td>{metrics['condition']}</td>
                <td>{metrics['samples']}</td>
                <td>{metrics['hybrid_mae']:.4f}</td>
                <td>{metrics['pure_ml_mae']:.4f}</td>
                <td>{metrics['avg_hybrid_diff']:.4f}</td>
                <td>{metrics['avg_pure_ml_diff']:.4f}</td>
            </tr>
            """
            table_rows += row
        
        # Build the HTML in parts to avoid nested f-strings
        summary_html_start = """
        <div id="performance-summary" class="section">
            <h2>Performance Summary by Condition</h2>
            
            <div class="performance-table">
                <table>
                    <thead>
                        <tr>
                            <th>Health Condition</th>
                            <th>Samples</th>
                            <th>Hybrid MAE</th>
                            <th>Pure ML MAE</th>
                            <th>Avg Hybrid Diff</th>
                            <th>Avg Pure ML Diff</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        summary_html_middle = table_rows
        
        summary_html_end = """
                    </tbody>
                </table>
            </div>
            
            <div class="chart-container">
                <h3>Model Performance by Condition</h3>
                <div style="display: flex; flex-wrap: wrap; justify-content: center;">
                    <div style="flex: 1; min-width: 400px; margin: 10px;">
                        <h4>Mean Absolute Error</h4>
                        <canvas id="maeChart"></canvas>
                    </div>
                    <div style="flex: 1; min-width: 400px; margin: 10px;">
                        <h4>Average Difference (ML - Rule)</h4>
                        <canvas id="diffChart"></canvas>
                    </div>
                </div>
                
                <script>
                    // MAE Chart
                    var maeCtx = document.getElementById('maeChart').getContext('2d');
                    var maeChart = new Chart(maeCtx, {
                        type: 'bar',
                        data: MAE_CHART_DATA_PLACEHOLDER,
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    title: {
                                        display: true,
                                        text: 'Mean Absolute Error'
                                    }
                                },
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Health Condition'
                                    }
                                }
                            }
                        }
                    });
                    
                    // Difference Chart
                    var diffCtx = document.getElementById('diffChart').getContext('2d');
                    var diffChart = new Chart(diffCtx, {
                        type: 'bar',
                        data: DIFF_CHART_DATA_PLACEHOLDER,
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Average Difference (ML - Rule)'
                                    }
                                },
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Health Condition'
                                    }
                                }
                            }
                        }
                    });
                </script>
            </div>
        </div>
        """
        
        # Replace placeholders with JSON data
        summary_html_end = summary_html_end.replace("MAE_CHART_DATA_PLACEHOLDER", mae_chart_json)
        summary_html_end = summary_html_end.replace("DIFF_CHART_DATA_PLACEHOLDER", diff_chart_json)
        
        # Combine all parts
        summary_html = summary_html_start + summary_html_middle + summary_html_end
        
        self.sections.append(summary_html)
        
        # Add risk threshold comparison to performance summary
        self._create_condition_threshold_comparison()
    
    def _create_model_comparison_overview(self):
        """Create model comparison overview section"""
        
        # Calculate overall performance
        overall_hybrid_mae = mean_absolute_error(self.df['rule_risk'], self.df['hybrid_risk'])
        overall_pure_ml_mae = mean_absolute_error(self.df['rule_risk'], self.df['pure_ml_risk'])
        
        # Scatter plot data for ML vs Rule comparison
        scatter_data_hybrid = []
        scatter_data_pure_ml = []
        
        # Limit to max 500 points for performance
        sample_df = self.df.sample(min(500, len(self.df))) if len(self.df) > 500 else self.df
        
        for _, row in sample_df.iterrows():
            scatter_data_hybrid.append({
                'x': float(row['rule_risk']),
                'y': float(row['hybrid_risk']),
                'condition': row['condition']
            })
            scatter_data_pure_ml.append({
                'x': float(row['rule_risk']),
                'y': float(row['pure_ml_risk']),
                'condition': row['condition']
            })
        
        # Convert conditions list to JSON
        conditions_json = json.dumps(self.health_conditions)
        # Convert scatter data to JSON
        hybrid_data_json = json.dumps(scatter_data_hybrid)
        pure_ml_data_json = json.dumps(scatter_data_pure_ml)
        
        comparison_html = f"""
        <div id="model-comparison" class="section">
            <h2>Model Comparison Overview</h2>
            <p>This section provides a detailed comparison between the three risk prediction models.</p>
            
            <div class="metrics-summary">
                <h3>Overall Performance</h3>
                <ul>
                    <li><strong>Overall Hybrid Model MAE:</strong> {overall_hybrid_mae:.4f}</li>
                    <li><strong>Overall Pure ML Model MAE:</strong> {overall_pure_ml_mae:.4f}</li>
                    <li><strong>Total Samples:</strong> {len(self.df)}</li>
                </ul>
            </div>
            
            <div class="chart-container">
                <h3>Hybrid Model vs Rule-based Model</h3>
                <canvas id="hybridScatterChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>Pure ML Model vs Rule-based Model</h3>
                <canvas id="pureMLScatterChart"></canvas>
            </div>
            
            <script>
                // Color scale for conditions
                const conditions = CONDITIONS_LIST_PLACEHOLDER;
                const colorScale = [
                    'rgba(255, 99, 132, 0.7)',
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(255, 206, 86, 0.7)',
                    'rgba(75, 192, 192, 0.7)',
                    'rgba(153, 102, 255, 0.7)',
                    'rgba(255, 159, 64, 0.7)',
                    'rgba(199, 199, 199, 0.7)',
                    'rgba(83, 102, 255, 0.7)',
                    'rgba(40, 159, 64, 0.7)',
                    'rgba(210, 99, 132, 0.7)'
                ];
                
                const getColor = (condition) => {{
                    const index = conditions.indexOf(condition);
                    return index >= 0 ? colorScale[index % colorScale.length] : 'gray';
                }};
                
                // Hybrid Scatter Chart
                var hybridCtx = document.getElementById('hybridScatterChart').getContext('2d');
                var hybridScatter = new Chart(hybridCtx, {{
                    type: 'scatter',
                    data: {{
                        datasets: conditions.map(condition => ({{
                            label: condition,
                            data: HYBRID_DATA_PLACEHOLDER.filter(point => point.condition === condition),
                            backgroundColor: getColor(condition),
                            pointRadius: 4
                        }}))
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        const point = context.raw;
                                        return point.condition + ' - Rule: ' + point.x.toFixed(2) + ', Hybrid: ' + point.y.toFixed(2);
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                title: {{
                                    display: true,
                                    text: 'Rule-based Risk Score'
                                }},
                                min: 0,
                                max: 100
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Hybrid Model Risk Score'
                                }},
                                min: 0,
                                max: 100
                            }}
                        }}
                    }}
                }});
                
                // Pure ML Scatter Chart
                var pureMLCtx = document.getElementById('pureMLScatterChart').getContext('2d');
                var pureMLScatter = new Chart(pureMLCtx, {{
                    type: 'scatter',
                    data: {{
                        datasets: conditions.map(condition => ({{
                            label: condition,
                            data: PURE_ML_DATA_PLACEHOLDER.filter(point => point.condition === condition),
                            backgroundColor: getColor(condition),
                            pointRadius: 4
                        }}))
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        const point = context.raw;
                                        return point.condition + ' - Rule: ' + point.x.toFixed(2) + ', Pure ML: ' + point.y.toFixed(2);
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                title: {{
                                    display: true,
                                    text: 'Rule-based Risk Score'
                                }},
                                min: 0,
                                max: 100
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Pure ML Model Risk Score'
                                }},
                                min: 0,
                                max: 100
                            }}
                        }}
                    }}
                }});
            </script>
        </div>
        """
        
        # Replace placeholders with actual data
        comparison_html = comparison_html.replace("CONDITIONS_LIST_PLACEHOLDER", conditions_json)
        comparison_html = comparison_html.replace("HYBRID_DATA_PLACEHOLDER", hybrid_data_json)
        comparison_html = comparison_html.replace("PURE_ML_DATA_PLACEHOLDER", pure_ml_data_json)
        
        self.sections.append(comparison_html)
    
    def _create_condition_detail_page(self, condition):
        """Create detailed analysis page for a specific health condition"""
        condition_data = self.condition_results[condition]
        metrics = condition_data['metrics']
        condition_df = condition_data['df']
        
        # Create heatmap data - sample to create a grid
        hr_min, hr_max = 40, 180
        bo_min, bo_max = 80, 100
        
        hr_range = list(range(hr_min, hr_max + 1, 10))
        bo_range = list(range(bo_max, bo_min - 1, -2))  # Reversed for display
        
        # Create heatmap data for each model
        rule_heatmap = [[0 for _ in range(len(hr_range))] for _ in range(len(bo_range))]
        hybrid_heatmap = [[0 for _ in range(len(hr_range))] for _ in range(len(bo_range))]
        pure_ml_heatmap = [[0 for _ in range(len(hr_range))] for _ in range(len(bo_range))]
        
        # Sample some points from condition_df to create heatmap
        sample_df = condition_df.sample(min(100, len(condition_df))) if len(condition_df) > 100 else condition_df
        
        for _, row in sample_df.iterrows():
            hr = row['heart_rate']
            bo = row['blood_oxygen']
            hr_idx = bisect.bisect_left(hr_range, hr)
            bo_idx = bisect.bisect_left(list(reversed(bo_range)), bo)
            
            if 0 <= hr_idx < len(hr_range) and 0 <= bo_idx < len(bo_range):
                rule_heatmap[bo_idx][hr_idx] = row['rule_risk']
                hybrid_heatmap[bo_idx][hr_idx] = row['hybrid_risk']
                pure_ml_heatmap[bo_idx][hr_idx] = row['pure_ml_risk']
        
        # Scatter data for differences
        hybrid_diff_data = []
        pure_ml_diff_data = []
        
        for _, row in sample_df.iterrows():
            hybrid_diff_data.append({
                'x': float(row['heart_rate']),
                'y': float(row['blood_oxygen']),
                'r': min(20, abs(row['hybrid_diff']) * 2 + 2),  # Size based on difference
                'diff': float(row['hybrid_diff'])
            })
            
            pure_ml_diff_data.append({
                'x': float(row['heart_rate']),
                'y': float(row['blood_oxygen']),
                'r': min(20, abs(row['pure_ml_diff']) * 2 + 2),  # Size based on difference
                'diff': float(row['pure_ml_diff'])
            })
        
        # Convert data to JSON strings to avoid f-string issues
        hr_labels_json = json.dumps(hr_range)
        bo_labels_json = json.dumps(bo_range)
        rule_heatmap_json = json.dumps(rule_heatmap)
        hybrid_heatmap_json = json.dumps(hybrid_heatmap)
        pure_ml_heatmap_json = json.dumps(pure_ml_heatmap)
        hybrid_diff_json = json.dumps(hybrid_diff_data)
        pure_ml_diff_json = json.dumps(pure_ml_diff_data)
        
        condition_html = f"""
        <div id="condition-{condition}" class="section">
            <h2>Condition Analysis: {condition}</h2>
            
            <div class="metrics-summary">
                <h3>Performance Metrics</h3>
                <ul>
                    <li><strong>Samples:</strong> {metrics['samples']}</li>
                    <li><strong>Average Rule-based Risk:</strong> {metrics['avg_rule_risk']:.2f}</li>
                    <li><strong>Average Hybrid Model Risk:</strong> {metrics['avg_hybrid_risk']:.2f}</li>
                    <li><strong>Average Pure ML Risk:</strong> {metrics['avg_pure_ml_risk']:.2f}</li>
                    <li><strong>Hybrid Model MAE:</strong> {metrics['hybrid_mae']:.2f}</li>
                    <li><strong>Pure ML Model MAE:</strong> {metrics['pure_ml_mae']:.2f}</li>
                    <li><strong>Average Hybrid Difference:</strong> {metrics['avg_hybrid_diff']:.2f}</li>
                    <li><strong>Average Pure ML Difference:</strong> {metrics['avg_pure_ml_diff']:.2f}</li>
                </ul>
            </div>
            
            <div class="chart-container">
                <h3>Risk Score Heatmaps</h3>
                <div style="display: flex; flex-wrap: wrap; justify-content: center;">
                    <div style="flex: 1; min-width: 300px; margin: 10px;">
                        <h4>Rule-based Model</h4>
                        <canvas id="ruleHeatmap-{condition}"></canvas>
                    </div>
                    <div style="flex: 1; min-width: 300px; margin: 10px;">
                        <h4>Hybrid Model</h4>
                        <canvas id="hybridHeatmap-{condition}"></canvas>
                    </div>
                    <div style="flex: 1; min-width: 300px; margin: 10px;">
                        <h4>Pure ML Model</h4>
                        <canvas id="pureMLHeatmap-{condition}"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Model Differences</h3>
                <div style="display: flex; flex-wrap: wrap; justify-content: center;">
                    <div style="flex: 1; min-width: 400px; margin: 10px;">
                        <h4>Hybrid Model Differences</h4>
                        <canvas id="hybridDiff-{condition}"></canvas>
                    </div>
                    <div style="flex: 1; min-width: 400px; margin: 10px;">
                        <h4>Pure ML Model Differences</h4>
                        <canvas id="pureMLDiff-{condition}"></canvas>
                    </div>
                </div>
            </div>
            
            <script>
                // Heatmap configuration helper
                function createHeatmap(ctx, data, labels, title) {{
                    const chartData = {{
                        labels: labels.x,
                        datasets: labels.y.map((label, i) => ({{
                            label: label,
                            data: data[i],
                            fill: false,
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            borderColor: 'rgb(255, 99, 132)',
                            pointBackgroundColor: 'rgb(255, 99, 132)',
                            pointBorderColor: '#fff',
                            pointHoverBackgroundColor: '#fff',
                            pointHoverBorderColor: 'rgb(255, 99, 132)'
                        }}))
                    }};
                    
                    return new Chart(ctx, {{
                        type: 'heatmap',
                        data: chartData,
                        options: {{
                            plugins: {{
                                tooltip: {{
                                    callbacks: {{
                                        label: function(context) {{
                                            return `Blood Oxygen: ${{labels.y[context.dataIndex]}}, Heart Rate: ${{labels.x[context.datasetIndex]}}, Risk: ${{context.raw.toFixed(2)}}`;
                                        }}
                                    }}
                                }}
                            }},
                            scales: {{
                                x: {{
                                    title: {{
                                        display: true,
                                        text: 'Heart Rate'
                                    }}
                                }},
                                y: {{
                                    title: {{
                                        display: true,
                                        text: 'Blood Oxygen'
                                    }}
                                }}
                            }}
                        }}
                    }});
                }}
                
                // Create heatmaps for this condition
                const hrLabels = HR_LABELS_PLACEHOLDER;
                const boLabels = BO_LABELS_PLACEHOLDER;
                const heatmapLabels = {{ x: hrLabels, y: boLabels }};
                
                // Rule-based heatmap
                const ruleCtx = document.getElementById('ruleHeatmap-{condition}').getContext('2d');
                createHeatmap(ruleCtx, RULE_HEATMAP_PLACEHOLDER, heatmapLabels);
                
                // Hybrid heatmap
                const hybridCtx = document.getElementById('hybridHeatmap-{condition}').getContext('2d');
                createHeatmap(hybridCtx, HYBRID_HEATMAP_PLACEHOLDER, heatmapLabels);
                
                // Pure ML heatmap
                const pureMLCtx = document.getElementById('pureMLHeatmap-{condition}').getContext('2d');
                createHeatmap(pureMLCtx, PURE_ML_HEATMAP_PLACEHOLDER, heatmapLabels);
                
                // Bubble chart for differences
                function createDiffBubbleChart(ctx, data, title) {{
                    return new Chart(ctx, {{
                        type: 'bubble',
                        data: {{
                            datasets: [{{
                                label: 'Positive Differences',
                                data: data.filter(point => point.diff >= 0),
                                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                                borderColor: 'rgba(75, 192, 192, 1)'
                            }},
                            {{
                                label: 'Negative Differences',
                                data: data.filter(point => point.diff < 0),
                                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                                borderColor: 'rgba(255, 99, 132, 1)'
                            }}]
                        }},
                        options: {{
                            plugins: {{
                                tooltip: {{
                                    callbacks: {{
                                        label: function(context) {{
                                            const point = context.raw;
                                            return `Heart Rate: ${{point.x}}, Blood Oxygen: ${{point.y}}, Diff: ${{point.diff.toFixed(2)}}`;
                                        }}
                                    }}
                                }}
                            }},
                            scales: {{
                                x: {{
                                    title: {{
                                        display: true,
                                        text: 'Heart Rate'
                                    }},
                                    min: 40,
                                    max: 180
                                }},
                                y: {{
                                    title: {{
                                        display: true,
                                        text: 'Blood Oxygen'
                                    }},
                                    min: 80,
                                    max: 100
                                }}
                            }}
                        }}
                    }});
                }}
                
                // Create difference bubble charts
                const hybridDiffCtx = document.getElementById('hybridDiff-{condition}').getContext('2d');
                createDiffBubbleChart(hybridDiffCtx, HYBRID_DIFF_PLACEHOLDER);
                
                const pureMLDiffCtx = document.getElementById('pureMLDiff-{condition}').getContext('2d');
                createDiffBubbleChart(pureMLDiffCtx, PURE_ML_DIFF_PLACEHOLDER);
            </script>
        </div>
        """
        
        # Replace placeholders with actual data
        condition_html = condition_html.replace("HR_LABELS_PLACEHOLDER", hr_labels_json)
        condition_html = condition_html.replace("BO_LABELS_PLACEHOLDER", bo_labels_json)
        condition_html = condition_html.replace("RULE_HEATMAP_PLACEHOLDER", rule_heatmap_json)
        condition_html = condition_html.replace("HYBRID_HEATMAP_PLACEHOLDER", hybrid_heatmap_json)
        condition_html = condition_html.replace("PURE_ML_HEATMAP_PLACEHOLDER", pure_ml_heatmap_json)
        condition_html = condition_html.replace("HYBRID_DIFF_PLACEHOLDER", hybrid_diff_json)
        condition_html = condition_html.replace("PURE_ML_DIFF_PLACEHOLDER", pure_ml_diff_json)
        
        self.sections.append(condition_html)

    def _create_condition_threshold_comparison(self):
        """Create a visualization comparing risk thresholds across conditions"""
        # Create comparison table HTML
        html = """
        <div class="threshold-comparison section">
            <h2 id="threshold-comparison">Risk Threshold Comparison by Condition</h2>
            <p>This table shows how normal ranges for vital signs differ across health conditions:</p>
            
            <div class="performance-table">
                <table>
                    <thead>
                        <tr>
                            <th>Health Condition</th>
                            <th>Normal Heart Rate (min)</th>
                            <th>Normal Heart Rate (max)</th>
                            <th>Normal Blood Oxygen (min)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Healthy</td>
                            <td>60 bpm</td>
                            <td>100 bpm</td>
                            <td>95%</td>
                        </tr>
                        <tr>
                            <td>COPD</td>
                            <td>60 bpm</td>
                            <td>100 bpm</td>
                            <td>92%</td>
                        </tr>
                        <tr>
                            <td>Anxiety</td>
                            <td>60 bpm</td>
                            <td>115 bpm</td>
                            <td>95%</td>
                        </tr>
                        <tr>
                            <td>Heart Disease</td>
                            <td>60 bpm</td>
                            <td>100 bpm</td>
                            <td>95%</td>
                        </tr>
                        <tr>
                            <td>Athlete</td>
                            <td>50 bpm</td>
                            <td>100 bpm</td>
                            <td>95%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="chart-container">
                <h3>Risk Threshold Visualization</h3>
                <div style="display: flex; justify-content: center; margin-top: 20px;">
                    <canvas id="thresholdChart" width="800" height="400"></canvas>
                </div>
                
                <script>
                    // Create the threshold visualization chart
                    const ctx = document.getElementById('thresholdChart').getContext('2d');
                    
                    // Define the data manually without complex JSON
                    const conditions = ["Healthy", "COPD", "Anxiety", "Heart Disease", "Athlete"];
                    const hrLowData = [60, 60, 60, 60, 50];
                    const hrHighData = [100, 100, 115, 100, 100];
                    const boLowData = [95, 92, 95, 95, 95];
                    
                    const thresholdChart = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: conditions,
                            datasets: [
                                {
                                    label: 'Min Heart Rate (bpm)',
                                    data: hrLowData,
                                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                                    borderColor: 'rgba(54, 162, 235, 1)',
                                    borderWidth: 1
                                },
                                {
                                    label: 'Max Heart Rate (bpm)',
                                    data: hrHighData,
                                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                                    borderColor: 'rgba(255, 99, 132, 1)',
                                    borderWidth: 1
                                },
                                {
                                    label: 'Min Blood Oxygen (%)',
                                    data: boLowData,
                                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                                    borderColor: 'rgba(75, 192, 192, 1)',
                                    borderWidth: 1
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: false,
                                    min: 40,
                                    max: 120,
                                    title: {
                                        display: true,
                                        text: 'Value'
                                    }
                                },
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Health Condition'
                                    }
                                }
                            },
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Normal Vital Sign Ranges by Health Condition'
                                }
                            }
                        }
                    });
                </script>
            </div>
            
            <div class="threshold-impact">
                <h3>Impact on Risk Calculation</h3>
                <p>These adjusted thresholds affect how risk scores are calculated for patients with different health conditions:</p>
                <ul>
                    <li><strong>COPD/Emphysema:</strong> Lower blood oxygen threshold (92% vs 95%) means that blood oxygen levels between 92-95% are considered normal for COPD patients but would indicate elevated risk for others.</li>
                    <li><strong>Anxiety:</strong> Higher maximum heart rate threshold (115 bpm vs 100 bpm) accounts for the fact that anxiety patients often have higher baseline heart rates without indicating the same level of risk.</li>
                    <li><strong>Athletes:</strong> Lower minimum heart rate threshold (50 bpm vs 60 bpm) recognizes that athletes typically have lower resting heart rates due to increased cardiovascular efficiency.</li>
                </ul>
                <p>The condition-aware models are trained to incorporate these threshold adjustments when calculating risk scores, leading to more personalized and accurate risk assessments.</p>
            </div>
        </div>
        """
        
        self.sections.append(html)
        
def main():
    """Create dashboard from test results"""
    parser = argparse.ArgumentParser(description="Create dashboard for condition-aware ML model")
    parser.add_argument("--results-dir", default=TEST_RESULTS_DIR, help="Directory with test results")
    parser.add_argument("--output-dir", default=DASHBOARD_DIR, help="Directory to save dashboard")
    parser.add_argument("--clean", action="store_true", help="Clean output directory before running")
    parser.add_argument("--test-file", help="Specific test result file to use (optional)")
    args = parser.parse_args()
    
    # Clean output directory if requested
    if args.clean and os.path.exists(args.output_dir):
        logger.info(f"Cleaning output directory: {args.output_dir}")
        
        # Create backup folder with timestamp
        if os.listdir(args.output_dir):  # Only backup if not empty
            backup_dir = f"{args.output_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(args.output_dir, backup_dir)
            logger.info(f"Backed up existing results to {backup_dir}")
        
        # Clean directory
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dashboard
    dashboard = ConditionAwareDashboard(test_results_dir=args.results_dir, dashboard_dir=args.output_dir)
    
    # Load the test data first
    if args.test_file:
        logger.info(f"Using specified test file: {args.test_file}")
        if not dashboard.load_test_data(args.test_file):
            logger.error("Failed to load specified test file")
            return 1
    else:
        logger.info("Searching for latest test results file")
        if not dashboard.load_test_data():
            logger.error("Failed to load test data")
            return 1
    
    # Now create the dashboard with the loaded data
    if dashboard.create_dashboard():
        logger.info(f"Dashboard created successfully in {args.output_dir}")
        
        # List generated visualizations
        files = os.listdir(args.output_dir)
        logger.info("Generated files:")
        for file in files:
            logger.info(f"  - {file}")
                
        return 0
    else:
        logger.error("Failed to create dashboard")
        return 1

if __name__ == "__main__":
    sys.exit(main())