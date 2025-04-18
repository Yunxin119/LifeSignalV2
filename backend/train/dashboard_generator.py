"""
Dashboard component generator for health model dashboard.
This module contains functions to generate HTML components for the dashboard.
"""

import json

def generate_condition_threshold_comparison():
    """
    Generate HTML for the condition threshold comparison section.
    Returns the complete HTML string for the section.
    """
    # Define condition thresholds
    condition_thresholds = {
        "Healthy": {"hr_low": 60, "hr_high": 100, "bo_low": 95},
        "COPD": {"hr_low": 60, "hr_high": 100, "bo_low": 92},
        "Anxiety": {"hr_low": 60, "hr_high": 115, "bo_low": 95},
        "Heart Disease": {"hr_low": 60, "hr_high": 100, "bo_low": 95},
        "Athlete": {"hr_low": 50, "hr_high": 100, "bo_low": 95}
    }
    
    # Generate table rows without f-strings
    table_rows = ""
    for condition, thresholds in condition_thresholds.items():
        row = (
            "<tr>"
            f"<td>{condition}</td>"
            f"<td>{thresholds['hr_low']} bpm</td>"
            f"<td>{thresholds['hr_high']} bpm</td>"
            f"<td>{thresholds['bo_low']}%</td>"
            "</tr>"
        )
        table_rows += row
    
    # Create table without f-strings
    table = (
        "<div class=\"performance-table\">"
        "<table>"
        "<thead>"
        "<tr>"
        "<th>Health Condition</th>"
        "<th>Normal Heart Rate (min)</th>"
        "<th>Normal Heart Rate (max)</th>"
        "<th>Normal Blood Oxygen (min)</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + table_rows +
        "</tbody>"
        "</table>"
        "</div>"
    )
    
    # Header section
    header = (
        "<div class=\"threshold-comparison section\">"
        "<h2 id=\"threshold-comparison\">Risk Threshold Comparison by Condition</h2>"
        "<p>This table shows how normal ranges for vital signs differ across health conditions:</p>"
    )
    
    # Chart container
    chart_container = (
        "<div class=\"chart-container\">"
        "<h3>Risk Threshold Visualization</h3>"
        "<div style=\"display: flex; justify-content: center; margin-top: 20px;\">"
        "<canvas id=\"thresholdChart\" width=\"800\" height=\"400\"></canvas>"
        "</div>"
    )
    
    # JavaScript part 1
    js_part1 = (
        "<script>"
        "// Create the threshold visualization chart\n"
        "const ctx = document.getElementById('thresholdChart').getContext('2d');\n"
        "\n"
        "// Load the threshold data\n"
        "const thresholdData = "
    )
    
    # JavaScript data as JSON
    js_data = json.dumps(condition_thresholds)
    
    # JavaScript part 2 (avoiding any f-string issues)
    js_part2 = (
        ";\n"
        "\n"
        "// Prepare data for the chart\n"
        "const conditions = Object.keys(thresholdData);\n"
        "\n"
        "const hrLowData = [];\n"
        "const hrHighData = [];\n"
        "const boLowData = [];\n"
        "\n"
        "conditions.forEach(function(condition) {\n"
        "    hrLowData.push(thresholdData[condition].hr_low);\n"
        "    hrHighData.push(thresholdData[condition].hr_high);\n"
        "    boLowData.push(thresholdData[condition].bo_low);\n"
        "});\n"
        "\n"
        "const thresholdChart = new Chart(ctx, {\n"
        "    type: 'bar',\n"
        "    data: {\n"
        "        labels: conditions,\n"
        "        datasets: [\n"
        "            {\n"
        "                label: 'Min Heart Rate (bpm)',\n"
        "                data: hrLowData,\n"
        "                backgroundColor: 'rgba(54, 162, 235, 0.5)',\n"
        "                borderColor: 'rgba(54, 162, 235, 1)',\n"
        "                borderWidth: 1\n"
        "            },\n"
        "            {\n"
        "                label: 'Max Heart Rate (bpm)',\n"
        "                data: hrHighData,\n"
        "                backgroundColor: 'rgba(255, 99, 132, 0.5)',\n"
        "                borderColor: 'rgba(255, 99, 132, 1)',\n"
        "                borderWidth: 1\n"
        "            },\n"
        "            {\n"
        "                label: 'Min Blood Oxygen (%)',\n"
        "                data: boLowData,\n"
        "                backgroundColor: 'rgba(75, 192, 192, 0.5)',\n"
        "                borderColor: 'rgba(75, 192, 192, 1)',\n"
        "                borderWidth: 1\n"
        "            }\n"
        "        ]\n"
        "    },\n"
        "    options: {\n"
        "        responsive: true,\n"
        "        scales: {\n"
        "            y: {\n"
        "                beginAtZero: false,\n"
        "                min: 40,\n"
        "                max: 120,\n"
        "                title: {\n"
        "                    display: true,\n"
        "                    text: 'Value'\n"
        "                }\n"
        "            },\n"
        "            x: {\n"
        "                title: {\n"
        "                    display: true,\n"
        "                    text: 'Health Condition'\n"
        "                }\n"
        "            }\n"
        "        },\n"
        "        plugins: {\n"
        "            title: {\n"
        "                display: true,\n"
        "                text: 'Normal Vital Sign Ranges by Health Condition'\n"
        "            },\n"
        "            tooltip: {\n"
        "                callbacks: {\n"
        "                    label: function(context) {\n"
        "                        const label = context.dataset.label || '';\n"
        "                        const value = context.raw;\n"
        "                        return label + ': ' + value;\n"
        "                    }\n"
        "                }\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "});\n"
        "</script>"
        "</div>"
    )
    
    # Impact section
    impact_section = (
        "<div class=\"threshold-impact\">"
        "<h3>Impact on Risk Calculation</h3>"
        "<p>These adjusted thresholds affect how risk scores are calculated for patients with different health conditions:</p>"
        "<ul>"
        "<li><strong>COPD/Emphysema:</strong> Lower blood oxygen threshold (92% vs 95%) means that blood oxygen levels between 92-95% are considered normal for COPD patients but would indicate elevated risk for others.</li>"
        "<li><strong>Anxiety:</strong> Higher maximum heart rate threshold (115 bpm vs 100 bpm) accounts for the fact that anxiety patients often have higher baseline heart rates without indicating the same level of risk.</li>"
        "<li><strong>Athletes:</strong> Lower minimum heart rate threshold (50 bpm vs 60 bpm) recognizes that athletes typically have lower resting heart rates due to increased cardiovascular efficiency.</li>"
        "</ul>"
        "<p>The condition-aware models are trained to incorporate these threshold adjustments when calculating risk scores, leading to more personalized and accurate risk assessments.</p>"
        "</div>"
        "</div>"
    )
    
    # Combine all parts
    return header + table + chart_container + js_part1 + js_data + js_part2 + impact_section 