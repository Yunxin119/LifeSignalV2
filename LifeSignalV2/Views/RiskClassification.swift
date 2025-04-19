import SwiftUI

struct RiskClassificationView: View {
    let riskClass: Int
    let riskCategory: String
    let riskProbabilities: [String: Double]?
    
    private var riskColor: Color {
        switch riskClass {
        case 0: return .green  // Low Risk
        case 1: return .orange // Medium Risk
        case 2: return .red    // High Risk
        default: return .gray  // Unknown
        }
    }
    
    private var riskIcon: String {
        switch riskClass {
        case 0: return "checkmark.circle.fill"  // Low Risk
        case 1: return "exclamationmark.circle.fill" // Medium Risk
        case 2: return "exclamationmark.triangle.fill" // High Risk
        default: return "questionmark.circle.fill" // Unknown
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Risk Assessment")
                    .font(.headline)
                
                Spacer()
                
                HStack {
                    Image(systemName: riskIcon)
                        .foregroundColor(riskColor)
                    Text(riskCategory)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(riskColor)
                }
            }
            
            // Risk visualization
            if let probabilities = riskProbabilities {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Risk Probability")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    HStack(spacing: 3) {
                        // Low risk bar
                        RiskProbabilityBar(
                            value: probabilities["low"] ?? 0,
                            color: .green,
                            label: "Low"
                        )
                        
                        // Medium risk bar
                        RiskProbabilityBar(
                            value: probabilities["medium"] ?? 0,
                            color: .orange,
                            label: "Medium"
                        )
                        
                        // High risk bar
                        RiskProbabilityBar(
                            value: probabilities["high"] ?? 0,
                            color: .red,
                            label: "High"
                        )
                    }
                }
            } else {
                // Fallback visualization based on risk class
                ZStack(alignment: .leading) {
                    // Background track
                    Rectangle()
                        .frame(height: 8)
                        .cornerRadius(4)
                        .foregroundColor(Color(.systemGray5))
                    
                    // Indicator - positioned based on risk class
                    HStack(spacing: 0) {
                        Rectangle()
                            .frame(width: 8, height: 8)
                            .cornerRadius(4)
                            .foregroundColor(riskColor)
                            .offset(x: CGFloat(riskClass) * UIScreen.main.bounds.width * 0.25)
                    }
                }
                
                HStack {
                    Text("Low")
                        .font(.caption2)
                        .foregroundColor(.green)
                    
                    Spacer()
                    
                    Text("Medium")
                        .font(.caption2)
                        .foregroundColor(.orange)
                    
                    Spacer()
                    
                    Text("High")
                        .font(.caption2)
                        .foregroundColor(.red)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
    }
}