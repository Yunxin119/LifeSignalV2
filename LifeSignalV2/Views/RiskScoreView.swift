//
//  RiskScoreView.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/17/25.
//

import SwiftUI
import Combine

struct RiskScoreView: View {
    let score: Double
    
    private var riskLevel: String {
        switch score {
        case 0..<20: return "Low"
        case 20..<40: return "Moderate"
        case 40..<60: return "Elevated"
        case 60..<80: return "High"
        default: return "Critical"
        }
    }
    
    private var riskColor: Color {
        switch score {
        case 0..<20: return .green
        case 20..<40: return .yellow
        case 40..<60: return .orange
        case 60..<80: return .red
        default: return .purple
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Risk Assessment")
                    .font(.headline)
                
                Spacer()
                
                Text(riskLevel)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(riskColor)
            }
            
            // Risk gauge
            ZStack(alignment: .leading) {
                // Background track
                Rectangle()
                    .frame(height: 8)
                    .cornerRadius(4)
                    .foregroundColor(Color(.systemGray5))
                
                // Indicator
                Rectangle()
                    .frame(width: CGFloat(score) / 100 * UIScreen.main.bounds.width * 0.8, height: 8)
                    .cornerRadius(4)
                    .foregroundColor(riskColor)
            }
            
            HStack {
                Text("0%")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text("100%")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
    }
}
