//
//  MetricCard.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/17/25.
//
import SwiftUI

struct MetricCard: View {
    let title: String
    let value: String
    let unit: String
    let icon: String
    let color: Color
    var isAnomalous: Bool = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundColor(color)
                
                Text(title)
                    .font(.headline)
                
                Spacer()
                
                if isAnomalous {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.yellow)
                }
            }
            
            HStack(alignment: .firstTextBaseline) {
                Text(value)
                    .font(.system(size: 36, weight: .bold))
                    .foregroundColor(isAnomalous ? .red : .primary)
                
                Text(unit)
                    .font(.headline)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: isAnomalous ? color.opacity(0.3) : Color.black.opacity(0.05),
                radius: isAnomalous ? 10 : 5,
                x: 0,
                y: 2)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(isAnomalous ? color : Color.clear, lineWidth: isAnomalous ? 2 : 0)
        )
    }
}
