//
//  HealthConditionsSelectionView.swift
//  LifeSignalV2
//

import SwiftUI

struct HealthConditionsSelectionView: View {
    let healthConditions: [String]
    @Binding var selectedConditions: [String]
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationStack {
            List {
                ForEach(healthConditions, id: \.self) { condition in
                    Button(action: {
                        if condition == "None" {
                            // If "None" is selected, clear all selections
                            if !selectedConditions.contains("None") {
                                selectedConditions = ["None"]
                            } else {
                                selectedConditions.removeAll()
                            }
                        } else {
                            // If other condition is selected, remove "None" if present
                            if selectedConditions.contains("None") {
                                selectedConditions.removeAll { $0 == "None" }
                            }
                            
                            // Toggle selection
                            if selectedConditions.contains(condition) {
                                selectedConditions.removeAll { $0 == condition }
                            } else {
                                selectedConditions.append(condition)
                            }
                        }
                    }) {
                        HStack {
                            Text(condition)
                            Spacer()
                            if selectedConditions.contains(condition) {
                                Image(systemName: "checkmark")
                                    .foregroundColor(.blue)
                            }
                        }
                    }
                    .foregroundColor(.primary)
                }
            }
            .navigationTitle("Health Conditions")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
} 