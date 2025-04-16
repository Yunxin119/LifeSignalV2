//
//  TestManualHealthInputView.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/16/25.
//

import SwiftUI
import HealthKit
import Combine

struct ManualHealthInputView: View {
    @EnvironmentObject private var authModel: UserAuthModel
    @StateObject private var submissionService = HealthDataSubmissionService()
    @Environment(\.dismiss) private var dismiss
    
    // Health metrics
    @State private var heartRate: Double = 75
    @State private var bloodOxygen: Double = 98
    @State private var temperature: Double = 36.6
    @State private var activityLevel: String = "Resting"
    
    // Anomaly simulation
    @State private var simulateAnomaly = false
    @State private var selectedAnomaly: AnomalyType = .none
    
    enum AnomalyType: String, CaseIterable, Identifiable {
        case none = "None"
        case highHeartRate = "High Heart Rate"
        case lowHeartRate = "Low Heart Rate"
        case lowBloodOxygen = "Low Blood Oxygen"
        case fever = "Fever"
        case multipleAnomalies = "Multiple Anomalies"
        
        var id: String { self.rawValue }
    }
    
    // Activity levels
    let activityLevels = ["Resting", "Light Activity", "Moderate Activity", "Vigorous Activity", "Sleeping"]
    
    var body: some View {
        NavigationStack {
            Form {
                // Basic health metrics section
                Section(header: Text("HEALTH METRICS")) {
                    VStack {
                        HStack {
                            Text("Heart Rate")
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("\(Int(heartRate)) BPM")
                                .foregroundColor(anomalyColor(for: .heartRate))
                                .fontWeight(.medium)
                        }
                        
                        Slider(value: $heartRate, in: 40...160, step: 1)
                            .accentColor(.pink)
                            .onChange(of: heartRate) { _ in
                                if simulateAnomaly {
                                    selectedAnomaly = .none
                                }
                            }
                    }
                    
                    VStack {
                        HStack {
                            Text("Blood Oxygen")
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(String(format: "%.1f%%", bloodOxygen))
                                .foregroundColor(anomalyColor(for: .bloodOxygen))
                                .fontWeight(.medium)
                        }
                        
                        Slider(value: $bloodOxygen, in: 85...100, step: 0.5)
                            .accentColor(.blue)
                            .onChange(of: bloodOxygen) { _ in
                                if simulateAnomaly {
                                    selectedAnomaly = .none
                                }
                            }
                    }
                    
                    VStack {
                        HStack {
                            Text("Temperature")
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(String(format: "%.1f°C", temperature))
                                .foregroundColor(anomalyColor(for: .temperature))
                                .fontWeight(.medium)
                        }
                        
                        Slider(value: $temperature, in: 35...40, step: 0.1)
                            .accentColor(.orange)
                            .onChange(of: temperature) { _ in
                                if simulateAnomaly {
                                    selectedAnomaly = .none
                                }
                            }
                    }
                    
                    Picker("Activity Level", selection: $activityLevel) {
                        ForEach(activityLevels, id: \.self) { level in
                            Text(level).tag(level)
                        }
                    }
                }
                
                // Anomaly simulation section
                Section(header: Text("ANOMALY SIMULATION")) {
                    Toggle("Simulate Health Anomaly", isOn: $simulateAnomaly)
                    
                    if simulateAnomaly {
                        Picker("Anomaly Type", selection: $selectedAnomaly) {
                            ForEach(AnomalyType.allCases) { anomaly in
                                Text(anomaly.rawValue).tag(anomaly)
                            }
                        }
                        .onChange(of: selectedAnomaly) { newValue in
                            applyAnomalyPreset(newValue)
                        }
                    }
                }
                
                // Submit section
                Section {
                    Button(action: submitHealthData) {
                        if submissionService.isSubmitting {
                            HStack {
                                Spacer()
                                ProgressView()
                                Spacer()
                            }
                        } else {
                            HStack {
                                Spacer()
                                Text("Submit Health Data")
                                    .fontWeight(.semibold)
                                Spacer()
                            }
                        }
                    }
                    .disabled(submissionService.isSubmitting)
                    
                    if submissionService.submissionSuccess {
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                            Text("Health data submitted successfully!")
                                .foregroundColor(.green)
                        }
                    }
                    
                    if let error = submissionService.submissionError {
                        HStack {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.red)
                            Text(error)
                                .foregroundColor(.red)
                                .font(.caption)
                        }
                    }
                }
            }
            .navigationTitle("Manual Health Input")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Close") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    private func submitHealthData() {
        guard let token = authModel.token else {
            submissionService.submissionError = "Authentication token missing"
            return
        }
        
        let additionalMetrics: [String: Any] = [
            "temperature": temperature,
            "activity_level": activityLevel
        ]
        
        submissionService.submitHealthData(
            token: token,
            heartRate: heartRate,
            bloodOxygen: bloodOxygen,
            additionalMetrics: additionalMetrics
        )
    }
    
    private func applyAnomalyPreset(_ anomalyType: AnomalyType) {
        // Reset to normal values first
        if anomalyType == .none {
            heartRate = 75
            bloodOxygen = 98
            temperature = 36.6
            return
        }
        
        // Apply anomaly preset
        switch anomalyType {
        case .highHeartRate:
            heartRate = 130
            bloodOxygen = 97
            temperature = 36.8
        case .lowHeartRate:
            heartRate = 45
            bloodOxygen = 97
            temperature = 36.5
        case .lowBloodOxygen:
            heartRate = 78
            bloodOxygen = 91
            temperature = 36.7
        case .fever:
            heartRate = 95
            bloodOxygen = 96
            temperature = 38.5
        case .multipleAnomalies:
            heartRate = 135
            bloodOxygen = 90
            temperature = 38.8
        default:
            break
        }
    }
    
    // Helper function to determine if a value is anomalous
    private enum MetricType {
        case heartRate, bloodOxygen, temperature
    }
    
    private func anomalyColor(for metric: MetricType) -> Color {
        switch metric {
        case .heartRate:
            return (heartRate > 100 || heartRate < 60) ? .red : .primary
        case .bloodOxygen:
            return bloodOxygen < 95 ? .red : .primary
        case .temperature:
            return temperature > 37.5 ? .red : .primary
        }
    }
}

struct ManualHealthInputView_Previews: PreviewProvider {
    static var previews: some View {
        ManualHealthInputView()
            .environmentObject(UserAuthModel())
    }
}
