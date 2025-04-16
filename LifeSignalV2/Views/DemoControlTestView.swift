//
//  DemoControlTestView.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/16/25.
//

import SwiftUI
import Combine

class DemoTimerManager: ObservableObject {
    @Published var isGenerating = false
    @Published var dataStatus = ""
    @Published var frequency: Double = 1.0
    
    private var cancellables = Set<AnyCancellable>()
    private var timerPublisher: AnyCancellable?
    
    func startGenerating(generator: @escaping () -> Void) {
        stopGenerating()
        
        let interval = 60.0 / frequency
        
        timerPublisher = Timer.publish(every: interval, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                generator()
            }
        
        if let timerPublisher = timerPublisher {
            cancellables.insert(timerPublisher)
        }
        
        dataStatus = "Auto-generating \(Int(frequency)) readings per minute"
    }
    
    func stopGenerating() {
        timerPublisher?.cancel()
        timerPublisher = nil
        dataStatus = isGenerating ? "" : "Data generation stopped"
    }
    
    deinit {
        stopGenerating()
        cancellables.forEach { $0.cancel() }
        cancellables.removeAll()
    }
}

struct DemoControlTestView: View {
    @EnvironmentObject var authModel: UserAuthModel
    @StateObject private var timerManager = DemoTimerManager()
    @State private var demoDataStatus = ""
    
    var body: some View {
        Form {
            Section(header: Text("DEMO DATA GENERATION")) {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Generate mock health data for the demo")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Button(action: generateSingleHealthReading) {
                        Label("Generate Single Health Reading", systemImage: "waveform.path.ecg")
                    }
                    .disabled(timerManager.isGenerating)
                    
                    Divider()
                    
                    Text("Automatic Data Generation")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    VStack {
                        HStack {
                            Text("Frequency:")
                                .font(.caption)
                            Spacer()
                            Text("\(Int(timerManager.frequency)) reading(s) per minute")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Slider(value: $timerManager.frequency, in: 1...10, step: 1)
                    }
                    
                    Toggle("Auto-generate Demo Data", isOn: $timerManager.isGenerating)
                        .onChange(of: timerManager.isGenerating) { newValue in
                            if newValue {
                                timerManager.startGenerating(generator: generateSingleHealthReading)
                            } else {
                                timerManager.stopGenerating()
                            }
                        }
                    
                    if !timerManager.dataStatus.isEmpty {
                        Text(timerManager.dataStatus)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.top, 4)
                    }
                }
            }
            
            Section(header: Text("ANOMALY SIMULATION")) {
                Button(action: simulateHeartRateAnomaly) {
                    Label("Simulate High Heart Rate", systemImage: "heart.fill")
                        .foregroundColor(.red)
                }
                .disabled(timerManager.isGenerating)
                
                Button(action: simulateBloodOxygenAnomaly) {
                    Label("Simulate Low Blood Oxygen", systemImage: "lungs.fill")
                        .foregroundColor(.blue)
                }
                .disabled(timerManager.isGenerating)
                
                Button(action: simulateMultipleAnomalies) {
                    Label("Simulate Critical Health Event", systemImage: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                }
                .disabled(timerManager.isGenerating)
            }
            
            Section(header: Text("DATA MANAGEMENT")) {
                Button(action: clearDemoData) {
                    Label("Clear All Demo Data", systemImage: "trash")
                        .foregroundColor(.red)
                }
                .disabled(timerManager.isGenerating)
            }
        }
        .navigationTitle("Demo Control Panel")
        .onDisappear {
            if timerManager.isGenerating {
                timerManager.isGenerating = false
                timerManager.stopGenerating()
            }
        }
    }
    
    // Function to generate a single health reading
    private func generateSingleHealthReading() {
        guard let token = authModel.token else {
            timerManager.dataStatus = "Error: Not authenticated"
            return
        }
        
        // Generate random but realistic health data
        let heartRate = Double.random(in: 65...85)
        let bloodOxygen = Double.random(in: 96...99)
        let temperature = Double.random(in: 36.4...36.8)
        
        // Define additional metrics
        let additionalMetrics: [String: Any] = [
            "temperature": temperature,
            "activity_level": "Resting"
        ]
        
        // Create service and submit data
        let service = HealthDataSubmissionService()
        service.submitHealthData(
            token: token,
            heartRate: heartRate,
            bloodOxygen: bloodOxygen,
            additionalMetrics: additionalMetrics
        )
        
        timerManager.dataStatus = "Generated data: HR=\(Int(heartRate)), O₂=\(Int(bloodOxygen))%, Temp=\(String(format: "%.1f°C", temperature))"
    }
    
    // Functions to simulate anomalies
    private func simulateHeartRateAnomaly() {
        guard let token = authModel.token else {
            timerManager.dataStatus = "Error: Not authenticated"
            return
        }
        
        let heartRate = Double.random(in: 125...140)
        let bloodOxygen = Double.random(in: 96...98)
        let temperature = Double.random(in: 36.7...37.0)
        
        // Define additional metrics
        let additionalMetrics: [String: Any] = [
            "temperature": temperature,
            "activity_level": "Light Activity"
        ]
        
        // Create service and submit data
        let service = HealthDataSubmissionService()
        service.submitHealthData(
            token: token,
            heartRate: heartRate,
            bloodOxygen: bloodOxygen,
            additionalMetrics: additionalMetrics
        )
        
        timerManager.dataStatus = "Simulated high heart rate: HR=\(Int(heartRate)), O₂=\(Int(bloodOxygen))%, Temp=\(String(format: "%.1f°C", temperature))"
    }
    
    private func simulateBloodOxygenAnomaly() {
        guard let token = authModel.token else {
            timerManager.dataStatus = "Error: Not authenticated"
            return
        }
        
        let heartRate = Double.random(in: 80...95)
        let bloodOxygen = Double.random(in: 89...93)
        let temperature = Double.random(in: 36.5...36.9)
        
        // Define additional metrics
        let additionalMetrics: [String: Any] = [
            "temperature": temperature,
            "activity_level": "Resting"
        ]
        
        // Create service and submit data
        let service = HealthDataSubmissionService()
        service.submitHealthData(
            token: token,
            heartRate: heartRate,
            bloodOxygen: bloodOxygen,
            additionalMetrics: additionalMetrics
        )
        
        timerManager.dataStatus = "Simulated low blood oxygen: HR=\(Int(heartRate)), O₂=\(Int(bloodOxygen))%, Temp=\(String(format: "%.1f°C", temperature))"
    }
    
    private func simulateMultipleAnomalies() {
        guard let token = authModel.token else {
            timerManager.dataStatus = "Error: Not authenticated"
            return
        }
        
        let heartRate = Double.random(in: 130...150)
        let bloodOxygen = Double.random(in: 88...91)
        let temperature = Double.random(in: 38.2...39.0)
        
        // Define additional metrics
        let additionalMetrics: [String: Any] = [
            "temperature": temperature,
            "activity_level": "Resting"
        ]
        
        // Create service and submit data
        let service = HealthDataSubmissionService()
        service.submitHealthData(
            token: token,
            heartRate: heartRate,
            bloodOxygen: bloodOxygen,
            additionalMetrics: additionalMetrics
        )
        
        timerManager.dataStatus = "Simulated critical event: HR=\(Int(heartRate)), O₂=\(Int(bloodOxygen))%, Temp=\(String(format: "%.1f°C", temperature))"
    }
    
    private func clearDemoData() {
        // TODO: This would typically call an API endpoint to clear demo data
        // For this demo, we'll just show a success message
        timerManager.dataStatus = "Demo data cleared (simulated)"
    }
}
