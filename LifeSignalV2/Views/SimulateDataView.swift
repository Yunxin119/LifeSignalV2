//
//  SimulateDataView.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/18/25.
//

import SwiftUI
import Combine

struct SimulateDataView: View {
    @Environment(\.presentationMode) var presentationMode
    @EnvironmentObject var authModel: UserAuthModel
    
    // Simulation states
    @State private var isSimulating = false
    @State private var simulationError: String?
    @State private var simulationResult: SimulationResponse?
    @State private var showSimulationOptions = false
    
    // Simulation parameters
    @State private var daysToSimulate = 30
    @State private var abnormalProbability = 0.1
    @State private var readingsPerDay = 3
    
    @State private var cancellables = Set<AnyCancellable>()
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Generate simulated health data for testing")
                    .font(.headline)
                    .padding(.top)
                
                // Simulation parameters
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Days to simulate:")
                        Spacer()
                        Stepper("\(daysToSimulate)", value: $daysToSimulate, in: 1...90)
                    }
                    
                    HStack {
                        Text("Anomaly probability:")
                        Spacer()
                        Text("\(Int(abnormalProbability * 100))%")
                        Stepper("", value: $abnormalProbability, in: 0.0...1.0, step: 0.05)
                            .labelsHidden()
                    }
                    
                    HStack {
                        Text("Readings per day:")
                        Spacer()
                        Stepper("\(readingsPerDay)", value: $readingsPerDay, in: 1...24)
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(8)
                
                if isSimulating {
                    loadingView(message: "Generating simulated data...")
                } else if let error = simulationError {
                    errorBanner(message: error)
                } else if let result = simulationResult {
                    simulationResultsView(result: result)
                }
                
                Button(action: simulateData) {
                    Text(simulationResult == nil ? "Generate Data" : "Generate More Data")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
                .disabled(isSimulating)
                .padding(.top, 8)
            }
            .padding()
        }
        .navigationTitle("Data Simulation")
        .navigationBarTitleDisplayMode(.inline)
    }
    
    private func simulationResultsView(result: SimulationResponse) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(result.message)
                .fontWeight(.medium)
            
            HStack {
                Text("Records Created:")
                Spacer()
                Text("\(result.recordsCreated)")
                    .fontWeight(.medium)
            }
            .padding(.vertical, 4)
            
            if let mlError = result.modelEvaluation.mlModelError,
               let hybridError = result.modelEvaluation.hybridModelError,
               let improvement = result.modelEvaluation.improvement {
                
                Divider()
                
                Text("Model Metrics After Simulation:")
                    .font(.headline)
                    .padding(.top, 4)
                
                HStack {
                    Text("ML Model Error:")
                    Spacer()
                    Text(String(format: "%.4f", mlError))
                        .fontWeight(.medium)
                }
                
                HStack {
                    Text("Hybrid Model Error:")
                    Spacer()
                    Text(String(format: "%.4f", hybridError))
                        .fontWeight(.medium)
                }
                
                HStack {
                    Text("Improvement:")
                    Spacer()
                    Text("\(String(format: "%.1f", improvement))%")
                        .fontWeight(.medium)
                        .foregroundColor(improvement > 0 ? .green : .red)
                }
            } else if let error = result.modelEvaluation.error {
                Text("Model evaluation error: \(error)")
                    .foregroundColor(.red)
                    .font(.footnote)
            }
            
            if !result.samples.isEmpty {
                Divider()
                
                Text("Sample Generated Data:")
                    .font(.headline)
                    .padding(.top, 4)
                
                VStack(spacing: 8) {
                    ForEach(result.samples.indices, id: \.self) { index in
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("HR: \(Int(result.samples[index].heartRate)) BPM, SpO₂: \(Int(result.samples[index].bloodOxygen))%")
                                    .font(.footnote)
                                
                                Text("Risk: \(String(format: "%.1f", result.samples[index].riskScore))")
                                    .font(.caption)
                                    .fontWeight(.medium)
                                
                                if let date = ISO8601DateFormatter().date(from: result.samples[index].timestamp) {
                                    Text(DateFormatter.localizedString(from: date, dateStyle: .short, timeStyle: .short))
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                            }
                            
                            Spacer()
                            
                            if result.samples[index].isAnomaly {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundColor(.orange)
                            }
                        }
                        .padding(8)
                        .background(Color(.systemGray6))
                        .cornerRadius(6)
                    }
                }
            }
        }
    }
    
    // MARK: - Shared Components
    
    private func loadingView(message: String) -> some View {
        HStack(spacing: 12) {
            ProgressView()
            Text(message)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
    
    private func errorBanner(message: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.orange)
            Text(message)
                .foregroundColor(.red)
            Spacer()
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
    
    // MARK: - API Calls
    
    private func simulateData() {
        guard let token = authModel.token else {
            simulationError = "Authentication token not found"
            return
        }
        
        isSimulating = true
        simulationError = nil
        
        APIService.simulateHealthData(
            token: token,
            days: daysToSimulate,
            abnormalProb: abnormalProbability,
            readingsPerDay: readingsPerDay
        )
        .receive(on: DispatchQueue.main)
        .sink { completion in
            isSimulating = false
            
            if case .failure(let error) = completion {
                simulationError = error.message
                print("Data simulation error: \(error)")
            }
        } receiveValue: { response in
            self.simulationResult = response
            print("Successfully created \(response.recordsCreated) simulated health records")
        }
        .store(in: &cancellables)
    }
}

struct SimulateDataView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            SimulateDataView()
                .environmentObject(UserAuthModel())
        }
    }
} 