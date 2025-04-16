//
//  HealthService.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/16/25.
//

import SwiftUI
import Combine
import Foundation

class HealthService: ObservableObject {
    @Published var healthData: [HealthData] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var rawResponse: Data?
    @Published var retryCount = 0
    @Published var lastRefreshTime: Date?
    
    private var cancellables = Set<AnyCancellable>()
    
    func fetchHealthHistory(token: String) {
        guard let url = URL(string: "\(Config.apiBaseURL)/api/health/history") else {
            self.errorMessage = "Invalid URL"
            return
        }
        
        isLoading = true
        errorMessage = nil
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        URLSession.shared.dataTaskPublisher(for: request)
            .tryMap { data, response -> Data in
                // Store the raw response for debugging
                self.rawResponse = data
                
                // Print raw response for debugging
                if let jsonString = String(data: data, encoding: .utf8) {
                    print("Raw Health History Response: \(jsonString)")
                }
                
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw URLError(.badServerResponse)
                }
                
                if !(200...299).contains(httpResponse.statusCode) {
                    // Try to extract error message if present
                    if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let errorMsg = json["error"] as? String {
                        throw NSError(domain: "", code: httpResponse.statusCode, 
                                     userInfo: [NSLocalizedDescriptionKey: errorMsg])
                    }
                    throw URLError(.badServerResponse)
                }
                
                return data
            }
            .decode(type: HealthHistoryResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink { completion in
                self.isLoading = false
                self.lastRefreshTime = Date()
                
                if case .failure(let error) = completion {
                    self.errorMessage = "Health Service error: failed to load health data - \(error.localizedDescription)"
                    print("Health data fetch error: \(error)")
                    
                    // Try alternative parsing if we received data but had decoding issues
                    if let rawData = self.rawResponse {
                        self.tryAlternativeParsing(data: rawData)
                    }
                }
            } receiveValue: { response in
                self.healthData = response.healthData
                
                // Check if we need to detect anomalies client-side for older data
                self.detectAnomalies()
                
                // Reset retry count on success
                self.retryCount = 0
                print("Successfully loaded \(response.healthData.count) health data entries")
            }
            .store(in: &cancellables)
    }
    
    // Add a retry mechanism
    func retryFetchHealthHistory(token: String) {
        retryCount += 1
        print("Retrying health data fetch (attempt \(retryCount))")
        
        // Wait longer between each retry
        DispatchQueue.main.asyncAfter(deadline: .now() + Double(retryCount)) {
            self.fetchHealthHistory(token: token)
        }
    }
    
    // Try to parse the raw data in alternative ways if standard decoding fails
    private func tryAlternativeParsing(data: Data) {
        print("Attempting alternative parsing of health data response")
        
        do {
            // First approach: Try modified key decoding strategy
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            
            if let response = try? decoder.decode(HealthHistoryResponse.self, from: data) {
                print("Alternative parsing strategy 1 succeeded")
                self.healthData = response.healthData
                self.detectAnomalies()
                return
            }
            
            // Second approach: Parse as dictionary and manually create objects
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                print("Parsing raw JSON: \(json)")
                
                // Try parsing history array from the backend
                if let historyArray = json["history"] as? [[String: Any]] {
                    print("Found history array with \(historyArray.count) items")
                    var parsedHealthData: [HealthData] = []
                    
                    for item in historyArray {
                        if let id = item["_id"] as? String,
                           let heartRate = item["heart_rate"] as? Double,
                           let bloodOxygen = item["blood_oxygen"] as? Double {
                            
                            // Get the timestamp from created_at
                            let timestamp = item["created_at"] as? String ?? ISO8601DateFormatter().string(from: Date())
                            
                            // Try to get analysis_result if present
                            var isAnomaly = false
                            var riskScore = 0.0
                            var recommendations: [String] = []
                            
                            if let analysisResult = item["analysis_result"] as? [String: Any] {
                                isAnomaly = analysisResult["is_anomaly"] as? Bool ?? false
                                riskScore = analysisResult["risk_score"] as? Double ?? 0.0
                                
                                if let recs = analysisResult["recommendations"] as? [String] {
                                    recommendations = recs
                                } else if let rec = analysisResult["recommendations"] as? String {
                                    recommendations = [rec]
                                }
                            }
                            
                            // Create health data object
                            let healthData = HealthData(
                                id: id,
                                heartRate: heartRate,
                                bloodOxygen: bloodOxygen,
                                timestamp: timestamp,
                                isAnomaly: isAnomaly,
                                riskScore: riskScore,
                                recommendations: recommendations
                            )
                            parsedHealthData.append(healthData)
                            print("Manually parsed data: \(id), HR: \(heartRate), BO: \(bloodOxygen)")
                        }
                    }
                    
                    if !parsedHealthData.isEmpty {
                        print("Manual parsing succeeded with \(parsedHealthData.count) entries")
                        self.healthData = parsedHealthData
                        self.detectAnomalies()
                        return
                    }
                } else {
                    print("Could not find 'history' array in response")
                }
                
                // Original approach looking for health_data
                if let healthDataArray = json["health_data"] as? [[String: Any]] {
                    var parsedHealthData: [HealthData] = []
                    
                    for item in healthDataArray {
                        if let id = item["_id"] as? String,
                           let heartRate = item["heart_rate"] as? Double,
                           let bloodOxygen = item["blood_oxygen"] as? Double,
                           let timestamp = item["timestamp"] as? String {
                            
                            // Parse additional metrics if available
                            var additionalMetrics: [String: Double] = [:]
                            if let metrics = item["additional_metrics"] as? [String: Double] {
                                additionalMetrics = metrics
                            }
                            
                            // Handle different formats of recommendations
                            var recommendations: [String] = []
                            if let recs = item["recommendations"] as? [String] {
                                recommendations = recs
                            } else if let rec = item["recommendations"] as? String {
                                recommendations = [rec]
                            }
                            
                            let healthData = HealthData(
                                id: id,
                                heartRate: heartRate,
                                bloodOxygen: bloodOxygen,
                                timestamp: timestamp,
                                isAnomaly: item["is_anomaly"] as? Bool ?? false,
                                riskScore: item["risk_score"] as? Double ?? 0.0,
                                recommendations: recommendations,
                                aiAnalysis: item["ai_analysis"] as? String,
                                additionalMetrics: additionalMetrics
                            )
                            parsedHealthData.append(healthData)
                        }
                    }
                    
                    if !parsedHealthData.isEmpty {
                        print("Alternative parsing strategy 2 succeeded with \(parsedHealthData.count) entries")
                        self.healthData = parsedHealthData
                        self.detectAnomalies()
                        return
                    }
                }
            }
            
            print("All alternative parsing attempts failed")
        } catch {
            print("Alternative parsing error: \(error)")
        }
    }
    
    // Detect anomalies on the client side if needed
    private func detectAnomalies() {
        // Check if any health data needs anomaly detection
        var needsUpdate = false
        
        for (index, data) in healthData.enumerated() {
            if !data.isAnomaly {
                // Simple anomaly detection for heart rate
                let heartRateAnomalyLow = data.heartRate < 50
                let heartRateAnomalyHigh = data.heartRate > 100
                
                // Simple anomaly detection for blood oxygen
                let bloodOxygenAnomaly = data.bloodOxygen < 92
                
                if heartRateAnomalyLow || heartRateAnomalyHigh || bloodOxygenAnomaly {
                    // Create updated version with anomaly flag
                    var updatedData = data
                    updatedData.isAnomaly = true
                    
                    // Calculate a simple risk score
                    var riskScore = 0.0
                    
                    if heartRateAnomalyLow {
                        riskScore += (50 - data.heartRate) / 10
                    }
                    
                    if heartRateAnomalyHigh {
                        riskScore += (data.heartRate - 100) / 10
                    }
                    
                    if bloodOxygenAnomaly {
                        riskScore += (92 - data.bloodOxygen) * 2
                    }
                    
                    updatedData.riskScore = min(max(riskScore, 0), 10)
                    
                    // Add simple recommendations
                    if updatedData.recommendations.isEmpty {
                        var newRecommendations: [String] = []
                        
                        if heartRateAnomalyLow {
                            newRecommendations.append("Your heart rate is below normal range. Consider consulting a healthcare provider.")
                        }
                        
                        if heartRateAnomalyHigh {
                            newRecommendations.append("Your heart rate is elevated. Try to rest and stay hydrated.")
                        }
                        
                        if bloodOxygenAnomaly {
                            newRecommendations.append("Your blood oxygen level is low. If you're experiencing shortness of breath, seek medical attention.")
                        }
                        
                        updatedData.recommendations = newRecommendations
                    }
                    
                    // Update the health data
                    healthData[index] = updatedData
                    needsUpdate = true
                }
            }
        }
        
        if needsUpdate {
            // Trigger UI update
            objectWillChange.send()
        }
    }
    
    func createTestHealthData(token: String, heartRate: Double, bloodOxygen: Double, completion: @escaping (Bool) -> Void) {
        guard let url = URL(string: "\(Config.apiBaseURL)/api/health/analyze") else {
            self.errorMessage = "Invalid URL"
            completion(false)
            return
        }
        
        let body: [String: Any] = [
            "heart_rate": heartRate,
            "blood_oxygen": bloodOxygen
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } catch {
            self.errorMessage = "Error creating request: \(error.localizedDescription)"
            completion(false)
            return
        }
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .tryMap { data -> Bool in
                if let jsonString = String(data: data, encoding: .utf8) {
                    print("Create health data response: \(jsonString)")
                    
                    if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let healthDataId = json["health_data_id"] as? String {
                        return true
                    }
                }
                return false
            }
            .receive(on: DispatchQueue.main)
            .sink { completion in
                if case .failure(let error) = completion {
                    self.errorMessage = "Failed to create health data: \(error.localizedDescription)"
                    print("Error details: \(error)")
                }
            } receiveValue: { success in
                if success {
                    self.fetchHealthHistory(token: token)
                }
                completion(success)
            }
            .store(in: &cancellables)
    }
    
    // MARK: THIS IS THE MOCK DATA JUST FOR PREVIEW
    static var previewData: HealthData {
        let dateFormatter = ISO8601DateFormatter()
        return HealthData(
            id: "1",
            heartRate: 75,
            bloodOxygen: 98,
            timestamp: dateFormatter.string(from: Date()),
            isAnomaly: false,
            riskScore: 12,
            recommendations: ["Stay hydrated", "Continue regular monitoring"],
            aiAnalysis: "Your vital signs are within normal ranges. Keep up the good work with regular exercise and proper hydration."
        )
    }
    
    static var previewAnomalyData: HealthData {
        let dateFormatter = ISO8601DateFormatter()
        return HealthData(
            id: "2",
            heartRate: 120,
            bloodOxygen: 92,
            timestamp: dateFormatter.string(from: Date().addingTimeInterval(-3600)),
            isAnomaly: true,
            riskScore: 68,
            recommendations: ["Rest and avoid physical exertion", "Monitor vital signs closely", "Contact your healthcare provider if symptoms persist"],
            aiAnalysis: "Your heart rate is elevated and blood oxygen is slightly below normal. This could be due to physical exertion, stress, or an underlying condition."
        )
    }
    
    static var previewHistory: [HealthData] {
        let dateFormatter = ISO8601DateFormatter()
        return [
            previewData,
            previewAnomalyData,
            HealthData(
                id: "3",
                heartRate: 68,
                bloodOxygen: 97,
                timestamp: dateFormatter.string(from: Date().addingTimeInterval(-7200)),
                isAnomaly: false,
                riskScore: 8,
                recommendations: ["Maintain healthy lifestyle"]
            ),
            HealthData(
                id: "4",
                heartRate: 72,
                bloodOxygen: 99,
                timestamp: dateFormatter.string(from: Date().addingTimeInterval(-10800)),
                isAnomaly: false,
                riskScore: 5,
                recommendations: ["Continue normal activities"]
            )
        ]
    }
    
    // MARK: - Computed Properties
    
    // Provide access to the latest health data entry
    var latestHealthData: HealthData? {
        return healthData.first
    }
    
    // Check if any health data entries have anomalies
    var anomalyDetected: Bool {
        return healthData.contains(where: { $0.isAnomaly })
    }
    
    // Alias for healthData for backward compatibility
    var healthHistory: [HealthData] {
        return healthData
    }
    
    // Convert rawResponse Data to String for UI display
    var rawResponseData: Data? {
        return rawResponse
    }
}
