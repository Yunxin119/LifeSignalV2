//
//  HealthDataSubmissionService.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/16/25.
//

import SwiftUI
import Combine

class HealthDataSubmissionService: ObservableObject {
    @Published var isSubmitting = false
    @Published var submissionError: String?
    @Published var submissionSuccess = false
    @Published var currentProgress: Int = 0
    @Published var totalToSubmit: Int = 0
    
    private var cancellables = Set<AnyCancellable>()
    
    func submitHealthData(token: String, heartRate: Double, bloodOxygen: Double, additionalMetrics: [String: Any]? = nil) {
        guard let url = URL(string: "\(Config.apiBaseURL)/api/health/analyze") else {
            self.submissionError = "Invalid URL"
            return
        }
        
        isSubmitting = true
        submissionSuccess = false
        submissionError = nil
        
        // Prepare request body
        var bodyData: [String: Any] = [
            "heart_rate": heartRate,
            "blood_oxygen": bloodOxygen
        ]
        
        // Add any additional metrics if provided
        if let additionalMetrics = additionalMetrics {
            for (key, value) in additionalMetrics {
                bodyData[key] = value
            }
        }
        
        // Create request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Serialize request body
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: bodyData)
        } catch {
            isSubmitting = false
            submissionError = "Failed to serialize request: \(error.localizedDescription)"
            return
        }
        
        // Make API call
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .receive(on: DispatchQueue.main)
            .sink { completion in
                self.isSubmitting = false
                
                if case .failure(let error) = completion {
                    self.submissionError = "Failed to submit health data: \(error.localizedDescription)"
                }
            } receiveValue: { data in
                // Check if response contains error
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let error = json["error"] as? String {
                    self.submissionError = "Server error: \(error)"
                } else {
                    self.submissionSuccess = true
                }
            }
            .store(in: &cancellables)
    }
    
    // New function to create multiple test data points to ensure we have enough for analysis
    func createMultipleTestDataPoints(token: String, count: Int = 10, completion: @escaping (Bool) -> Void) {
        guard count > 0 else {
            completion(true)
            return
        }
        
        isSubmitting = true
        submissionSuccess = false
        submissionError = nil
        currentProgress = 0
        totalToSubmit = count
        
        // Create test data points with timestamps spread over the past 'count' days
        // This ensures better distribution of data for trend analysis
        createNextTestDataPoint(token: token, remainingCount: count, daysAgo: count) { success in
            self.isSubmitting = false
            completion(success)
        }
    }
    
    private func createNextTestDataPoint(token: String, remainingCount: Int, daysAgo: Int, completion: @escaping (Bool) -> Void) {
        guard remainingCount > 0 else {
            completion(true)
            return
        }
        
        // Calculate timestamp - distribute one data point per day for the past 'daysAgo' days
        // This creates a better spread of data for trend analysis
        let dayOffset = remainingCount - 1 // Adjust to spread evenly across days
        let calendar = Calendar.current
        let dateToUse = calendar.date(byAdding: .day, value: -dayOffset, to: Date()) ?? Date()
        
        // Add some random hour component to make it more realistic
        let randomHour = Int.random(in: 8...20) // Between 8 AM and 8 PM
        let randomMinute = Int.random(in: 0...59)
        var components = calendar.dateComponents([.year, .month, .day], from: dateToUse)
        components.hour = randomHour
        components.minute = randomMinute
        let timestamp = calendar.date(from: components) ?? dateToUse
        
        print("Creating test data point for day: -\(dayOffset), time: \(timestamp)")
        
        // Generate realistic data with some variation based on the day
        // Create some slight trend for more interesting analysis
        let trendFactor = Double(dayOffset) / Double(daysAgo) // 0 to 1
        let heartRate = Double.random(in: 65 + (trendFactor * 5)...(85 - (trendFactor * 3)))
        let bloodOxygen = Double.random(in: 95.5 + (trendFactor * 0.5)...(99.5 - (trendFactor * 0.2)))
        
        // Create the URL
        guard let url = URL(string: "\(Config.apiBaseURL)/api/health/analyze") else {
            self.submissionError = "Invalid URL"
            completion(false)
            return
        }
        
        // Format timestamp as ISO8601 string
        let isoFormatter = ISO8601DateFormatter()
        let timestampString = isoFormatter.string(from: timestamp)
        
        // Prepare request body with timestamp
        var bodyData: [String: Any] = [
            "heart_rate": heartRate,
            "blood_oxygen": bloodOxygen,
            "timestamp": timestampString // Add explicit timestamp to the request
        ]
        
        // Create request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Serialize request body
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: bodyData)
        } catch {
            self.submissionError = "Failed to serialize request: \(error.localizedDescription)"
            completion(false)
            return
        }
        
        // Make API call
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .receive(on: DispatchQueue.main)
            .sink { completion in
                if case .failure(let error) = completion {
                    self.submissionError = "Failed to submit health data: \(error.localizedDescription)"
                }
            } receiveValue: { data in
                // Update progress
                self.currentProgress += 1
                
                // Check if response contains error
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let error = json["error"] as? String {
                    self.submissionError = "Server error: \(error)"
                    completion(false)
                } else {
                    // Continue with next data point after a short delay
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        self.createNextTestDataPoint(
                            token: token,
                            remainingCount: remainingCount - 1,
                            daysAgo: daysAgo,
                            completion: completion
                        )
                    }
                }
            }
            .store(in: &cancellables)
    }
}
