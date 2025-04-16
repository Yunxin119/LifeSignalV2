//
//  HealthTrendsService.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/16/25.
//

import SwiftUI
import Combine
import Foundation

class HealthTrendsService: ObservableObject {
    @Published var healthTrends: HealthTrends?
    @Published var aiAnalysis: AITrendAnalysis?
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var rawResponse: Data?
    
    private var cancellables = Set<AnyCancellable>()
    
    // Get both trends and AI analysis in one call
    func fetchHealthAnalysis(token: String, days: Int = 30) {
        guard let url = URL(string: "\(Config.apiBaseURL)/api/health/trends/analyze?days=\(days)") else {
            self.errorMessage = "Invalid URL"
            return
        }
        
        isLoading = true
        errorMessage = nil
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .tryMap { data -> Data in
                // Store for debugging
                self.rawResponse = data
                
                // Print response for debugging
                if let jsonString = String(data: data, encoding: .utf8) {
                    print("Received health analysis response: \(jsonString)")
                }
                
                // Check for error responses
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let error = json["error"] as? String {
                    throw NSError(domain: "", code: 0, userInfo: [NSLocalizedDescriptionKey: error])
                }
                
                return data
            }
            .decode(type: AIAnalysisResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink { completion in
                self.isLoading = false
                
                if case .failure(let error) = completion {
                    self.errorMessage = "Failed to load health analysis: \(error.localizedDescription)"
                    print("Error details: \(error)")
                    
                    // Try alternative parsing if possible
                    if let rawData = self.rawResponse {
                        self.tryAlternativeParsing(data: rawData)
                    }
                }
            } receiveValue: { response in
                // Process the trends data
                if let trends = response.trends {
                    self.healthTrends = trends
                    print("Successfully loaded health trends with \(trends.dataPoints) data points")
                }
                
                // Process the AI analysis
                if let serverAnalysis = response.aiAnalysis {
                    self.aiAnalysis = AITrendAnalysis(from: serverAnalysis)
                    print("Successfully loaded AI analysis")
                }
            }
            .store(in: &cancellables)
    }
    
    // Only get trends data
    func fetchHealthTrends(token: String, days: Int = 30) {
        guard let url = URL(string: "\(Config.apiBaseURL)/api/health/trends?days=\(days)") else {
            self.errorMessage = "Invalid URL"
            return
        }
        
        isLoading = true
        errorMessage = nil  // Clear previous errors
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .tryMap { data -> Data in
                // For debugging
                self.rawResponse = data
                
                if let jsonString = String(data: data, encoding: .utf8) {
                    print("Received health trends response: \(jsonString)")
                }
                
                // Check for error responses
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let error = json["error"] as? String {
                    throw NSError(domain: "", code: 0, userInfo: [NSLocalizedDescriptionKey: error])
                }
                
                return data
            }
            .decode(type: HealthTrends.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink { completion in
                self.isLoading = false
                
                if case .failure(let error) = completion {
                    self.errorMessage = "Failed to load health trends: \(error.localizedDescription)"
                    print("Health trends error details: \(error)")
                }
            } receiveValue: { trends in
                self.healthTrends = trends
                print("Successfully loaded health trends data with \(trends.dataPoints) data points")
            }
            .store(in: &cancellables)
    }
    
    // Alternative parsing method if normal decoding fails
    private func tryAlternativeParsing(data: Data) {
        do {
            // First, try to see if it's a string response (unstructured AI analysis)
            if let stringResponse = String(data: data, encoding: .utf8) {
                // Check if this might be a plain text AI analysis (not JSON)
                if !stringResponse.starts(with: "{") && !stringResponse.starts(with: "[") {
                    // It's likely an unstructured text response from the AI model
                    let serverAnalysis = ServerAIAnalysis(fromUnstructuredText: stringResponse)
                    self.aiAnalysis = AITrendAnalysis(from: serverAnalysis)
                    self.isLoading = false
                    self.errorMessage = nil
                    print("Successfully parsed unstructured text response")
                    return
                }
            }
            
            // If not a plain text response, try to parse as JSON
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                // Parse trends data
                if let trendsDict = json["trends"] as? [String: Any] {
                    parseAndCreateTrends(from: trendsDict)
                }
                
                // Parse AI analysis
                if let aiDict = json["ai_analysis"] as? [String: Any] {
                    parseAndCreateAnalysis(from: aiDict)
                } else if let aiString = json["ai_analysis"] as? String {
                    // Handle case where ai_analysis is a string
                    let serverAnalysis = ServerAIAnalysis(fromUnstructuredText: aiString)
                    self.aiAnalysis = AITrendAnalysis(from: serverAnalysis)
                }
                
                self.isLoading = false
                self.errorMessage = nil
            }
        } catch {
            print("Alternative parsing failed: \(error)")
            self.isLoading = false
            self.errorMessage = "Failed to process server response. Please try again later."
        }
    }
    
    // Helper to parse trends data from dictionary
    private func parseAndCreateTrends(from dict: [String: Any]) {
        guard let hrDict = dict["heart_rate"] as? [String: Any],
              let boDict = dict["blood_oxygen"] as? [String: Any],
              let dataPoints = dict["data_points"] as? Int,
              let daysAnalyzed = dict["days_analyzed"] as? Int else {
            print("Missing required fields in trends data")
            return
        }
        
        // Create heart rate metric
        let hrMetric = HealthTrends.MetricTrend(
            mean: hrDict["mean"] as? Double ?? 0,
            std: hrDict["std"] as? Double ?? 0,
            min: hrDict["min"] as? Double ?? 0,
            max: hrDict["max"] as? Double ?? 0,
            trend: hrDict["trend"] as? String ?? "stable"
        )
        
        // Create blood oxygen metric
        let boMetric = HealthTrends.MetricTrend(
            mean: boDict["mean"] as? Double ?? 0,
            std: boDict["std"] as? Double ?? 0,
            min: boDict["min"] as? Double ?? 0,
            max: boDict["max"] as? Double ?? 0,
            trend: boDict["trend"] as? String ?? "stable"
        )
        
        // Create trends object
        self.healthTrends = HealthTrends(
            daysAnalyzed: daysAnalyzed,
            dataPoints: dataPoints,
            heartRate: hrMetric,
            bloodOxygen: boMetric
        )
        
        print("Successfully created trends data from dictionary")
    }
    
    // Helper to parse AI analysis from dictionary
    private func parseAndCreateAnalysis(from dict: [String: Any]) {
        let assessment = dict["assessment"] as? String ?? ""
        let healthTrajectory = dict["health_trajectory"] as? String ?? "stable"
        let trendSummary = dict["trend_summary"] as? String ?? ""
        let medicalConsultation = dict["medical_consultation"] as? String ?? ""
        
        // Get arrays
        let concerns = dict["potential_concerns"] as? [String] ?? []
        let recommendations = dict["recommendations"] as? [String] ?? []
        let lifestyle = dict["lifestyle_adjustments"] as? [String] ?? []
        
        // Create analysis object
        self.aiAnalysis = AITrendAnalysis(
            trendSummary: trendSummary,
            healthTrajectory: healthTrajectory,
            assessment: assessment,
            potentialConcerns: concerns,
            recommendations: recommendations,
            lifestyleAdjustments: lifestyle,
            medicalConsultation: medicalConsultation
        )
        
        print("Successfully created AI analysis from dictionary")
    }
    
    // Mock data for preview
    static var previewTrends: HealthTrends {
        HealthTrends(
            daysAnalyzed: 30,
            dataPoints: 28,
            heartRate: HealthTrends.MetricTrend(
                mean: 72.5,
                std: 5.2,
                min: 61.0,
                max: 88.0,
                trend: "stable"
            ),
            bloodOxygen: HealthTrends.MetricTrend(
                mean: 97.8,
                std: 1.1,
                min: 95.0,
                max: 99.5,
                trend: "stable"
            )
        )
    }
    
    static var previewAIAnalysis: AITrendAnalysis {
        AITrendAnalysis(
            trendSummary: "Your heart rate and blood oxygen levels have remained stable over the past month, indicating consistent cardiovascular function.",
            healthTrajectory: "stable",
            assessment: "Overall health patterns show normal variation within expected ranges for your age and activity level.",
            potentialConcerns: ["Slight decrease in blood oxygen during night hours", "Minor heart rate variability reduction"],
            recommendations: ["Continue regular monitoring", "Maintain current activity levels", "Consider light breathing exercises before bed"],
            lifestyleAdjustments: ["Increase outdoor activity by 10-15 minutes daily", "Practice deep breathing exercises"],
            medicalConsultation: "Routine check-up recommended within the next 3 months, no urgent concerns identified"
        )
    }
}
