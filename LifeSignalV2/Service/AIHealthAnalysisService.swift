//
//  AIHealthAnalysisService.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/16/25.
//

import SwiftUI
import Combine
import Foundation

class AIHealthAnalysisService: ObservableObject {
    @Published var analysis: AITrendAnalysis?
    @Published var trends: HealthTrends?
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var hasNoData = false
    @Published var rawResponse: Data?
    
    private var cancellables = Set<AnyCancellable>()
    
    func fetchAIAnalysis(token: String, days: Int = 30) {
        guard let url = URL(string: "\(Config.apiBaseURL)/api/health/trends/analyze?days=\(days)") else {
            self.errorMessage = "Invalid URL"
            return
        }
        
        isLoading = true
        errorMessage = nil
        hasNoData = false
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        URLSession.shared.dataTaskPublisher(for: request)
            .tryMap { data, response -> Data in
                // Store raw response for debugging
                self.rawResponse = data
                
                // Print received JSON for debugging
                if let jsonString = String(data: data, encoding: .utf8) {
                    print("Received JSON from trends/analyze: \(jsonString)")
                    
                    // Check for error responses
                    if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        if let error = json["error"] as? String {
                            throw NSError(domain: "", code: 0, userInfo: [NSLocalizedDescriptionKey: error])
                        }
                        
                        if let message = json["message"] as? String, message.contains("No health data") {
                            DispatchQueue.main.async {
                                self.hasNoData = true
                            }
                            throw NSError(domain: "", code: 0, userInfo: [NSLocalizedDescriptionKey: message])
                        }
                    }
                }
                
                // Validate HTTP response
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw URLError(.badServerResponse)
                }
                
                if !(200...299).contains(httpResponse.statusCode) {
                    throw URLError(.badServerResponse)
                }
                
                return data
            }
            .decode(type: AIAnalysisResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink { completion in
                self.isLoading = false
                
                if case .failure(let error) = completion {
                    if self.hasNoData {
                        self.errorMessage = "Insufficient health data for AI analysis. Please add more health data points first."
                    } else {
                        self.errorMessage = "Failed to load AI analysis: \(error.localizedDescription)"
                        print("Error details: \(error)")
                        
                        // Try alternative parsing if needed
                        if let rawData = self.rawResponse {
                            self.tryAlternativeDecoding(data: rawData)
                        }
                    }
                }
            } receiveValue: { response in
                self.processResponse(response)
            }
            .store(in: &cancellables)
    }
    
    private func processResponse(_ response: AIAnalysisResponse) {
        // Process trends data
        if let trendsData = response.trends {
            self.trends = trendsData
            print("Valid trends data received with \(trendsData.dataPoints) data points")
        }
        
        // Process AI analysis data
        if let serverAnalysis = response.aiAnalysis {
            // Convert server analysis model to UI model
            self.analysis = AITrendAnalysis(from: serverAnalysis)
            print("Successfully processed AI analysis")
        } else {
            // If we have trends but no AI analysis, create a default analysis
            if let trendsData = self.trends {
                self.createDefaultAnalysisFromTrends(trendsData)
            } else {
                self.errorMessage = "No valid AI analysis found in response"
                self.hasNoData = true
            }
        }
    }
    
    // Create a basic analysis from trends data when the AI analysis is missing
    private func createDefaultAnalysisFromTrends(_ trends: HealthTrends) {
        // Create a simple analysis based on trends data
        let hrTrend = trends.heartRate.trend
        let boTrend = trends.bloodOxygen.trend
        
        // Determine health trajectory based on trends
        let healthTrajectory: String
        if hrTrend == "increasing" && trends.heartRate.mean > 85 {
            healthTrajectory = "concerning"
        } else if boTrend == "decreasing" && trends.bloodOxygen.mean < 96 {
            healthTrajectory = "concerning"
        } else {
            healthTrajectory = "stable"
        }
        
        // Create a simple trend summary
        let trendSummary = "Over the past \(trends.daysAnalyzed) days, analyzed from \(trends.dataPoints) data points. " +
                         "Your average heart rate is \(String(format: "%.1f", trends.heartRate.mean)) BPM (\(hrTrend)) and " +
                         "blood oxygen is \(String(format: "%.1f", trends.bloodOxygen.mean))% (\(boTrend))."
        
        // Create assessment
        let assessment = "Based on your data, your heart rate is \(hrTrend) and blood oxygen is \(boTrend). " +
                       "Heart rate average is \(trends.heartRate.mean >= 60 && trends.heartRate.mean <= 100 ? "within normal range" : "outside normal range"). " +
                       "Blood oxygen average is \(trends.bloodOxygen.mean >= 95 ? "healthy" : "below optimal levels")."
        
        // Generate appropriate recommendations and concerns
        var recommendations = [String]()
        var concerns = [String]()
        var lifestyleAdjustments = [String]()
        
        // Heart rate recommendations
        if trends.heartRate.mean > 90 {
            concerns.append("Heart rate slightly elevated above ideal resting range")
            recommendations.append("Monitor your heart rate more frequently")
            lifestyleAdjustments.append("Practice relaxation techniques to lower resting heart rate")
        } else if trends.heartRate.mean < 60 {
            concerns.append("Heart rate below typical resting range")
            recommendations.append("Monitor for symptoms of fatigue or dizziness")
            lifestyleAdjustments.append("Consider gradually increasing physical activity")
        } else {
            recommendations.append("Continue your current heart health habits")
            lifestyleAdjustments.append("Maintain regular physical activity")
        }
        
        // Blood oxygen recommendations
        if trends.bloodOxygen.mean < 95 {
            concerns.append("Blood oxygen below ideal range")
            recommendations.append("Monitor your blood oxygen levels more closely")
            lifestyleAdjustments.append("Practice deep breathing exercises daily")
            recommendations.append("Consider consulting with a healthcare provider")
        } else {
            recommendations.append("Continue monitoring your blood oxygen levels regularly")
            lifestyleAdjustments.append("Maintain good respiratory health practices")
        }
        
        // Add general recommendations
        recommendations.append("Continue collecting health data regularly for better trend analysis")
        lifestyleAdjustments.append("Stay hydrated throughout the day")
        lifestyleAdjustments.append("Ensure you're getting adequate sleep (7-9 hours)")
        
        // Determine medical consultation recommendation
        let medicalConsultation = healthTrajectory == "concerning" ? 
            "Consider consulting with a healthcare provider to discuss your health trends." :
            "Routine health check-ups are recommended, but no urgent medical attention appears necessary based on your current data."
        
        // Create analysis object
        self.analysis = AITrendAnalysis(
            trendSummary: trendSummary,
            healthTrajectory: healthTrajectory,
            assessment: assessment,
            potentialConcerns: concerns,
            recommendations: recommendations,
            lifestyleAdjustments: lifestyleAdjustments,
            medicalConsultation: medicalConsultation
        )
        
        print("Created default AI analysis from trends data")
    }
    
    // Manual parsing fallback
    private func tryAlternativeDecoding(data: Data) {
        do {
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                // Parse trends data first
                if let trendsDict = json["trends"] as? [String: Any] {
                    // Create HealthTrends object
                    self.parseAndCreateTrends(from: trendsDict)
                }
                
                // Parse AI analysis data
                if let aiDict = json["ai_analysis"] as? [String: Any] {
                    self.parseAndCreateAnalysis(from: aiDict)
                }
            }
        } catch {
            print("Alternative parsing failed: \(error)")
        }
    }
    
    // Helper method to parse and create trends from dictionary
    private func parseAndCreateTrends(from dict: [String: Any]) {
        guard let hrDict = dict["heart_rate"] as? [String: Any],
              let boDict = dict["blood_oxygen"] as? [String: Any],
              let dataPoints = dict["data_points"] as? Int,
              let daysAnalyzed = dict["days_analyzed"] as? Int else {
            print("Missing required fields in trends data")
            return
        }
        
        // Extract heart rate data
        let hrMean = hrDict["mean"] as? Double ?? 0
        let hrStd = hrDict["std"] as? Double ?? 0
        let hrMin = hrDict["min"] as? Double ?? 0
        let hrMax = hrDict["max"] as? Double ?? 0
        let hrTrend = hrDict["trend"] as? String ?? "stable"
        
        // Extract blood oxygen data
        let boMean = boDict["mean"] as? Double ?? 0
        let boStd = boDict["std"] as? Double ?? 0
        let boMin = boDict["min"] as? Double ?? 0
        let boMax = boDict["max"] as? Double ?? 0
        let boTrend = boDict["trend"] as? String ?? "stable"
        
        // Create HealthTrends object
        let heartRateMetric = HealthTrends.MetricTrend(mean: hrMean, std: hrStd, min: hrMin, max: hrMax, trend: hrTrend)
        let bloodOxygenMetric = HealthTrends.MetricTrend(mean: boMean, std: boStd, min: boMin, max: boMax, trend: boTrend)
        
        self.trends = HealthTrends(
            daysAnalyzed: daysAnalyzed,
            dataPoints: dataPoints,
            heartRate: heartRateMetric,
            bloodOxygen: bloodOxygenMetric
        )
        
        print("Successfully created HealthTrends from dictionary")
    }
    
    // Helper method to parse and create AI analysis from dictionary
    private func parseAndCreateAnalysis(from dict: [String: Any]) {
        let assessment = dict["assessment"] as? String ?? "No assessment available"
        let trajectory = dict["health_trajectory"] as? String ?? "unknown"
        let trendSummary = dict["trend_summary"] as? String ?? "No trend summary available"
        let medicalConsultation = dict["medical_consultation"] as? String ?? "No medical advice available"
        
        // Get arrays
        let concerns = dict["potential_concerns"] as? [String] ?? []
        let recommendations = dict["recommendations"] as? [String] ?? []
        let lifestyle = dict["lifestyle_adjustments"] as? [String] ?? []
        
        // Create AITrendAnalysis
        self.analysis = AITrendAnalysis(
            trendSummary: trendSummary,
            healthTrajectory: trajectory,
            assessment: assessment,
            potentialConcerns: concerns,
            recommendations: recommendations,
            lifestyleAdjustments: lifestyle,
            medicalConsultation: medicalConsultation
        )
        
        print("Successfully created AITrendAnalysis from dictionary")
    }
    
    // MARK: Sample data for preview
    static var previewAnalysis: AITrendAnalysis {
        AITrendAnalysis(
            trendSummary: "Over the past 30 days, analyzed from 240 data points. Your average heart rate is 72 BPM (stable) and blood oxygen is 98.0% (stable).",
            healthTrajectory: "normal",
            assessment: "Your vital signs are within normal ranges with good stability over time.",
            potentialConcerns: ["Minor fluctuations in heart rate during nighttime", "Slightly reduced blood oxygen when sleeping"],
            recommendations: [
                "Continue your regular exercise routine",
                "Maintain good hydration habits",
                "Consider incorporating 5-10 minutes of deep breathing exercises daily",
                "Ensure your sleeping environment has proper ventilation"
            ],
            lifestyleAdjustments: [
                "Stay hydrated throughout the day",
                "Ensure you're getting adequate sleep"
            ],
            medicalConsultation: "Routine health check-ups are recommended, but no urgent medical attention appears necessary based on your current data."
        )
    }
}
