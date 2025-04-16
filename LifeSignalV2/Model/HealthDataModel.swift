//
//  HealthDataModel.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/16/25.
//

import Foundation

struct HealthData: Identifiable, Decodable, Equatable {
    let id: String
    let heartRate: Double
    let bloodOxygen: Double
    let timestamp: String  // Using string initially to handle different timestamp formats
    var isAnomaly: Bool
    var riskScore: Double
    var recommendations: [String]
    var aiAnalysis: String?
    var anomalyScore: Double?
    var trendAnalysis: [String: String]?
    var additionalMetrics: [String: Double]?
    
    enum CodingKeys: String, CodingKey {
        case id = "_id"
        case heartRate = "heart_rate"
        case bloodOxygen = "blood_oxygen"
        case timestamp
        case createdAt = "created_at"
        case additionalMetrics = "additional_metrics"
        case aiAnalysis = "ai_analysis"
        case recommendations
        case isAnomaly = "is_anomaly"
        case riskScore = "risk_score"
        case anomalyScore = "anomaly_score"
        case trendAnalysis = "trend_analysis"
        case analysisResult = "analysis_result"
    }
    
    enum AdditionalMetricsKeys: String, CodingKey {
        case analysisResult = "analysis_result"
    }
    
    enum AnalysisResultKeys: String, CodingKey {
        case isAnomaly = "is_anomaly"
        case riskScore = "risk_score"
        case recommendations
        case anomalyScore = "anomaly_score"
        case trendAnalysis = "trend_analysis"
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        heartRate = try container.decode(Double.self, forKey: .heartRate)
        bloodOxygen = try container.decode(Double.self, forKey: .bloodOxygen)
        
        // Handle the timestamp - try multiple field names
        if let timestampValue = try? container.decode(String.self, forKey: .timestamp) {
            timestamp = timestampValue
        } else if let createdAtValue = try? container.decode(String.self, forKey: .createdAt) {
            timestamp = createdAtValue
        } else {
            print("Warning: Could not find timestamp or created_at in response")
            timestamp = ISO8601DateFormatter().string(from: Date())
        }
        
        // Try direct access to analysis_result at top level
        if let analysisResultContainer = try? container.nestedContainer(keyedBy: AnalysisResultKeys.self, forKey: .analysisResult) {
            isAnomaly = (try? analysisResultContainer.decode(Bool.self, forKey: .isAnomaly)) ?? false
            riskScore = (try? analysisResultContainer.decode(Double.self, forKey: .riskScore)) ?? 0.0
            
            if let recArray = try? analysisResultContainer.decode([String].self, forKey: .recommendations) {
                recommendations = recArray
            } else if let singleRec = try? analysisResultContainer.decode(String.self, forKey: .recommendations) {
                recommendations = [singleRec]
            }
        } else if let topLevelIsAnomaly = try? container.decode(Bool.self, forKey: .isAnomaly) {
            isAnomaly = topLevelIsAnomaly
        } else {
            do {
                let additionalMetrics = try container.nestedContainer(keyedBy: AdditionalMetricsKeys.self, forKey: .additionalMetrics)
                let analysisResult = try additionalMetrics.nestedContainer(keyedBy: AnalysisResultKeys.self, forKey: .analysisResult)
                isAnomaly = try analysisResult.decode(Bool.self, forKey: .isAnomaly)
            } catch {
                print("Warning: Could not find isAnomaly in expected locations: \(error)")
                isAnomaly = false
            }
        }
        
        // Recommendation logic - this might not be in the response at all,
        // so initialize it here if we haven't already
        if let topLevelRecommendations = try? container.decode([String].self, forKey: .recommendations) {
            recommendations = topLevelRecommendations
        } else if let singleRec = try? container.decode(String.self, forKey: .recommendations) {
            recommendations = [singleRec]
        } else {
            // If not set in analysis_result block above and not present at top level
            do {
                let additionalMetrics = try container.nestedContainer(keyedBy: AdditionalMetricsKeys.self, forKey: .additionalMetrics)
                let analysisResult = try additionalMetrics.nestedContainer(keyedBy: AnalysisResultKeys.self, forKey: .analysisResult)
                if let recArray = try? analysisResult.decode([String].self, forKey: .recommendations) {
                    recommendations = recArray
                } else if let singleRec = try? analysisResult.decode(String.self, forKey: .recommendations) {
                    recommendations = [singleRec]
                } else {
                    recommendations = []
                }
            } catch {
                print("Warning: Could not find recommendations in any expected location")
                recommendations = []
            }
        }
        
        if let topLevelRiskScore = try? container.decode(Double.self, forKey: .riskScore) {
            riskScore = topLevelRiskScore
        } else {
            do {
                let additionalMetrics = try container.nestedContainer(keyedBy: AdditionalMetricsKeys.self, forKey: .additionalMetrics)
                let analysisResult = try additionalMetrics.nestedContainer(keyedBy: AnalysisResultKeys.self, forKey: .analysisResult)
                riskScore = try analysisResult.decode(Double.self, forKey: .riskScore)
            } catch {
                print("Warning: Could not find riskScore in expected locations: \(error)")
                riskScore = 0
            }
        }
        
        // Optional fields
        if let topLevelAnomalyScore = try? container.decode(Double.self, forKey: .anomalyScore) {
            anomalyScore = topLevelAnomalyScore
        } else {
            anomalyScore = try? container.nestedContainer(keyedBy: AdditionalMetricsKeys.self, forKey: .additionalMetrics)
                .nestedContainer(keyedBy: AnalysisResultKeys.self, forKey: .analysisResult)
                .decode(Double.self, forKey: .anomalyScore)
        }
        
        // Try to decode AI analysis if present
        if let aiResponse = try? container.decode([String: String].self, forKey: .aiAnalysis) {
            // If it's a dictionary, extract the relevant part
            aiAnalysis = aiResponse["ai_response"] ?? aiResponse.description
        } else if let aiResponseString = try? container.decode(String.self, forKey: .aiAnalysis) {
            // If it's a string, use it directly
            aiAnalysis = aiResponseString
        } else {
            aiAnalysis = nil
        }
        
        // Try to decode trend analysis
        if let topLevelTrendAnalysis = try? container.decode([String: String].self, forKey: .trendAnalysis) {
            trendAnalysis = topLevelTrendAnalysis
        } else {
            trendAnalysis = try? container.nestedContainer(keyedBy: AdditionalMetricsKeys.self, forKey: .additionalMetrics)
                .nestedContainer(keyedBy: AnalysisResultKeys.self, forKey: .analysisResult)
                .decode([String: String].self, forKey: .trendAnalysis)
        }
        
        // Get additional metrics if available
        additionalMetrics = try? container.decode([String: Double].self, forKey: .additionalMetrics)
    }
    
    init(id: String, heartRate: Double, bloodOxygen: Double, timestamp: String, isAnomaly: Bool, riskScore: Double, recommendations: [String], aiAnalysis: String? = nil, additionalMetrics: [String: Double]? = nil) {
        self.id = id
        self.heartRate = heartRate
        self.bloodOxygen = bloodOxygen
        self.timestamp = timestamp
        self.isAnomaly = isAnomaly
        self.riskScore = riskScore
        self.recommendations = recommendations
        self.aiAnalysis = aiAnalysis
        self.anomalyScore = nil
        self.trendAnalysis = nil
        self.additionalMetrics = additionalMetrics
    }
    
    // Returns a date object from the timestamp string
    func getDate() -> Date {
        // Try ISO8601 format first
        let iso8601DateFormatter = ISO8601DateFormatter()
        iso8601DateFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        
        if let date = iso8601DateFormatter.date(from: timestamp) {
            return date
        }
        
        // Try alternate formats
        let dateFormatter = DateFormatter()
        let formats = [
            "yyyy-MM-dd'T'HH:mm:ss.SSSZ",
            "yyyy-MM-dd'T'HH:mm:ssZ",
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd"
        ]
        
        for format in formats {
            dateFormatter.dateFormat = format
            if let date = dateFormatter.date(from: timestamp) {
                return date
            }
        }
        
        // If all parsing fails, return current date
        print("Warning: Could not parse timestamp: \(timestamp)")
        return Date()
    }
    
    // Formatted date string for display
    func formattedDate() -> String {
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        dateFormatter.timeStyle = .short
        return dateFormatter.string(from: getDate())
    }
    
    static func == (lhs: HealthData, rhs: HealthData) -> Bool {
        return lhs.id == rhs.id &&
               lhs.heartRate == rhs.heartRate &&
               lhs.bloodOxygen == rhs.bloodOxygen &&
               lhs.timestamp == rhs.timestamp &&
               lhs.isAnomaly == rhs.isAnomaly &&
               lhs.riskScore == rhs.riskScore &&
               lhs.recommendations == rhs.recommendations &&
               lhs.aiAnalysis == rhs.aiAnalysis
    }
}

struct HealthHistoryResponse: Decodable {
    let healthData: [HealthData]
    let count: Int
    
    enum CodingKeys: String, CodingKey {
        case healthData = "health_data"
        case history
        case count
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        // Try to decode from "health_data" first
        if let data = try? container.decode([HealthData].self, forKey: .healthData) {
            self.healthData = data
            print("Decoded \(data.count) items from health_data field")
        } 
        // Fall back to "history" if "health_data" is not present
        else if let history = try? container.decode([HealthData].self, forKey: .history) {
            self.healthData = history
            print("Decoded \(history.count) items from history field")
        } else {
            // If neither is present, log the error and use an empty array
            print("ERROR: Could not find health_data or history fields in API response")
            self.healthData = []
            
            // Just log the error - don't try to inspect raw data in ways that won't compile
            print("Failed to decode either 'health_data' or 'history' fields")
        }
        
        // Try to get the count or calculate it
        if let dataCount = try? container.decode(Int.self, forKey: .count) {
            self.count = dataCount
            print("Using count \(dataCount) from response")
        } else {
            self.count = self.healthData.count
            print("Using calculated count \(self.healthData.count)")
        }
    }
}
