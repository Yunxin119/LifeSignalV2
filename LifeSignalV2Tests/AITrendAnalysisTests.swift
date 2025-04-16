//
//  AITrendAnalysisTests.swift
//  LifeSignalV2Tests
//
//  Created by Yunxin Liu on 4/16/25.
//

import XCTest
@testable import LifeSignalV2

final class AITrendAnalysisTests: XCTestCase {
    
    // Test decoding a structured JSON response
    func testStructuredResponseDecoding() throws {
        // Create sample JSON data with structured response
        let jsonString = """
        {
            "trends": {
                "days_analyzed": 30,
                "data_points": 120,
                "heart_rate": {
                    "mean": 72.5,
                    "std": 5.2,
                    "min": 61.0,
                    "max": 88.0,
                    "trend": "stable"
                },
                "blood_oxygen": {
                    "mean": 97.8,
                    "std": 1.1,
                    "min": 95.0,
                    "max": 99.5,
                    "trend": "stable"
                }
            },
            "ai_analysis": {
                "assessment": "Your health metrics are within normal ranges.",
                "health_trajectory": "stable",
                "lifestyle_adjustments": ["Consider more outdoor activities", "Stay hydrated"],
                "medical_consultation": "Routine check-up recommended in 6 months",
                "potential_concerns": ["Slight elevation in nighttime heart rate"],
                "recommendations": ["Continue regular exercise", "Monitor sleep quality"],
                "trend_summary": "Your heart rate and blood oxygen levels have remained stable."
            }
        }
        """
        
        let jsonData = jsonString.data(using: .utf8)!
        
        // Decode the data
        let decoder = JSONDecoder()
        let response = try decoder.decode(AIAnalysisResponse.self, from: jsonData)
        
        // Verify trends are decoded correctly
        XCTAssertNotNil(response.trends)
        XCTAssertEqual(response.trends?.daysAnalyzed, 30)
        XCTAssertEqual(response.trends?.dataPoints, 120)
        XCTAssertEqual(response.trends?.heartRate.mean, 72.5)
        XCTAssertEqual(response.trends?.bloodOxygen.trend, "stable")
        
        // Verify AI analysis is decoded correctly
        XCTAssertNotNil(response.aiAnalysis)
        XCTAssertEqual(response.aiAnalysis?.healthTrajectory, "stable")
        XCTAssertEqual(response.aiAnalysis?.assessment, "Your health metrics are within normal ranges.")
        XCTAssertEqual(response.aiAnalysis?.lifestyleAdjustments.count, 2)
        XCTAssertEqual(response.aiAnalysis?.potentialConcerns.first, "Slight elevation in nighttime heart rate")
        
        // Create UI model from server model
        let uiModel = AITrendAnalysis(from: response.aiAnalysis!)
        
        // Verify UI model has correct data
        XCTAssertEqual(uiModel.trendSummary, "Your heart rate and blood oxygen levels have remained stable.")
        XCTAssertEqual(uiModel.recommendations.count, 2)
    }
    
    // Test decoding an unstructured (string) response
    func testUnstructuredResponseDecoding() throws {
        // Create sample JSON data with string response
        let jsonString = """
        {
            "trends": {
                "days_analyzed": 30,
                "data_points": 120,
                "heart_rate": {
                    "mean": 72.5,
                    "std": 5.2,
                    "min": 61.0,
                    "max": 88.0,
                    "trend": "stable"
                },
                "blood_oxygen": {
                    "mean": 97.8,
                    "std": 1.1,
                    "min": 95.0,
                    "max": 99.5,
                    "trend": "stable"
                }
            },
            "ai_analysis": "Your heart rate and blood oxygen levels have remained stable over the past month. This indicates good cardiovascular health.\\n\\nBased on your data, your health patterns are within normal ranges for your demographic group.\\n\\nI have no concerns about your current health metrics.\\n\\nI recommend you continue with your current activity levels and regular monitoring.\\n\\nFor lifestyle, consider incorporating more outdoor activities which can further improve your cardiovascular health.\\n\\nNo need for immediate medical attention. A routine check-up within the next 6 months would be sufficient."
        }
        """
        
        let jsonData = jsonString.data(using: .utf8)!
        
        // Decode the data
        let decoder = JSONDecoder()
        let response = try decoder.decode(AIAnalysisResponse.self, from: jsonData)
        
        // Verify trends are decoded correctly (same as before)
        XCTAssertNotNil(response.trends)
        XCTAssertEqual(response.trends?.daysAnalyzed, 30)
        
        // Verify AI analysis is converted from string to structured format
        XCTAssertNotNil(response.aiAnalysis)
        XCTAssertEqual(response.aiAnalysis?.healthTrajectory, "stable")
        XCTAssertFalse(response.aiAnalysis?.trendSummary.isEmpty ?? true)
        
        // Check that we extracted recommendations
        XCTAssertTrue(response.aiAnalysis?.recommendations.count ?? 0 > 0)
        
        // Create UI model from server model
        let uiModel = AITrendAnalysis(from: response.aiAnalysis!)
        
        // Verify UI model has data extracted from text
        XCTAssertFalse(uiModel.trendSummary.isEmpty)
    }
    
    // Test direct string response (no JSON)
    func testPlainTextResponse() {
        // Create a sample plain text response
        let textResponse = """
        Your health metrics are showing improvement. Heart rate and blood oxygen levels are trending positively.
        
        Your overall assessment is positive, with metrics within healthy ranges for your demographic.
        
        I have concerns about occasional spikes in heart rate during evenings, which might indicate stress.
        
        I recommend increasing water intake and ensuring 7-8 hours of sleep consistently.
        
        For lifestyle adjustments, consider incorporating meditation or deep breathing exercises to manage stress levels.
        
        No urgent medical consultation needed, but discuss these trends at your next regular check-up.
        """
        
        // Create response using unstructured text initializer
        let serverAnalysis = ServerAIAnalysis(fromUnstructuredText: textResponse)
        
        // Verify text was parsed into structured format
        XCTAssertEqual(serverAnalysis.healthTrajectory, "improving")
        XCTAssertFalse(serverAnalysis.trendSummary.isEmpty)
        XCTAssertTrue(serverAnalysis.concerns.count > 0)
        XCTAssertTrue(serverAnalysis.recommendations.count > 0)
        XCTAssertTrue(serverAnalysis.lifestyleAdjustments.count > 0)
        XCTAssertFalse(serverAnalysis.medicalConsultation.isEmpty)
        
        // Create UI model
        let uiModel = AITrendAnalysis(from: serverAnalysis)
        
        // Verify UI model has properly extracted data
        XCTAssertEqual(uiModel.healthTrajectory, "improving")
        XCTAssertTrue(uiModel.recommendations.count > 0)
    }
} 