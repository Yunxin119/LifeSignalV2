//
//  DetailedAIAnalysis.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/16/25.
//

struct DetailedAIAnalysis: Decodable {
    let status: String
    let assessment: String
    let explanation: String
    let concerns: [String]
    let recommendations: [String]
    let trendAnalysis: String 
    let seekMedicalAttention: String
    let followUp: String
    
    enum CodingKeys: String, CodingKey {
        case status, assessment, explanation, concerns, recommendations
        case trendAnalysis = "trend_analysis"
        case seekMedicalAttention = "seek_medical_attention"
        case followUp = "follow_up"
        case healthTrajectory = "health_trajectory"
        case potentialConcerns = "potential_concerns"
        case lifestyleAdjustments = "lifestyle_adjustments"
        case medicalConsultation = "medical_consultation"
        case trendSummary = "trend_summary"
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        // Try to decode status with fallbacks
        if let statusValue = try? container.decode(String.self, forKey: .status) {
            status = statusValue
        } else if let trajectoryValue = try? container.decode(String.self, forKey: .healthTrajectory) {
            status = trajectoryValue
        } else {
            status = "unknown"
        }
        
        // Try to decode assessment
        assessment = try container.decodeIfPresent(String.self, forKey: .assessment) ?? ""
        
        // Try to decode explanation with fallbacks
        if let explValue = try? container.decode(String.self, forKey: .explanation) {
            explanation = explValue
        } else if let summaryValue = try? container.decode(String.self, forKey: .trendSummary) {
            explanation = summaryValue
        } else {
            explanation = "Analysis based on recent health data trends."
        }
        
        // Try to decode concerns with fallbacks
        if let concernsList = try? container.decode([String].self, forKey: .concerns) {
            concerns = concernsList
        } else if let potentialList = try? container.decode([String].self, forKey: .potentialConcerns) {
            concerns = potentialList
        } else {
            concerns = []
        }
        
        // Try to decode recommendations
        recommendations = try container.decodeIfPresent([String].self, forKey: .recommendations) ?? []
        
        // Try to decode trend analysis
        if let analysisValue = try? container.decode(String.self, forKey: .trendAnalysis) {
            trendAnalysis = analysisValue
        } else if let summaryValue = try? container.decode(String.self, forKey: .trendSummary) {
            trendAnalysis = summaryValue
        } else {
            trendAnalysis = ""
        }
        
        // Try to decode medical attention recommendations
        if let attentionValue = try? container.decode(String.self, forKey: .seekMedicalAttention) {
            seekMedicalAttention = attentionValue
        } else if let consultationValue = try? container.decode(String.self, forKey: .medicalConsultation) {
            seekMedicalAttention = consultationValue
        } else {
            seekMedicalAttention = ""
        }
        
        // Try to decode follow up with fallbacks
        if let followUpValue = try? container.decode(String.self, forKey: .followUp) {
            followUp = followUpValue
        } else if let lifestyleValue = try? container.decode([String].self, forKey: .lifestyleAdjustments) {
            followUp = lifestyleValue.joined(separator: ". ")
        } else {
            followUp = "Continue monitoring your health metrics regularly."
        }
    }
    
    // Constructor for creating instances manually (used for fallback/default responses)
    init(status: String, assessment: String, explanation: String, concerns: [String], 
         recommendations: [String], trendAnalysis: String, seekMedicalAttention: String, followUp: String) {
        self.status = status
        self.assessment = assessment
        self.explanation = explanation
        self.concerns = concerns
        self.recommendations = recommendations
        self.trendAnalysis = trendAnalysis
        self.seekMedicalAttention = seekMedicalAttention
        self.followUp = followUp
    }
}


