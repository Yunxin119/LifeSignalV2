//
//  AITrendAnalysisModel.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/16/25.
//

// This model directly maps to the "ai_analysis" field in the server response
struct AIAnalysisResponse: Decodable {
    let trends: HealthTrends?
    let aiAnalysis: ServerAIAnalysis?
    
    enum CodingKeys: String, CodingKey {
        case trends
        case aiAnalysis = "ai_analysis"
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        // Decode trends normally
        trends = try container.decodeIfPresent(HealthTrends.self, forKey: .trends)
        
        // Try to decode aiAnalysis as an object first
        if let analysisContainer = try? container.nestedContainer(keyedBy: ServerAIAnalysis.CodingKeys.self, forKey: .aiAnalysis) {
            aiAnalysis = try ServerAIAnalysis(from: analysisContainer)
        } else {
            // If that fails, try to decode as a string
            if let analysisString = try? container.decode(String.self, forKey: .aiAnalysis) {
                // Convert the string to a ServerAIAnalysis object
                aiAnalysis = ServerAIAnalysis(fromUnstructuredText: analysisString)
            } else {
                aiAnalysis = nil
            }
        }
    }
}

// This model directly maps to the structure inside the "ai_analysis" field
struct ServerAIAnalysis: Decodable {
    let assessment: String
    let healthTrajectory: String
    let lifestyleAdjustments: [String]
    let medicalConsultation: String
    let potentialConcerns: [String]
    let recommendations: [String]
    let trendSummary: String
    
    enum CodingKeys: String, CodingKey {
        case assessment
        case healthTrajectory = "health_trajectory"
        case lifestyleAdjustments = "lifestyle_adjustments"
        case medicalConsultation = "medical_consultation"
        case potentialConcerns = "potential_concerns"
        case recommendations
        case trendSummary = "trend_summary"
    }
    
    // Custom initializer to handle parsing from container
    init(from container: KeyedDecodingContainer<CodingKeys>) throws {
        assessment = try container.decodeIfPresent(String.self, forKey: .assessment) ?? ""
        healthTrajectory = try container.decodeIfPresent(String.self, forKey: .healthTrajectory) ?? "stable"
        
        // For arrays, try to decode them but provide empty arrays if it fails
        do {
            lifestyleAdjustments = try container.decode([String].self, forKey: .lifestyleAdjustments)
        } catch {
            lifestyleAdjustments = []
        }
        
        medicalConsultation = try container.decodeIfPresent(String.self, forKey: .medicalConsultation) ?? ""
        
        do {
            potentialConcerns = try container.decode([String].self, forKey: .potentialConcerns)
        } catch {
            potentialConcerns = []
        }
        
        do {
            recommendations = try container.decode([String].self, forKey: .recommendations)
        } catch {
            recommendations = []
        }
        
        trendSummary = try container.decodeIfPresent(String.self, forKey: .trendSummary) ?? ""
    }
    
    // Implement standard Decodable initializer
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(from: container)
    }
    
    // Initialize from unstructured text (when the AI model returns a plain string)
    init(fromUnstructuredText text: String) {
        // Extract summary from the start of the text (first paragraph)
        let paragraphs = text.components(separatedBy: "\n\n").filter { !$0.isEmpty }
        self.trendSummary = paragraphs.first ?? text
        
        // Try to identify health trajectory from keywords
        if text.lowercased().contains("improving") || text.lowercased().contains("improvement") {
            self.healthTrajectory = "improving"
        } else if text.lowercased().contains("declining") || text.lowercased().contains("worse") {
            self.healthTrajectory = "declining"
        } else {
            self.healthTrajectory = "stable"
        }
        
        // Extract assessment - use second paragraph if available
        self.assessment = paragraphs.count > 1 ? paragraphs[1] : ""
        
        // Extract potential concerns - look for sections with keywords
        var concerns: [String] = []
        for paragraph in paragraphs {
            let lower = paragraph.lowercased()
            if lower.contains("concern") || lower.contains("issue") || lower.contains("problem") || lower.contains("warning") {
                // Split paragraph into sentences and add them as concerns
                let sentences = paragraph.components(separatedBy: ". ").filter { !$0.isEmpty }
                concerns.append(contentsOf: sentences)
                break
            }
        }
        self.potentialConcerns = concerns
        
        // Extract recommendations
        var recommendations: [String] = []
        for paragraph in paragraphs {
            let lower = paragraph.lowercased()
            if lower.contains("recommend") || lower.contains("suggestion") || lower.contains("advised") {
                // Split paragraph into sentences and add them as recommendations
                let sentences = paragraph.components(separatedBy: ". ").filter { !$0.isEmpty }
                recommendations.append(contentsOf: sentences)
                break
            }
        }
        self.recommendations = recommendations
        
        // Extract lifestyle adjustments
        var lifestyle: [String] = []
        for paragraph in paragraphs {
            let lower = paragraph.lowercased()
            if lower.contains("lifestyle") || lower.contains("habit") || lower.contains("routine") || lower.contains("daily") {
                // Split paragraph into sentences
                let sentences = paragraph.components(separatedBy: ". ").filter { !$0.isEmpty }
                lifestyle.append(contentsOf: sentences)
                break
            }
        }
        self.lifestyleAdjustments = lifestyle
        
        // Extract medical consultation advice
        var consultation = ""
        for paragraph in paragraphs {
            let lower = paragraph.lowercased()
            if lower.contains("doctor") || lower.contains("medical") || lower.contains("healthcare") || lower.contains("physician") {
                consultation = paragraph
                break
            }
        }
        self.medicalConsultation = consultation
    }
    
    // Regular initializer for manually creating instances
    init(trendSummary: String, healthTrajectory: String, assessment: String, 
         potentialConcerns: [String], recommendations: [String],
         lifestyleAdjustments: [String], medicalConsultation: String) {
        self.trendSummary = trendSummary
        self.healthTrajectory = healthTrajectory
        self.assessment = assessment
        self.potentialConcerns = potentialConcerns
        self.recommendations = recommendations
        self.lifestyleAdjustments = lifestyleAdjustments
        self.medicalConsultation = medicalConsultation
    }
}

// This is the model used by the UI - we'll convert the server model to this format
struct AITrendAnalysis: Decodable {
    let trendSummary: String
    let healthTrajectory: String
    let assessment: String
    let potentialConcerns: [String]
    let recommendations: [String]
    let lifestyleAdjustments: [String]
    let medicalConsultation: String
    
    // This initializer allows creating instances directly
    init(trendSummary: String, healthTrajectory: String, assessment: String, 
         potentialConcerns: [String], recommendations: [String],
         lifestyleAdjustments: [String], medicalConsultation: String) {
        self.trendSummary = trendSummary
        self.healthTrajectory = healthTrajectory
        self.assessment = assessment
        self.potentialConcerns = potentialConcerns
        self.recommendations = recommendations
        self.lifestyleAdjustments = lifestyleAdjustments
        self.medicalConsultation = medicalConsultation
    }
    
    // This initializer allows converting from the server model
    init(from serverAnalysis: ServerAIAnalysis) {
        self.trendSummary = serverAnalysis.trendSummary
        self.healthTrajectory = serverAnalysis.healthTrajectory
        self.assessment = serverAnalysis.assessment
        self.potentialConcerns = serverAnalysis.potentialConcerns
        self.recommendations = serverAnalysis.recommendations
        self.lifestyleAdjustments = serverAnalysis.lifestyleAdjustments
        self.medicalConsultation = serverAnalysis.medicalConsultation
    }
}
