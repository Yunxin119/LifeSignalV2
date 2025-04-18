//
//  APIService.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/15/25.
//

import Foundation
import Combine

enum APIError: Error {
    case invalidURL
    case networkError(Error)
    case responseError(Int, String?)
    case decodingError(Error)
    case unknown
    
    var message: String {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .responseError(let statusCode, let message):
            if let message = message {
                return "Server error (\(statusCode)): \(message)"
            }
            return "Server error with status code: \(statusCode)"
        case .decodingError(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .unknown:
            return "Unknown error occurred"
        }
    }
}

struct APIService {
    // Use the base URL from Config
    private static let baseURL = Config.apiBaseURL
    
    // MARK: - Authentication
    
    static func login(email: String, password: String) -> AnyPublisher<AuthResponse, APIError> {
        let endpoint = "/api/auth/login"
        let body: [String: Any] = [
            "username_or_email": email,
            "password": password
        ]
        
        return makeRequest(endpoint: endpoint, method: "POST", body: body)
    }
    
    static func register(username: String, email: String, password: String, healthConditions: [String]? = nil) -> AnyPublisher<AuthResponse, APIError> {
        let endpoint = "/api/auth/register"
        var body: [String: Any] = [
            "username": username,
            "email": email,
            "password": password
        ]
        
        if let healthConditions = healthConditions, !healthConditions.isEmpty {
            body["health_conditions"] = healthConditions
        }
        
        return makeRequest(endpoint: endpoint, method: "POST", body: body)
    }
    
    // MARK: - Helper methods
    
    private static func makeRequest<T: Decodable>(endpoint: String, method: String, body: [String: Any]? = nil, token: String? = nil) -> AnyPublisher<T, APIError> {
        
        guard let url = URL(string: baseURL + endpoint) else {
            return Fail(error: APIError.invalidURL).eraseToAnyPublisher()
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = Config.requestTimeoutSeconds
        
        if let token = token {
            request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        
        if let body = body {
            do {
                request.httpBody = try JSONSerialization.data(withJSONObject: body)
            } catch {
                return Fail(error: APIError.networkError(error)).eraseToAnyPublisher()
            }
        }
        
        return URLSession.shared.dataTaskPublisher(for: request)
            .mapError { APIError.networkError($0) }
            .tryMap { data, response in
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw APIError.unknown
                }
                
                if !(200...299).contains(httpResponse.statusCode) {
                    // Try to parse error message from response
                    var errorMessage: String? = nil
                    if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let message = json["error"] as? String {
                        errorMessage = message
                    }
                    throw APIError.responseError(httpResponse.statusCode, errorMessage)
                }
                
                return data
            }
            .decode(type: T.self, decoder: JSONDecoder())
            .mapError { error in
                if let apiError = error as? APIError {
                    return apiError
                } else {
                    return APIError.decodingError(error)
                }
            }
            .eraseToAnyPublisher()
    }
}

// MARK: - Response Models

struct AuthResponse: Decodable {
    let success: Bool
    let token: String?
    let user: User?
    let error: String?
    
    struct User: Codable {
        let id: String
        let username: String
        let email: String
        let healthConditions: [String]?
    }
}

// MARK: - Health Conditions
extension APIService {
    static func updateHealthConditions(healthConditions: [String], token: String) -> AnyPublisher<AuthResponse, APIError> {
        let endpoint = "/api/auth/update-health"
        let body: [String: Any] = [
            "health_conditions": healthConditions
        ]
        
        return makeRequest(endpoint: endpoint, method: "POST", body: body, token: token)
    }
}

// MARK: - Model Evaluation
extension APIService {
    static func evaluateUserModel(token: String) -> AnyPublisher<ModelEvaluationResponse, APIError> {
        let endpoint = "/api/health/evaluate-model"
        return makeRequest(endpoint: endpoint, method: "GET", token: token)
    }
    
    static func simulateHealthData(token: String, days: Int, abnormalProb: Double, readingsPerDay: Int) -> AnyPublisher<SimulationResponse, APIError> {
        let endpoint = "/api/health/simulate"
        let body: [String: Any] = [
            "days": days,
            "abnormal_prob": abnormalProb,
            "readings_per_day": readingsPerDay
        ]
        
        return makeRequest(endpoint: endpoint, method: "POST", body: body, token: token)
    }
}

// MARK: - Additional Response Models
struct ModelEvaluationResponse: Decodable {
    let testPoints: Int
    let mlModelError: Double
    let hybridModelError: Double
    let improvement: Double
    let sampleData: [SampleDataPoint]
    
    enum CodingKeys: String, CodingKey {
        case testPoints = "test_points"
        case mlModelError = "ml_model_error"
        case hybridModelError = "hybrid_model_error"
        case improvement
        case sampleData = "sample_data"
    }
    
    struct SampleDataPoint: Decodable {
        let heartRate: Double
        let bloodOxygen: Double
        let trueRisk: Double
        let mlRisk: Double
        let hybridRisk: Double
        
        enum CodingKeys: String, CodingKey {
            case heartRate = "heart_rate"
            case bloodOxygen = "blood_oxygen"
            case trueRisk = "true_risk"
            case mlRisk = "ml_risk"
            case hybridRisk = "hybrid_risk"
        }
    }
}

struct SimulationResponse: Decodable {
    let message: String
    let recordsCreated: Int
    let samples: [HealthSample]
    let modelEvaluation: ModelEvaluation
    
    enum CodingKeys: String, CodingKey {
        case message
        case recordsCreated = "records_created"
        case samples
        case modelEvaluation = "model_evaluation"
    }
    
    struct HealthSample: Decodable {
        let id: String
        let heartRate: Double
        let bloodOxygen: Double
        let riskScore: Double
        let isAnomaly: Bool
        let timestamp: String
        
        enum CodingKeys: String, CodingKey {
            case id
            case heartRate = "heart_rate"
            case bloodOxygen = "blood_oxygen"
            case riskScore = "risk_score"
            case isAnomaly = "is_anomaly"
            case timestamp
        }
    }
    
    struct ModelEvaluation: Decodable {
        let mlModelError: Double?
        let hybridModelError: Double?
        let improvement: Double?
        let error: String?
        
        enum CodingKeys: String, CodingKey {
            case mlModelError = "ml_model_error"
            case hybridModelError = "hybrid_model_error"
            case improvement
            case error
        }
    }
} 
