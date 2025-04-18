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
    
    static func register(username: String, email: String, password: String) -> AnyPublisher<AuthResponse, APIError> {
        let endpoint = "/api/auth/register"
        let body: [String: Any] = [
            "username": username,
            "email": email,
            "password": password
        ]
        
        return makeRequest(endpoint: endpoint, method: "POST", body: body)
    }
    
    static func registerWithHealthProfile(body: [String: Any]) -> AnyPublisher<AuthResponse, APIError> {
        let endpoint = "/api/auth/register"
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
    }
} 
