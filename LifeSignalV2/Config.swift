//
//  Config.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/15/25.
//

import Foundation

enum AppEnvironment {
    case development
    case staging
    case production
    
    var apiBaseURL: String {
        switch self {
        case .development:
            return "http://localhost:5100"
        case .staging:
            return "https://staging.lifesignal-api.example.com/api"
        case .production:
            return "https://api.lifesignal.example.com/api"
        }
    }
}

struct Config {
    // Set the current environment here
    static let environment: AppEnvironment = .development
    
    // API configuration
    static let apiBaseURL = environment.apiBaseURL
    
    // Feature flags
    static let enableNotifications = true
    static let enableOfflineMode = false
    
    // Timeouts and limits
    static let requestTimeoutSeconds: TimeInterval = 30
    static let maxRetryAttempts = 3
} 
