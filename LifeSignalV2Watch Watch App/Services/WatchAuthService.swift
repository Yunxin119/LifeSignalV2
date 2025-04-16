import Foundation
import SwiftUI

class WatchAuthService: ObservableObject {
    static let shared = WatchAuthService()
    
    @Published var userId: String?
    @Published var isAuthenticated = false
    @Published var isLoading = false
    @Published var error: String?
    @Published var isPaired = false
    
    private let userDefaults = UserDefaults.standard
    private let userIdKey = "userId"
    private let pairingStatusKey = "isPaired"
    private let watchTokenKey = "watchAuthToken"
    
    // Base URL for the API server - update this for your production environment
    #if DEBUG
    public let baseURL = "http://localhost:5100"
    #else
    public let baseURL = "https://yourproductionurl.com"
    #endif
    
    private init() {
        loadUserFromStorage()
    }
    
    func submitPairingCode(pairingCode: String) async -> Bool {
        DispatchQueue.main.async {
            self.isLoading = true
            self.error = nil
        }
        
        do {
            // Create request body
            let requestBody: [String: Any] = [
                "pairing_code": pairingCode,
                "device_type": "apple_watch"
            ]
            
            // Convert request body to JSON data
            let jsonData = try JSONSerialization.data(withJSONObject: requestBody)
            
            // Create URL request
            let url = URL(string: "\(baseURL)/api/pair/validate")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = jsonData
            
            // Make network request
            let (data, response) = try await URLSession.shared.data(for: request)
            
            // Check response status
            guard let httpResponse = response as? HTTPURLResponse else {
                DispatchQueue.main.async {
                    self.error = "Invalid response"
                    self.isLoading = false
                }
                return false
            }
            
            if httpResponse.statusCode == 200 || httpResponse.statusCode == 201 {
                // Parse response
                let responseJSON = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                
                if let success = responseJSON?["success"] as? Bool,
                   let userData = responseJSON?["user"] as? [String: Any],
                   let userId = userData["_id"] as? String,
                   let token = responseJSON?["token"] as? String,
                   success {
                    
                    // Save user ID, pairing status and token
                    DispatchQueue.main.async {
                        self.userId = userId
                        self.isPaired = true
                        self.isAuthenticated = true
                        self.saveUserToStorage(userId: userId)
                        self.saveToken(token: token)
                        self.isLoading = false
                    }
                    return true
                } else {
                    let errorMessage = responseJSON?["error"] as? String ?? "Pairing failed: incomplete information received"
                    DispatchQueue.main.async {
                        self.error = errorMessage
                        self.isLoading = false
                    }
                    return false
                }
            } else {
                // Handle error response
                let errorJSON = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                let errorMessage = errorJSON?["error"] as? String ?? "Pairing failed with status \(httpResponse.statusCode)"
                DispatchQueue.main.async {
                    self.error = errorMessage
                    self.isLoading = false
                }
                return false
            }
        } catch {
            DispatchQueue.main.async {
                self.error = error.localizedDescription
                self.isLoading = false
            }
            return false
        }
    }
    
    // Check if the device is already paired
    func checkPairingStatus() async -> Bool {
        guard let token = getToken(), let userId = userId else {
            return false
        }
        
        do {
            // Create URL request
            let url = URL(string: "\(baseURL)/api/pair/status?device_type=apple_watch")!
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            
            // Make network request
            let (data, response) = try await URLSession.shared.data(for: request)
            
            // Check response status
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                return false
            }
            
            // Parse response
            if let responseJSON = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let success = responseJSON["success"] as? Bool,
               let isPaired = responseJSON["is_paired"] as? Bool {
                
                DispatchQueue.main.async {
                    self.isPaired = isPaired
                }
                
                return isPaired
            }
            
            return false
        } catch {
            print("Failed to check pairing status: \(error.localizedDescription)")
            return false
        }
    }
    
    // Unpair the device
    func unpairDevice() async -> Bool {
        guard let token = getToken() else {
            DispatchQueue.main.async {
                self.error = "Not authenticated"
            }
            return false
        }
        
        do {
            // Create request body
            let requestBody: [String: Any] = [
                "device_type": "apple_watch"
            ]
            
            // Convert request body to JSON data
            let jsonData = try JSONSerialization.data(withJSONObject: requestBody)
            
            // Create URL request
            let url = URL(string: "\(baseURL)/api/pair/unpair")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            request.httpBody = jsonData
            
            // Make network request
            let (data, response) = try await URLSession.shared.data(for: request)
            
            // Check response status
            guard let httpResponse = response as? HTTPURLResponse else {
                return false
            }
            
            if httpResponse.statusCode == 200 || httpResponse.statusCode == 201 {
                // Parse response
                if let responseJSON = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let success = responseJSON["success"] as? Bool, success {
                    
                    // Reset pairing status
                    logout()
                    return true
                }
            }
            
            return false
        } catch {
            print("Failed to unpair device: \(error.localizedDescription)")
            return false
        }
    }
    
    // Upload health data to the server
    func uploadHealthData(_ healthData: HealthData) async -> Bool {
        guard let userId = userId, let token = getToken() else {
            DispatchQueue.main.async {
                self.error = "Not authenticated"
            }
            return false
        }
        
        do {
            // Create request body
            var requestBody = healthData.toDictionary()
            requestBody["user_id"] = userId
            
            // Convert request body to JSON data
            let jsonData = try JSONSerialization.data(withJSONObject: requestBody)
            
            // Create URL request
            let url = URL(string: "\(baseURL)/api/health/analyze")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            request.httpBody = jsonData
            
            // Make network request
            let (data, response) = try await URLSession.shared.data(for: request)
            
            // Check response status
            guard let httpResponse = response as? HTTPURLResponse else {
                return false
            }
            
            if httpResponse.statusCode == 200 || httpResponse.statusCode == 201 {
                // Parse response to check if alert was triggered
                if let responseJSON = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let alert = responseJSON["alert"] as? Bool, alert {
                    // If server detected anomaly, notify the app
                    DispatchQueue.main.async {
                        NotificationCenter.default.post(
                            name: Notification.Name.serverAnomalyDetected,
                            object: nil,
                            userInfo: ["healthData": healthData]
                        )
                    }
                }
                return true
            }
            
            return false
        } catch {
            print("Failed to upload health data: \(error.localizedDescription)")
            return false
        }
    }
    
    // Fetch health history from the server
    func fetchHealthHistory(limit: Int = 10) async -> [HealthData]? {
        guard let token = getToken() else {
            DispatchQueue.main.async {
                self.error = "Not authenticated"
            }
            return nil
        }
        
        do {
            // Create URL request
            let url = URL(string: "\(baseURL)/api/health/history?limit=\(limit)")!
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            
            // Make network request
            let (data, response) = try await URLSession.shared.data(for: request)
            
            // Check response status
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                return nil
            }
            
            // Parse response
            if let responseJSON = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let historyData = responseJSON["history"] as? [[String: Any]] {
                
                // Convert to HealthData objects
                var healthDataList: [HealthData] = []
                
                for item in historyData {
                    let heartRate = item["heart_rate"] as? Double
                    let bloodOxygen = item["blood_oxygen"] as? Double
                    let hasFallDetected = item["hasFallDetected"] as? Bool ?? false
                    let timestamp = item["timestamp"] as? Double ?? Date().timeIntervalSince1970
                    
                    var locationData: HealthData.LocationData? = nil
                    if let location = item["location"] as? [String: Double],
                       let latitude = location["latitude"],
                       let longitude = location["longitude"] {
                        locationData = HealthData.LocationData(latitude: latitude, longitude: longitude)
                    }
                    
                    let healthData = HealthData(
                        timestamp: Date(timeIntervalSince1970: timestamp),
                        heartRate: heartRate,
                        bloodOxygen: bloodOxygen,
                        hasFallDetected: hasFallDetected,
                        location: locationData
                    )
                    
                    healthDataList.append(healthData)
                }
                
                return healthDataList
            }
            
            return nil
        } catch {
            print("Failed to fetch health history: \(error.localizedDescription)")
            return nil
        }
    }
    
    // Send emergency alert to server
    func sendEmergencyAlert(healthData: HealthData) async -> Bool {
        guard let userId = userId, let token = getToken() else {
            DispatchQueue.main.async {
                self.error = "Not authenticated"
            }
            return false
        }
        
        do {
            // Create request body
            var requestBody = healthData.toDictionary()
            requestBody["user_id"] = userId
            requestBody["alert_type"] = "emergency"
            
            // Convert request body to JSON data
            let jsonData = try JSONSerialization.data(withJSONObject: requestBody)
            
            // Create URL request - assuming you have an emergency endpoint
            let url = URL(string: "\(baseURL)/api/health/emergency")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            request.httpBody = jsonData
            
            // Make network request
            let (_, response) = try await URLSession.shared.data(for: request)
            
            // Check response status
            guard let httpResponse = response as? HTTPURLResponse else {
                return false
            }
            
            return httpResponse.statusCode == 200 || httpResponse.statusCode == 201
        } catch {
            print("Failed to send emergency alert: \(error.localizedDescription)")
            return false
        }
    }
    
    // MARK: - Helper Methods
    
    func isValidAuthentication() -> Bool {
        return isPaired && isAuthenticated && userId != nil && getToken() != nil
    }
    
    func logout() {
        userDefaults.removeObject(forKey: userIdKey)
        userDefaults.removeObject(forKey: pairingStatusKey)
        userDefaults.removeObject(forKey: watchTokenKey)
        
        DispatchQueue.main.async {
            self.userId = nil
            self.isPaired = false
            self.isAuthenticated = false
        }
    }
    
    private func loadUserFromStorage() {
        if let storedUserId = userDefaults.string(forKey: userIdKey) {
            userId = storedUserId
            isPaired = userDefaults.bool(forKey: pairingStatusKey)
            isAuthenticated = getToken() != nil
        }
    }
    
    private func saveUserToStorage(userId: String) {
        userDefaults.set(userId, forKey: userIdKey)
        userDefaults.set(true, forKey: pairingStatusKey)
    }
    
    private func saveToken(token: String) {
        userDefaults.set(token, forKey: watchTokenKey)
    }
    
    private func getToken() -> String? {
        return userDefaults.string(forKey: watchTokenKey)
    }
}

// MARK: - Notification Names
extension Notification.Name {
    static let serverAnomalyDetected = Notification.Name("serverAnomalyDetected")
} 