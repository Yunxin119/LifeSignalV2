//
//  UserAuthModel.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/15/25.
//

import SwiftUI
import Combine

class UserAuthModel: ObservableObject {
    @Published var isAuthenticated = false
    @Published var currentUser: User?
    @Published var loginError: String?
    @Published var registrationError: String?
    @Published var isLoading = false
    
    private var cancellables = Set<AnyCancellable>()
    
    // Keychain keys
    private let tokenKey = "auth_token"
    private let userKey = "user_data"
    
    // Add computed property to get the token
    var token: String? {
        return UserDefaults.standard.string(forKey: tokenKey)
    }
    
    init() {
        // Check for stored token and user data on startup
        loadUserFromKeychain()
    }
    
    struct User {
        var id: String
        var username: String
        var email: String
    }
    
    func login(email: String, password: String) {
        // Reset error state
        loginError = nil
        isLoading = true
        
        // Basic validation
        if email.isEmpty || password.isEmpty {
            loginError = "Email and password cannot be empty"
            isLoading = false
            return
        }
        
        // Call API service
        APIService.login(email: email, password: password)
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { [weak self] completion in
                    self?.isLoading = false
                    
                    if case .failure(let error) = completion {
                        self?.loginError = error.message
                    }
                },
                receiveValue: { [weak self] response in
                    if response.success, let token = response.token, let userData = response.user {
                        // Store token and user data securely
                        self?.saveToKeychain(token: token, user: userData)
                        
                        // Update current user
                        self?.currentUser = User(
                            id: userData.id,
                            username: userData.username,
                            email: userData.email
                        )
                        
                        // Set authenticated state
                        self?.isAuthenticated = true
                        
                        // Send login success notification
                        if let notificationService = NotificationService.shared {
                            notificationService.sendLoginSuccessNotification(username: userData.username)
                        }
                    } else {
                        self?.loginError = response.error ?? "Login failed"
                    }
                }
            )
            .store(in: &cancellables)
    }
    
    func register(username: String, email: String, password: String, confirmPassword: String) {
        registrationError = nil
        isLoading = true
        
        // Simple validation
        if username.isEmpty || email.isEmpty || password.isEmpty {
            registrationError = "All fields are required"
            isLoading = false
            return
        }
        
        if !email.contains("@") {
            registrationError = "Please enter a valid email"
            isLoading = false
            return
        }
        
        if password.count < 6 {
            registrationError = "Password must be at least 6 characters"
            isLoading = false
            return
        }
        
        if password != confirmPassword {
            registrationError = "Passwords do not match"
            isLoading = false
            return
        }
        
        // Call API service
        APIService.register(username: username, email: email, password: password)
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { [weak self] completion in
                    self?.isLoading = false
                    
                    if case .failure(let error) = completion {
                        self?.registrationError = error.message
                    }
                },
                receiveValue: { [weak self] response in
                    if response.success, let token = response.token, let userData = response.user {
                        // Store token and user data securely
                        self?.saveToKeychain(token: token, user: userData)
                        
                        // Update current user
                        self?.currentUser = User(
                            id: userData.id,
                            username: userData.username,
                            email: userData.email
                        )
                        
                        // Set authenticated state
                        self?.isAuthenticated = true
                    } else {
                        self?.registrationError = response.error ?? "Registration failed"
                    }
                }
            )
            .store(in: &cancellables)
    }
    
    func registerWithHealthProfile(username: String, email: String, password: String, confirmPassword: String, healthProfile: [String: Any]?) {
        registrationError = nil
        isLoading = true
        
        // Simple validation
        if username.isEmpty || email.isEmpty || password.isEmpty {
            registrationError = "All fields are required"
            isLoading = false
            return
        }
        
        if !email.contains("@") {
            registrationError = "Please enter a valid email"
            isLoading = false
            return
        }
        
        if password.count < 6 {
            registrationError = "Password must be at least 6 characters"
            isLoading = false
            return
        }
        
        if password != confirmPassword {
            registrationError = "Passwords do not match"
            isLoading = false
            return
        }
        
        // Create registration payload with health profile if provided
        var body: [String: Any] = [
            "username": username,
            "email": email,
            "password": password
        ]
        
        // Add health profile data if available
        if let healthProfile = healthProfile {
            body["health_profile"] = healthProfile
        }
        
        // Call API service with health profile data
        APIService.registerWithHealthProfile(body: body)
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { [weak self] completion in
                    self?.isLoading = false
                    
                    if case .failure(let error) = completion {
                        self?.registrationError = error.message
                    }
                },
                receiveValue: { [weak self] response in
                    if response.success, let token = response.token, let userData = response.user {
                        // Store token and user data securely
                        self?.saveToKeychain(token: token, user: userData)
                        
                        // Update current user
                        self?.currentUser = User(
                            id: userData.id,
                            username: userData.username,
                            email: userData.email
                        )
                        
                        // Set authenticated state
                        self?.isAuthenticated = true
                    } else {
                        self?.registrationError = response.error ?? "Registration failed"
                    }
                }
            )
            .store(in: &cancellables)
    }
    
    func logout() {
        // Clear stored token and user data
        removeFromKeychain()
        
        // Reset state
        isAuthenticated = false
        currentUser = nil
        
        // Cancel any ongoing requests
        cancellables.forEach { $0.cancel() }
        cancellables.removeAll()
    }
    
    // Method to store a cancellable from outside the class
    func store(_ cancellable: AnyCancellable) {
        cancellable.store(in: &cancellables)
    }
    
    // MARK: - Keychain methods
    
    private func saveToKeychain(token: String, user: AuthResponse.User) {
        // THIS IS A DEMO PURPOSE THING, In future if we wanna make real app, we might change it to real keychain feature.
        UserDefaults.standard.set(token, forKey: tokenKey)
        
        // Convert user to JSON and save
        if let userData = try? JSONEncoder().encode(user) {
            UserDefaults.standard.set(userData, forKey: userKey)
        }
    }
    
    private func loadUserFromKeychain() {
        // Get token and user data
        guard let token = UserDefaults.standard.string(forKey: tokenKey),
              let userData = UserDefaults.standard.data(forKey: userKey) else {
            return
        }
        
        // Decode the user data
        do {
            let user = try JSONDecoder().decode(AuthResponse.User.self, from: userData)
            
            currentUser = User(id: user.id, username: user.username, email: user.email)
            isAuthenticated = true
        } catch {
            print("Error decoding user data: \(error.localizedDescription)")
            removeFromKeychain()
        }
    }
    
    private func removeFromKeychain() {
        UserDefaults.standard.removeObject(forKey: tokenKey)
        UserDefaults.standard.removeObject(forKey: userKey)
    }
} 
