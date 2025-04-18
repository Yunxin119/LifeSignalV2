//
//  RegisterView.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/15/25.
//

import SwiftUI

struct RegisterView: View {
    @EnvironmentObject var authModel: UserAuthModel
    @EnvironmentObject var notificationService: NotificationService
    @Environment(\.dismiss) private var dismiss
    
    @State private var username = ""
    @State private var email = ""
    @State private var password = ""
    @State private var confirmPassword = ""
    @State private var age: String = ""
    @State private var showHealthConditionsSheet = false
    @State private var selectedHealthConditions: [String] = []
    
    // Common health conditions
    private let healthConditions = [
        "Anxiety", "Depression", "Asthma", "COPD", 
        "Heart Disease", "Hypertension", "Diabetes", 
        "Arthritis", "Cancer", "None"
    ]
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header
                VStack(spacing: 8) {
                    Text("Create Account")
                        .font(.system(size: 28, weight: .bold))
                        .foregroundColor(.primary)
                    
                    Text("Join Life Signal today")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 40)
                .padding(.bottom, 30)
                
                // Registration form
                VStack(spacing: 16) {
                    TextField("Username", text: $username)
                        .textContentType(.username)
                        .autocapitalization(.none)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)
                    
                    TextField("Email", text: $email)
                        .textContentType(.emailAddress)
                        .keyboardType(.emailAddress)
                        .autocapitalization(.none)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)
                    
                    SecureField("Password", text: $password)
                        .textContentType(.newPassword)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)
                    
                    SecureField("Confirm Password", text: $confirmPassword)
                        .textContentType(.newPassword)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)
                    
                    TextField("Age", text: $age)
                        .keyboardType(.numberPad)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)
                    
                    Button(action: {
                        showHealthConditionsSheet = true
                    }) {
                        HStack {
                            Text(selectedHealthConditions.isEmpty ? "Select Health Conditions" : "\(selectedHealthConditions.count) conditions selected")
                                .foregroundColor(selectedHealthConditions.isEmpty ? .secondary : .primary)
                            Spacer()
                            Image(systemName: "chevron.right")
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)
                    }
                    
                    if let error = authModel.registrationError {
                        Text(error)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding(.top, -10)
                    }
                    
                    // Register button
                    Button(action: performRegistration) {
                        if authModel.isLoading {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        } else {
                            Text("Create Account")
                                .fontWeight(.semibold)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .padding(.top, 10)
                    .disabled(authModel.isLoading)
                }
                .padding(.horizontal, 20)
                
                Spacer()
                
                // Back to login button
                Button(action: { dismiss() }) {
                    HStack {
                        Image(systemName: "arrow.left")
                        Text("Back to Login")
                    }
                    .font(.callout)
                    .foregroundColor(.blue)
                }
                .padding(.bottom, 30)
            }
            .padding()
        }
        .navigationBarBackButtonHidden(true)
        .sheet(isPresented: $showHealthConditionsSheet) {
            HealthConditionsSelectionView(
                healthConditions: healthConditions,
                selectedConditions: $selectedHealthConditions
            )
        }
    }
    
    private func performRegistration() {
        authModel.register(
            username: username,
            email: email,
            password: password,
            confirmPassword: confirmPassword,
            healthConditions: selectedHealthConditions.isEmpty ? nil : selectedHealthConditions
        )
        
        // Delay the notification to ensure authentication completes
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            if authModel.isAuthenticated, let user = authModel.currentUser {
                // Send registration success notification
                notificationService.sendRegistrationSuccessNotification(username: user.username)
                dismiss()
            }
        }
    }
}

#Preview {
    NavigationStack {
        RegisterView()
            .environmentObject(UserAuthModel())
            .environmentObject(NotificationService())
    }
} 