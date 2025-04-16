//
//  LoginView.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/15/25.
//

import SwiftUI

struct LoginView: View {
    @EnvironmentObject var authModel: UserAuthModel
    @EnvironmentObject var notificationService: NotificationService
    
    @State private var email = ""
    @State private var password = ""
    @State private var isShowingRegister = false
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                // Logo and app name
                VStack(spacing: 12) {
                    Image(systemName: "waveform.path.ecg")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 80, height: 80)
                        .foregroundColor(.blue)
                    
                    Text("Life Signal")
                        .font(.system(size: 32, weight: .bold))
                        .foregroundColor(.primary)
                }
                .padding(.top, 60)
                .padding(.bottom, 40)
                
                // Login form
                VStack(spacing: 20) {
                    TextField("Email", text: $email)
                        .textContentType(.emailAddress)
                        .keyboardType(.emailAddress)
                        .autocapitalization(.none)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)
                    
                    SecureField("Password", text: $password)
                        .textContentType(.password)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)
                    
                    if let error = authModel.loginError {
                        Text(error)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding(.top, -10)
                    }
                    
                    Button(action: performLogin) {
                        if authModel.isLoading {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        } else {
                            Text("Log In")
                                .fontWeight(.semibold)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .disabled(authModel.isLoading)
                    
                    Button("Forgot Password?") {
                        // Handle forgot password
                    }
                    .font(.footnote)
                    .foregroundColor(.blue)
                }
                .padding(.horizontal)
                
                Spacer()
                
                // Register button
                VStack(spacing: 15) {
                    HStack {
                        Text("Don't have an account?")
                            .foregroundColor(.secondary)
                        
                        Button("Register") {
                            isShowingRegister = true
                        }
                        .foregroundColor(.blue)
                        .fontWeight(.semibold)
                    }
                    .font(.callout)
                }
                .padding(.bottom, 30)
            }
            .padding()
            .navigationDestination(isPresented: $isShowingRegister) {
                RegisterView()
            }
        }
    }
    
    private func performLogin() {
        authModel.login(email: email, password: password)
        
        // Delay the notification to ensure authentication completes
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            if authModel.isAuthenticated, let user = authModel.currentUser {
                // Send login success notification
                notificationService.sendLoginSuccessNotification(username: user.username)
            }
        }
    }
}

#Preview {
    LoginView()
        .environmentObject(UserAuthModel())
        .environmentObject(NotificationService())
} 