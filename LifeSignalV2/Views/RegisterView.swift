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
    
    // New health profile fields
    @State private var age = ""
    @State private var gender = "Prefer not to say"
    @State private var activityLevel = "Moderate"
    @State private var medicalHistory = ""
    @State private var showHealthProfileSection = false
    @State private var preExistingConditions: [String] = []
    
    // Constants for picker options
    let genderOptions = ["Male", "Female", "Non-binary", "Prefer not to say"]
    let activityLevelOptions = ["Sedentary", "Light", "Moderate", "Active", "Very Active", "Athlete"]
    
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
                    
                    if let error = authModel.registrationError {
                        Text(error)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding(.top, -10)
                    }
                    
                    // Health Profile Section Toggle
                    DisclosureGroup("Add Health Profile (Optional)", isExpanded: $showHealthProfileSection) {
                        VStack(spacing: 16) {
                            // Age field
                            TextField("Age", text: $age)
                                .keyboardType(.numberPad)
                                .padding()
                                .background(Color(.systemGray6))
                                .cornerRadius(10)
                            
                            // Gender picker
                            VStack(alignment: .leading) {
                                Text("Gender")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Picker("Gender", selection: $gender) {
                                    ForEach(genderOptions, id: \.self) {
                                        Text($0)
                                    }
                                }
                                .pickerStyle(MenuPickerStyle())
                                .padding()
                                .background(Color(.systemGray6))
                                .cornerRadius(10)
                            }
                            
                            // Activity level picker
                            VStack(alignment: .leading) {
                                Text("Activity Level")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Picker("Activity Level", selection: $activityLevel) {
                                    ForEach(activityLevelOptions, id: \.self) {
                                        Text($0)
                                    }
                                }
                                .pickerStyle(MenuPickerStyle())
                                .padding()
                                .background(Color(.systemGray6))
                                .cornerRadius(10)
                            }
                            
                            // Medical history
                            VStack(alignment: .leading) {
                                Text("Medical History (Optional)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                TextEditor(text: $medicalHistory)
                                    .frame(height: 100)
                                    .padding(4)
                                    .background(Color(.systemGray6))
                                    .cornerRadius(10)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 10)
                                            .stroke(Color(.systemGray4), lineWidth: 1)
                                    )
                                Text("Include relevant health conditions or medications that might affect your heart rate or blood oxygen levels")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                    .padding(.top, 2)
                            }
                            
                            // Pre-existing Conditions
                            VStack(alignment: .leading) {
                                Text("Pre-existing Conditions")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                
                                // Initialize state variables for pre-existing conditions
                                Group {
                                    Toggle("Heart Disease", isOn: Binding<Bool>(
                                        get: { self.preExistingConditions.contains("Heart Disease") },
                                        set: { newValue in
                                            if newValue {
                                                self.preExistingConditions.append("Heart Disease")
                                            } else {
                                                self.preExistingConditions.removeAll { $0 == "Heart Disease" }
                                            }
                                        }
                                    ))
                                    
                                    Toggle("Hypertension", isOn: Binding<Bool>(
                                        get: { self.preExistingConditions.contains("Hypertension") },
                                        set: { newValue in
                                            if newValue {
                                                self.preExistingConditions.append("Hypertension")
                                            } else {
                                                self.preExistingConditions.removeAll { $0 == "Hypertension" }
                                            }
                                        }
                                    ))
                                    
                                    Toggle("COPD", isOn: Binding<Bool>(
                                        get: { self.preExistingConditions.contains("COPD") },
                                        set: { newValue in
                                            if newValue {
                                                self.preExistingConditions.append("COPD")
                                            } else {
                                                self.preExistingConditions.removeAll { $0 == "COPD" }
                                            }
                                        }
                                    ))
                                    
                                    Toggle("Asthma", isOn: Binding<Bool>(
                                        get: { self.preExistingConditions.contains("Asthma") },
                                        set: { newValue in
                                            if newValue {
                                                self.preExistingConditions.append("Asthma")
                                            } else {
                                                self.preExistingConditions.removeAll { $0 == "Asthma" }
                                            }
                                        }
                                    ))
                                    
                                    Toggle("Diabetes", isOn: Binding<Bool>(
                                        get: { self.preExistingConditions.contains("Diabetes") },
                                        set: { newValue in
                                            if newValue {
                                                self.preExistingConditions.append("Diabetes")
                                            } else {
                                                self.preExistingConditions.removeAll { $0 == "Diabetes" }
                                            }
                                        }
                                    ))
                                    
                                    Toggle("Anxiety", isOn: Binding<Bool>(
                                        get: { self.preExistingConditions.contains("Anxiety") },
                                        set: { newValue in
                                            if newValue {
                                                self.preExistingConditions.append("Anxiety")
                                            } else {
                                                self.preExistingConditions.removeAll { $0 == "Anxiety" }
                                            }
                                        }
                                    ))
                                    
                                    Toggle("Sleep Apnea", isOn: Binding<Bool>(
                                        get: { self.preExistingConditions.contains("Sleep Apnea") },
                                        set: { newValue in
                                            if newValue {
                                                self.preExistingConditions.append("Sleep Apnea")
                                            } else {
                                                self.preExistingConditions.removeAll { $0 == "Sleep Apnea" }
                                            }
                                        }
                                    ))
                                    
                                    Toggle("Anemia", isOn: Binding<Bool>(
                                        get: { self.preExistingConditions.contains("Anemia") },
                                        set: { newValue in
                                            if newValue {
                                                self.preExistingConditions.append("Anemia")
                                            } else {
                                                self.preExistingConditions.removeAll { $0 == "Anemia" }
                                            }
                                        }
                                    ))
                                    
                                    Toggle("Tachycardia", isOn: Binding<Bool>(
                                        get: { self.preExistingConditions.contains("Tachycardia") },
                                        set: { newValue in
                                            if newValue {
                                                self.preExistingConditions.append("Tachycardia")
                                            } else {
                                                self.preExistingConditions.removeAll { $0 == "Tachycardia" }
                                            }
                                        }
                                    ))
                                    
                                    Toggle("Bradycardia", isOn: Binding<Bool>(
                                        get: { self.preExistingConditions.contains("Bradycardia") },
                                        set: { newValue in
                                            if newValue {
                                                self.preExistingConditions.append("Bradycardia")
                                            } else {
                                                self.preExistingConditions.removeAll { $0 == "Bradycardia" }
                                            }
                                        }
                                    ))
                                }
                                .toggleStyle(SwitchToggleStyle(tint: .blue))
                                .padding(.horizontal)
                                
                                Text("Select any conditions that may affect your heart rate or blood oxygen levels")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                    .padding(.top, 2)
                            }
                            .padding(.top, 10)
                        }
                        .padding(.top, 10)
                    }
                    
                    // Terms and conditions
                    // HStack {
                    //     Text("By registering, you agree to our ")
                    //         .font(.caption)
                    //         .foregroundColor(.secondary)
                        
                    //     Button("Terms & Conditions") {
                    //         // Show terms and conditions
                    //     }
                    //     .font(.caption)
                    //     .foregroundColor(.blue)
                    // }
                    // .padding(.top, 8)
                    
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
    }
    
    private func performRegistration() {
        // Create a dictionary with health profile data if provided
        var healthProfile: [String: Any]? = nil
        
        if showHealthProfileSection {
            healthProfile = [:]
            
            if !age.isEmpty, let ageValue = Int(age) {
                healthProfile?["age"] = ageValue
            }
            
            if gender != "Prefer not to say" {
                healthProfile?["gender"] = gender
            }
            
            healthProfile?["activity_level"] = activityLevel
            
            if !medicalHistory.isEmpty {
                healthProfile?["medical_history"] = medicalHistory
            }
            
            // Add pre-existing conditions to health profile
            if !preExistingConditions.isEmpty {
                healthProfile?["medical_conditions"] = preExistingConditions
                
                // Also add conditions to a structured medical_history for the HealthService to use
                if healthProfile?["medical_history"] == nil {
                    healthProfile?["medical_history"] = [
                        "conditions": preExistingConditions
                    ]
                } else if var medHistory = healthProfile?["medical_history"] as? String {
                    // If medical history already exists as a string, append conditions
                    let conditionsString = "Pre-existing conditions: " + preExistingConditions.joined(separator: ", ")
                    medHistory += "\n\n" + conditionsString
                    healthProfile?["medical_history"] = medHistory
                }
            }
        }
        
        // Call the updated register method with health profile
        authModel.registerWithHealthProfile(
            username: username,
            email: email,
            password: password,
            confirmPassword: confirmPassword,
            healthProfile: healthProfile
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