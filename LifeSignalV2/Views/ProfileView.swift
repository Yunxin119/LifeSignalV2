//
//  ProfileView.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/16/25.
//


import SwiftUI

struct ProfileView: View {
    @EnvironmentObject var authModel: UserAuthModel
    @State private var showingWatchPairing = false
    @State private var showingLogoutConfirmation = false
    @State private var showingHealthConditionsEditor = false
    @State private var showingHealthConditionsInfo = false
    // TODO: IMPLEMENT API FOR PAIRING HERE
    @State private var isWatchConnected = false
    
    // Common health conditions for editing
    private let healthConditions = [
        "Anxiety", "Depression", "Asthma", "COPD", 
        "Heart Disease", "Hypertension", "Diabetes", 
        "Arthritis", "Cancer", "None"
    ]
    
    var body: some View {
        NavigationStack {
            List {
                // Profile Header
                Section {
                    if let user = authModel.currentUser {
                        VStack(alignment: .center, spacing: 15) {
                            // Profile Image/Initials
                            ZStack {
                                Circle()
                                    .fill(Color(.systemGray5))
                                    .frame(width: 100, height: 100)
                                
                                // Use first letter of username as initials
                                Text(String(user.username.prefix(1)).uppercased())
                                    .font(.system(size: 36, weight: .bold))
                                    .foregroundColor(.blue)
                            }
                            
                            // User Name
                            Text(user.username)
                                .font(.title2)
                                .fontWeight(.bold)
                            
                            // User Email
                            Text(user.email)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 20)
                    }
                }
                
                // Health Conditions
                Section(header: Text("HEALTH CONDITIONS")) {
                    if let user = authModel.currentUser {
                        if let conditions = user.healthConditions, !conditions.isEmpty {
                            ForEach(conditions, id: \.self) { condition in
                                Text(condition)
                            }
                            
                            Button(action: {
                                showingHealthConditionsInfo = true
                            }) {
                                HStack {
                                    Image(systemName: "info.circle")
                                    Text("How Health Conditions Affect Analysis")
                                }
                                .foregroundColor(.blue)
                            }
                        } else {
                            Text("No health conditions specified")
                                .foregroundColor(.secondary)
                        }
                        
                        Button(action: {
                            showingHealthConditionsEditor = true
                        }) {
                            HStack {
                                Image(systemName: "pencil")
                                Text("Edit Health Conditions")
                            }
                            .foregroundColor(.blue)
                        }
                    }
                }
                
                // Watch Connection
                Section(header: Text("DEVICE CONNECTION")) {
                    if isWatchConnected {
                        HStack {
                            Image(systemName: "applewatch")
                                .foregroundColor(.green)
                            Text("Apple Watch")
                            Spacer()
                            Text("Connected")
                                .foregroundColor(.green)
                        }
                    } else {
                        Button(action: {
                            showingWatchPairing = true
                        }) {
                            HStack {
                                Image(systemName: "applewatch")
                                    .foregroundColor(.blue)
                                Text("Connect Apple Watch")
                                    .foregroundColor(.primary)
                            }
                        }
                    }
                }
                
                // Health Data
                Section(header: Text("HEALTH INSIGHTS")) {
                    NavigationLink(destination: HealthTrendsView()) {
                        HStack {
                            Image(systemName: "chart.xyaxis.line")
                                .foregroundColor(.purple)
                            Text("View Health Trends")
                        }
                    }
                    
                    NavigationLink(destination: AIHealthAnalysisView()) {
                        HStack {
                            Image(systemName: "brain")
                                .foregroundColor(.indigo)
                            Text("AI Health Analysis")
                        }
                    }
                }
                
                // App Settings
                Section(header: Text("APP SETTINGS")) {
                    NavigationLink(destination: EmptyView()) {
                        HStack {
                            Image(systemName: "questionmark.circle")
                                .foregroundColor(.blue)
                            Text("Help & Support")
                        }
                    }
                    
                    NavigationLink(destination: EmptyView()) {
                        HStack {
                            Image(systemName: "info.circle")
                                .foregroundColor(.blue)
                            Text("About LifeSignal")
                        }
                    }
                }
                
                // Logout
                Section {
                    Button(action: {
                        showingLogoutConfirmation = true
                    }) {
                        HStack {
                            Spacer()
                            Text("Sign Out")
                                .foregroundColor(.red)
                            Spacer()
                        }
                    }
                }
            }
            .listStyle(InsetGroupedListStyle())
            .navigationTitle("Profile")
            .sheet(isPresented: $showingWatchPairing) {
                WatchPairingView()
                    .environmentObject(authModel)
            }
            .sheet(isPresented: $showingHealthConditionsEditor) {
                HealthConditionsSelectionView(
                    healthConditions: healthConditions,
                    selectedConditions: Binding(
                        get: { authModel.currentUser?.healthConditions ?? [] },
                        set: { newValue in
                            // Save health conditions when done
                            authModel.updateHealthConditions(healthConditions: newValue)
                        }
                    )
                )
            }
            .sheet(isPresented: $showingHealthConditionsInfo) {
                HealthConditionsInfoView(healthConditions: authModel.currentUser?.healthConditions ?? [])
            }
            .alert(isPresented: $showingLogoutConfirmation) {
                Alert(
                    title: Text("Sign Out"),
                    message: Text("Are you sure you want to sign out?"),
                    primaryButton: .destructive(Text("Sign Out")) {
                        authModel.logout()
                    },
                    secondaryButton: .cancel()
                )
            }
        }
    }
}

struct HealthConditionsInfoView: View {
    let healthConditions: [String]
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("How Your Health Conditions Affect Analysis")
                        .font(.headline)
                        .padding(.bottom, 5)
                    
                    Text("LifeSignal adjusts its health analysis based on your specified health conditions to provide more personalized insights.")
                        .padding(.bottom, 10)
                    
                    if healthConditions.isEmpty {
                        Text("You haven't specified any health conditions yet. Adding your conditions helps us provide more accurate health analysis.")
                            .foregroundColor(.secondary)
                    } else {
                        VStack(alignment: .leading, spacing: 15) {
                            ForEach(healthConditions, id: \.self) { condition in
                                VStack(alignment: .leading, spacing: 5) {
                                    Text(condition)
                                        .font(.subheadline)
                                        .fontWeight(.medium)
                                    
                                    Text(conditionImpactDescription(condition))
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                .padding(.vertical, 5)
                            }
                        }
                    }
                    
                    Spacer()
                }
                .padding()
            }
            .navigationTitle("Health Conditions Impact")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    func conditionImpactDescription(_ condition: String) -> String {
        let lowercasedCondition = condition.lowercased()
        
        switch lowercasedCondition {
        case let x where x.contains("anxiety"):
            return "We adjust heart rate thresholds as anxiety can naturally elevate heart rate readings without indicating physical health issues."
        case let x where x.contains("copd") || x.contains("emphysema") || x.contains("bronchitis"):
            return "We use lower baseline blood oxygen thresholds since COPD patients typically have lower blood oxygen saturation levels."
        case let x where x.contains("heart") || x.contains("hypertension") || x.contains("cardiovascular"):
            return "We provide more careful monitoring of heart-related metrics and apply stricter risk assessment for abnormal readings."
        case let x where x.contains("diabetes"):
            return "We track glucose-related impacts on cardiovascular metrics and adjust risk calculations for heart rate variability."
        case let x where x.contains("asthma"):
            return "We account for respiratory impacts on blood oxygen readings and consider additional factors during symptom episodes."
        case let x where x.contains("arthritis"):
            return "We consider mobility limitations that may affect health data patterns and activity-related metrics."
        case let x where x.contains("cancer"):
            return "We account for treatment impacts on vital signs and provide more cautious interpretation of readings."
        case let x where x.contains("depression"):
            return "We consider potential impacts on activity patterns and account for medication effects on vital signs."
        case let x where x.contains("none"):
            return "Standard health metrics and thresholds are applied without condition-specific adjustments."
        default:
            return "We adjust our analysis to consider potential impacts of this condition on your health metrics."
        }
    }
}

struct EnhancedProfileView_Previews: PreviewProvider {
    static var previews: some View {
        ProfileView()
            .environmentObject(UserAuthModel())
    }
}
