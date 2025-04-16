//
//  HomeView.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/15/25.
//

import SwiftUI

struct HomeView: View {
    @EnvironmentObject var authModel: UserAuthModel
    @EnvironmentObject var notificationService: NotificationService
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // Dashboard Tab
            DashboardView()
                .tabItem {
                    Label("Dashboard", systemImage: "heart.text.square.fill")
                }
                .tag(0)
            
            // Profile Tab
            EnhancedProfileView()
                .tabItem {
                    Label("Profile", systemImage: "person.fill")
                }
                .tag(1)
            
            // Settings Tab
            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
                .tag(2)
        }
    }
}

// Enhanced Profile Tab View
struct EnhancedProfileView: View {
    @EnvironmentObject var authModel: UserAuthModel
    @State private var showingWatchPairing = false
    @State private var showingLogoutConfirmation = false
    @State private var isWatchConnected = false  // This would come from the API in a real app
    
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
                
                // Account Settings
                Section(header: Text("ACCOUNT SETTINGS")) {
                    NavigationLink(destination: EmptyView()) {
                        HStack {
                            Image(systemName: "person.crop.circle.badge.exclamationmark")
                                .foregroundColor(.red)
                            Text("Emergency Contacts")
                        }
                    }
                    
                    // TODO: NOTIFICATION SETTINGS - REMOVE IT IF WE ARE NOT GONNA MAKE IT
                    NavigationLink(destination: EmptyView()) {
                        HStack {
                            Image(systemName: "bell")
                                .foregroundColor(.blue)
                            Text("Notification Settings")
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

// Original Profile Tab View - keeping for reference
struct ProfileView: View {
    @EnvironmentObject var authModel: UserAuthModel
    
    var body: some View {
        NavigationStack {
            VStack {
                // Header with welcome message
                VStack(alignment: .leading, spacing: 8) {
                    Text("Welcome back,")
                        .font(.title2)
                        .foregroundColor(.secondary)
                    
                    if let user = authModel.currentUser {
                        Text(user.username)
                            .font(.largeTitle)
                            .fontWeight(.bold)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                
                // Profile Information
                VStack(spacing: 20) {
                    // User Info Card
                    VStack(alignment: .center, spacing: 12) {
                        Image(systemName: "person.circle.fill")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 100, height: 100)
                            .foregroundColor(.blue)
                            .padding(.bottom, 10)
                        
                        if let user = authModel.currentUser {
                            Text(user.username)
                                .font(.title)
                                .fontWeight(.bold)
                            
                            Text(user.email)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        
                        Button(action: {
                            // Edit profile action
                        }) {
                            Text("Edit Profile")
                                .padding(.horizontal, 20)
                                .padding(.vertical, 8)
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                        .padding(.top, 10)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
                    // Linked Devices Section
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Linked Devices")
                            .font(.headline)
                        
                        HStack {
                            Image(systemName: "applewatch")
                                .font(.title2)
                                .foregroundColor(.blue)
                            
                            VStack(alignment: .leading) {
                                Text("Apple Watch")
                                    .fontWeight(.medium)
                                Text("Connected")
                                    .font(.caption)
                                    .foregroundColor(.green)
                            }
                            
                            Spacer()
                            
                            Button(action: {
                                // Device settings action
                            }) {
                                Image(systemName: "ellipsis")
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding()
                        .background(Color(.systemBackground))
                        .cornerRadius(10)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                }
                .padding()
                
                Spacer()
            }
            .navigationTitle("Profile")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        authModel.logout()
                    }) {
                        Text("Logout")
                            .foregroundColor(.blue)
                    }
                }
            }
        }
    }
}

// Settings Tab View
struct SettingsView: View {
    @EnvironmentObject var notificationService: NotificationService
    @State private var showingTestButtons = false
    
    var body: some View {
        NavigationStack {
            Form {
                Section(header: Text("Emergency Contacts")) {
                    NavigationLink(destination: EmptyView()) {
                        Text("Manage Emergency Contacts")
                    }
                }
                
                Section(header: Text("Developer Options")) {
                    Toggle("Show Test Features", isOn: $showingTestButtons)
                    
                    if showingTestButtons {
                        Button("Test Heart Rate Alert") {
                            notificationService.scheduleNotification(
                                title: "High Heart Rate Detected",
                                body: "Heart rate of 130 BPM detected, which is above normal threshold."
                            )
                        }
                        
                        Button("Test Blood Oxygen Alert") {
                            notificationService.scheduleNotification(
                                title: "Low Blood Oxygen Alert",
                                body: "Blood oxygen level of 92% detected, which is below normal range."
                            )
                        }
                        
                        Button("Test Fall Detection") {
                            notificationService.scheduleNotification(
                                title: "Fall Detected!",
                                body: "A potential fall has been detected. Emergency contacts will be notified if no response."
                            )
                        }
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }
}

struct HomeView_Previews: PreviewProvider {
    static var previews: some View {
        HomeView()
            .environmentObject(UserAuthModel())
            .environmentObject(NotificationService())
    }
} 
