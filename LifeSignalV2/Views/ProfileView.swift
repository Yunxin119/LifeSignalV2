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
    // TODO: IMPLEMENT API FOR PAIRING HERE
    @State private var isWatchConnected = false
    
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
                
                // Account Settings
                Section(header: Text("ACCOUNT SETTINGS")) {
                    NavigationLink(destination: EmptyView()) {
                        HStack {
                            Image(systemName: "person.crop.circle.badge.exclamationmark")
                                .foregroundColor(.red)
                            Text("Emergency Contacts")
                        }
                    }
                    
                    // Notification Settings
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

struct EnhancedProfileView_Previews: PreviewProvider {
    static var previews: some View {
        ProfileView()
            .environmentObject(UserAuthModel())
    }
}
