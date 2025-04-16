//
//  SettingsView.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/16/25.
//

import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var notificationService: NotificationService
    @State private var showingTestButtons = false
    @State private var showingDemoOptions = false
    @State private var notificationsEnabled = false
    
    var body: some View {
        NavigationStack {
            Form {
                Section(header: Text("Notifications")) {
                    Toggle("Enable Notifications", isOn: $notificationsEnabled)
                        .onChange(of: notificationsEnabled) { newValue in
                            if newValue {
                                notificationService.requestPermission()
                            }
                        }
                        .onAppear {
                            notificationsEnabled = notificationService.notificationsEnabled
                        }
                }
                
                Section(header: Text("Emergency Contacts")) {
                    NavigationLink(destination: EmptyView()) {
                        Text("Manage Emergency Contacts")
                    }
                }
                
                Section(header: Text("Demo Features")) {
                    Toggle("Show Demo Options", isOn: $showingDemoOptions)
                    
                    if showingDemoOptions {
                        NavigationLink(destination: DemoControlTestView()) {
                            HStack {
                                Image(systemName: "hammer.fill")
                                    .foregroundColor(.orange)
                                Text("Open Demo Control Panel")
                            }
                        }
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
                        
                        Button("Test AI Analysis Notification") {
                            notificationService.scheduleNotification(
                                title: "New Health Insights Available",
                                body: "Our AI has analyzed your recent health data and has personalized recommendations for you."
                            )
                        }
                    }
                }
                
                Section(header: Text("About")) {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0 (Demo)")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Backend API")
                        Spacer()
                        Text(Config.apiBaseURL)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }
}
