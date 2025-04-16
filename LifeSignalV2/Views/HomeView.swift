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
    @State private var showingManualInputSheet = false
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // Dashboard Tab
            DashboardView(selectedTab: $selectedTab)
                .tabItem {
                    Label("Dashboard", systemImage: "heart.text.square.fill")
                }
                .tag(0)
                .overlay(
                    // Manual Input FAB (Floating Action Button)
                    VStack {
                        Spacer()
                        HStack {
                            Spacer()
                            Button(action: {
                                showingManualInputSheet = true
                            }) {
                                Image(systemName: "plus")
                                    .font(.title2)
                                    .foregroundColor(.white)
                                    .frame(width: 56, height: 56)
                                    .background(Color.blue)
                                    .clipShape(Circle())
                                    .shadow(radius: 4)
                            }
                            .padding(.trailing, 20)
                            .padding(.bottom, 20)
                        }
                    }
                )
            
            // Health Trends Tab (New)
            HealthTrendsView()
                .tabItem {
                    Label("Trends", systemImage: "chart.xyaxis.line")
                }
                .tag(1)
                
            // AI Analysis Tab (New)
            AIHealthAnalysisView()
                .tabItem {
                    Label("AI Analysis", systemImage: "brain")
                }
                .tag(2)
            
            // Profile Tab
            ProfileView()
                .tabItem {
                    Label("Profile", systemImage: "person.fill")
                }
                .tag(3)
            
            // Settings Tab
            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
                .tag(4)
        }
        .sheet(isPresented: $showingManualInputSheet) {
            ManualHealthInputView()
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
