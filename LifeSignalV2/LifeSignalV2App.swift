//
//  LifeSignalV2App.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/15/25.
//

import SwiftUI

@main
struct LifeSignalV2App: App {
    let persistenceController = PersistenceController.shared
    
    // Create instances as StateObject to maintain their state across the app lifecycle
    @StateObject private var authModel = UserAuthModel()
    @StateObject private var notificationService = NotificationService()
    @StateObject private var networkMonitor = NetworkMonitor()

    var body: some Scene {
        WindowGroup {
            MainContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
                .environmentObject(authModel)
                .environmentObject(notificationService)
                .environmentObject(networkMonitor)
                .onAppear {
                    notificationService.requestPermission()
                }
        }
    }
}

struct MainContentView: View {
    @EnvironmentObject var authModel: UserAuthModel
    @EnvironmentObject var networkMonitor: NetworkMonitor
    @State private var showingConnectionAlert = false
    
    var body: some View {
        ZStack {
            // Main content based on authentication state
            if authModel.isAuthenticated {
                HomeView()
            } else {
                LoginView()
            }
            
            // Connection status overlay
            if !networkMonitor.isConnected {
                VStack {
                    Spacer()
                    HStack {
                        Image(systemName: "wifi.slash")
                        Text("No internet connection")
                    }
                    .font(.footnote)
                    .padding(8)
                    .background(Color(.systemRed))
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .padding(.bottom, 10)
                .transition(.move(edge: .bottom))
                .animation(.easeInOut, value: networkMonitor.isConnected)
            }
        }
        .onAppear {
            // Check server connection on appear
            checkServerConnection()
        }
        .alert(isPresented: $showingConnectionAlert) {
            Alert(
                title: Text("Server Connection Error"),
                message: Text("Unable to connect to the Life Signal server. Please check your internet connection and try again."),
                dismissButton: .default(Text("OK"))
            )
        }
    }
    
    private func checkServerConnection() {
        if networkMonitor.isConnected {
            let cancellable = networkMonitor.checkServerConnection()
                .sink { isConnected in
                    if !isConnected {
                        showingConnectionAlert = true
                    }
                }
            
            // Use the authModel's store method instead of directly accessing cancellables
            authModel.store(cancellable)
        }
    }
}
