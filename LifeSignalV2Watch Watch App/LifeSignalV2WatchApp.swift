//
//  LifeSignalV2WatchApp.swift
//  LifeSignalV2Watch Watch App
//
//  Created by Yunxin Liu on 4/15/25.
//

import SwiftUI

@main
struct LifeSignalV2Watch_Watch_AppApp: App {
    @StateObject private var authService = WatchAuthService.shared
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(authService)
        }
    }
}
