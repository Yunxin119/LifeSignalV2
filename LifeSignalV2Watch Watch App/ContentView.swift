//
//  ContentView.swift
//  LifeSignalV2Watch Watch App
//
//  Created by Yunxin Liu on 4/15/25.
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var authService: WatchAuthService
    
    var body: some View {
        Group {
            if authService.isPaired {
                WatchDashboardView()
                    .environmentObject(authService)
            } else {
                WatchPairingView()
                    .environmentObject(authService)
            }
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(WatchAuthService.shared)
}
