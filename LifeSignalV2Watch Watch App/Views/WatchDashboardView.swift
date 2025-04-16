import SwiftUI

struct WatchDashboardView: View {
    @EnvironmentObject var authService: WatchAuthService
    @StateObject private var healthMonitor = WatchHealthMonitor.shared
    @State private var showSettingsSheet = false
    
    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                HStack {
                    Text("LifeSignal")
                        .font(.headline)
                    
                    Spacer()
                    
                    Button(action: { showSettingsSheet = true }) {
                        Image(systemName: "gear")
                            .foregroundColor(.gray)
                    }
                }
                .padding(.horizontal)
                
                VStack(spacing: 10) {
                    HeartRateView(healthMonitor: healthMonitor)
                    BloodOxygenView(healthMonitor: healthMonitor)
                    EmergencyAlertView(healthMonitor: healthMonitor)
                }
                .padding(.horizontal)
                
                Toggle("Active Monitoring", isOn: .init(
                    get: { healthMonitor.monitoringActive },
                    set: { newValue in
                        if newValue {
                            healthMonitor.startMonitoring()
                        } else {
                            healthMonitor.stopMonitoring()
                        }
                    }
                ))
                .toggleStyle(.switch)
                .tint(.blue)
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.gray)
                        .shadow(color: Color.black.opacity(0.1), radius: 3, x: 0, y: 1)
                )
                .padding(.horizontal)
            }
            .padding(.vertical)
        }
        .onAppear {
            if !healthMonitor.monitoringActive && authService.isValidAuthentication() {
                healthMonitor.startMonitoring()
                
                // Check pairing status when the view appears
                Task {
                    let _ = await authService.checkPairingStatus()
                }
            }
        }
        .sheet(isPresented: $showSettingsSheet) {
            SettingsView()
                .environmentObject(authService)
        }
    }
}

struct SettingsView: View {
    @EnvironmentObject var authService: WatchAuthService
    @Environment(\.presentationMode) var presentationMode
    @State private var showDisconnectAlert = false
    @State private var isUnpairing = false
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                Text("Settings")
                    .font(.headline)
                
                if let userId = authService.userId {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("User ID:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Text(userId)
                            .font(.caption2)
                            .foregroundColor(.primary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.gray.opacity(0.1))
                    )
                }
                
                Button(action: { showDisconnectAlert = true }) {
                    if isUnpairing {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle())
                    } else {
                        Text("Disconnect Device")
                            .foregroundColor(.red)
                    }
                }
                .buttonStyle(.bordered)
                .disabled(isUnpairing)
                .padding(.top)
                
                Button(action: { presentationMode.wrappedValue.dismiss() }) {
                    Text("Close")
                }
                .buttonStyle(.bordered)
            }
            .padding()
        }
        .alert(isPresented: $showDisconnectAlert) {
            Alert(
                title: Text("Disconnect Device"),
                message: Text("Are you sure you want to disconnect this device? You'll need to pair again to use LifeSignal."),
                primaryButton: .destructive(Text("Disconnect")) {
                    unpairDevice()
                },
                secondaryButton: .cancel()
            )
        }
    }
    
    private func unpairDevice() {
        isUnpairing = true
        
        Task {
            let success = await authService.unpairDevice()
            
            DispatchQueue.main.async {
                isUnpairing = false
                
                if success {
                    // The unpairDevice method will call logout, which will reset isPaired
                    // Close the settings view
                    presentationMode.wrappedValue.dismiss()
                }
            }
        }
    }
}

#Preview {
    WatchDashboardView()
        .environmentObject(WatchAuthService.shared)
} 
