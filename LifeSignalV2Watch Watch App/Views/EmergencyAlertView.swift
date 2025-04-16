import SwiftUI
import WatchKit

struct EmergencyAlertView: View {
    @ObservedObject var healthMonitor: WatchHealthMonitor
    @State private var isAlertActive = false
    @State private var alertCountdown = 30
    @State private var timer: Timer?
    
    var body: some View {
        VStack(spacing: 12) {
            if isAlertActive {
                Text("Emergency Alert")
                    .font(.headline)
                    .foregroundColor(.red)
                
                Text("Sending in \(alertCountdown)s")
                    .font(.caption)
                
                ProgressView(value: Double(30 - alertCountdown), total: 30)
                    .progressViewStyle(LinearProgressViewStyle(tint: .red))
                    .frame(height: 8)
                
                HStack {
                    Button("Cancel") {
                        cancelAlert()
                    }
                    .buttonStyle(.bordered)
                    .tint(.gray)
                    
                    Button("Send Now") {
                        sendAlertNow()
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                }
            } else {
                Button(action: activateAlert) {
                    HStack {
                        Image(systemName: "sos")
                            .font(.headline)
                        Text("Emergency Alert")
                            .font(.headline)
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(.red)
                .padding(.vertical, 8)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(isAlertActive ? Color.red.opacity(0.1) : Color.gray)
                .shadow(color: Color.black.opacity(0.1), radius: 3, x: 0, y: 1)
        )
        .animation(.easeInOut, value: isAlertActive)
    }
    
    private func activateAlert() {
        isAlertActive = true
        alertCountdown = 30
        WKInterfaceDevice.current().play(.notification)
        
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            if alertCountdown > 0 {
                alertCountdown -= 1
                
                // Play haptic feedback every 5 seconds
                if alertCountdown % 5 == 0 {
                    WKInterfaceDevice.current().play(.notification)
                }
                
                // Send alert automatically when countdown ends
                if alertCountdown == 0 {
                    sendAlertNow()
                }
            }
        }
    }
    
    private func cancelAlert() {
        timer?.invalidate()
        timer = nil
        isAlertActive = false
    }
    
    private func sendAlertNow() {
        healthMonitor.triggerManualEmergencyAlert()
        timer?.invalidate()
        timer = nil
        
        // Reset UI after brief pause
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            isAlertActive = false
        }
    }
}

#Preview {
    let monitor = WatchHealthMonitor.shared
    return EmergencyAlertView(healthMonitor: monitor)
        .padding()
        .previewLayout(.sizeThatFits)
} 
