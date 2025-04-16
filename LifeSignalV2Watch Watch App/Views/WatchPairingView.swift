import SwiftUI

struct WatchPairingView: View {
    @EnvironmentObject var authService: WatchAuthService
    @State private var pairingCode = ""
    @State private var isSubmitting = false
    @State private var showAlert = false
    @State private var alertMessage = ""
    
    var body: some View {
        ScrollView {
            VStack(spacing: 15) {
                Image(systemName: "applewatch.radiowaves.left.and.right")
                    .font(.system(size: 40))
                    .foregroundColor(.blue)
                    .padding(.bottom, 5)
                
                Text("Connect LifeSignal")
                    .font(.headline)
                    .multilineTextAlignment(.center)
                
                Text("Enter the pairing code from your mobile app")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                
                TextField("Pairing Code", text: $pairingCode)
                    .multilineTextAlignment(.center)
                    .frame(height: 40)
                    .padding()
                    .background(Color.gray.opacity(0.2))
                    .cornerRadius(8)
                
                Button(action: submitPairingCode) {
                    if isSubmitting {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle())
                    } else {
                        Text("Connect")
                            .bold()
                    }
                }
                .buttonStyle(.bordered)
                .tint(.blue)
                .disabled(pairingCode.isEmpty || isSubmitting)
                .padding(.top, 10)
                
                if let error = authService.error {
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.red)
                        .multilineTextAlignment(.center)
                        .padding(.top, 5)
                }
            }
            .padding()
        }
        .alert(isPresented: $showAlert) {
            Alert(
                title: Text("Pairing Error"),
                message: Text(alertMessage),
                dismissButton: .default(Text("OK"))
            )
        }
    }
    
    private func submitPairingCode() {
        guard !pairingCode.isEmpty else { return }
        
        isSubmitting = true
        
        Task {
            let success = await authService.submitPairingCode(pairingCode: pairingCode)
            
            DispatchQueue.main.async {
                isSubmitting = false
                
                if !success && authService.error != nil {
                    alertMessage = authService.error ?? "Failed to pair device"
                    showAlert = true
                }
            }
        }
    }
}

#Preview {
    WatchPairingView()
        .environmentObject(WatchAuthService.shared)
} 