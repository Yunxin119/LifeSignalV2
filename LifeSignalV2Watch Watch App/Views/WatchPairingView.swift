import SwiftUI

struct WatchPairingView: View {
    @EnvironmentObject var authService: WatchAuthService
    @State private var pairingCode = ""
    @State private var isSubmitting = false
    @State private var showAlert = false
    @State private var alertMessage = ""

    let buttons: [[String]] = [
        ["1", "2", "3"],
        ["4", "5", "6"],
        ["7", "8", "9"],
        ["⌫", "0", "✓"]
    ]
    
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

                // 显示当前输入
                Text(pairingCode)
                    .font(.title)
                    .frame(height: 40)
                    .padding()
                    .background(Color.gray.opacity(0.2))
                    .cornerRadius(8)

                // 九键布局
                ForEach(buttons, id: \.self) { row in
                    HStack {
                        ForEach(row, id: \.self) { title in
                            Button(action: {
                                handleButtonPress(title)
                            }) {
                                Text(title)
                                    .frame(width: 45, height: 45)
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .clipShape(Circle())
                                    .font(.headline)
                            }
                            .disabled(isSubmitting)
                        }
                    }
                }

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
    
    private func handleButtonPress(_ title: String) {
        switch title {
        case "⌫":
            if !pairingCode.isEmpty {
                pairingCode.removeLast()
            }
        case "✓":
            submitPairingCode()
        default:
            pairingCode += title
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

