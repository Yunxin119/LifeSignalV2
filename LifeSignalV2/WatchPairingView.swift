//
//  WatchPairingView.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/15/25.
//

import SwiftUI
import Combine

struct WatchPairingView: View {
    @EnvironmentObject private var authModel: UserAuthModel
    @State private var pairingCode: String = ""
    @State private var pairingError: String?
    @State private var showingError = false
    @State private var isPaired = false
    @State private var isLoading = false
    
    private var cancellables = Set<AnyCancellable>()
    
    var body: some View {
        ScrollView {
            VStack(spacing: 30) {
                // Header
                VStack(spacing: 10) {
                    Text("Connect Your Watch")
                        .font(.system(size: 28, weight: .bold))
                    
                    Text("Enter this code on your LifeSignal watch app")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                
                // Pairing Code Display
                VStack {
                    if !pairingCode.isEmpty {
                        Text(pairingCode)
                            .font(.system(size: 48, weight: .bold, design: .monospaced))
                            .kerning(10)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color(.systemGray6))
                            .cornerRadius(10)
                            .shadow(radius: 2)
                    } else {
                        // Placeholder or loading indicator
                        ZStack {
                            Rectangle()
                                .fill(Color(.systemGray6))
                                .frame(height: 100)
                                .cornerRadius(10)
                            
                            if isLoading {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle())
                            } else {
                                Text("Generating code...")
                                    .foregroundColor(.gray)
                            }
                        }
                    }
                }
                .padding()
                
                // Instructions
                VStack(alignment: .leading, spacing: 15) {
                    Text("Pairing Instructions:")
                        .font(.headline)
                    
                    HStack(alignment: .top) {
                        Text("1.")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        Text("Open the LifeSignal app on your Apple Watch")
                            .font(.subheadline)
                    }
                    
                    HStack(alignment: .top) {
                        Text("2.")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        Text("Tap on 'Pair with Phone' in the watch app")
                            .font(.subheadline)
                    }
                    
                    HStack(alignment: .top) {
                        Text("3.")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        Text("Enter the 6-digit code shown above on your watch")
                            .font(.subheadline)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .padding(.horizontal)
                
                // Retry Button (if needed)
                if pairingCode.isEmpty && !isLoading {
                    Button(action: generatePairingCode) {
                        Text("Generate New Code")
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                            .padding(.horizontal)
                    }
                }
                
                // Check Pairing Status button
                if !pairingCode.isEmpty {
                    Button(action: checkPairingStatus) {
                        Text("Check Pairing Status")
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.green)
                            .cornerRadius(10)
                            .padding(.horizontal)
                    }
                    .disabled(isLoading)
                }
                
                Spacer()
            }
            .padding()
            .onAppear(perform: generatePairingCode)
            .alert(isPresented: $showingError) {
                Alert(
                    title: Text("Error"),
                    message: Text(pairingError ?? "Unknown error occurred"),
                    dismissButton: .default(Text("OK"))
                )
            }
            .sheet(isPresented: $isPaired) {
                pairingSuccessView
            }
        }
    }
    
    private var pairingSuccessView: some View {
        VStack(spacing: 20) {
            Image(systemName: "checkmark.circle.fill")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 100, height: 100)
                .foregroundColor(.green)
                .padding()
            
            Text("Watch Paired Successfully!")
                .font(.title)
                .fontWeight(.bold)
            
            Text("Your watch is now paired with this LifeSignal account. Health data will be shared securely between devices.")
                .multilineTextAlignment(.center)
                .padding()
            
            Button(action: {
                isPaired = false
            }) {
                Text("Continue")
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(10)
                    .padding(.horizontal)
            }
            .padding()
        }
        .padding()
    }
    
    private func generatePairingCode() {
        // Set loading state
        isLoading = true
        
        // TODO: CHANGE THIS TO REAL API CALL IN APP
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            // Generate a random 6-digit code
            let code = String(format: "%06d", Int.random(in: 100000...999999))
            pairingCode = code
            isLoading = false
        }
        
        // WE MIGHT DO IT LIKE THIS - I'S NOT SURE IT DEPENDS ON BACKEND SITUATION
        /*
        guard let token = authModel.token else {
            pairingError = "Authentication token missing"
            showingError = true
            isLoading = false
            return
        }
        
        let url = URL(string: "\(Config.apiBaseURL)/api/watch/generate-pairing-code")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: PairingResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink { completion in
                isLoading = false
                
                if case .failure(let error) = completion {
                    pairingError = error.localizedDescription
                    showingError = true
                }
            } receiveValue: { response in
                if let code = response.code {
                    pairingCode = code
                } else {
                    pairingError = response.error ?? "Failed to generate pairing code"
                    showingError = true
                }
            }
            .store(in: &cancellables)
        */
    }
    
    private func checkPairingStatus() {
        // Set loading state
        isLoading = true
        
        // TODO: CHECK OUT WHETHER IT IS CONNECTED TO THE WATCH
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            // For demo purposes, simulate a successful pairing
            isPaired = true
            isLoading = false
        }
        
        // WE MAY DO IT LIKE THIS:
        /*
        guard let token = authModel.token else {
            pairingError = "Authentication token missing"
            showingError = true
            isLoading = false
            return
        }
        
        let url = URL(string: "\(Config.apiBaseURL)/api/watch/check-pairing-status")!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: PairingStatusResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink { completion in
                isLoading = false
                
                if case .failure(let error) = completion {
                    pairingError = error.localizedDescription
                    showingError = true
                }
            } receiveValue: { response in
                if response.isPaired {
                    isPaired = true
                } else {
                    pairingError = "Watch not yet paired. Please enter the code on your watch."
                    showingError = true
                }
            }
            .store(in: &cancellables)
        */
    }
}

// preview
struct WatchPairingView_Previews: PreviewProvider {
    static var previews: some View {
        WatchPairingView()
            .environmentObject(UserAuthModel())
    }
} 
