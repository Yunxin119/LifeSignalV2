//
//  WatchPairingView.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/15/25.
//

import SwiftUI
import Combine

// 创建一个ViewModel来管理状态和网络请求
class WatchPairingViewModel: ObservableObject {
    @Published var pairingCode: String = ""
    @Published var pairingError: String?
    @Published var showingError = false
    @Published var isPaired = false
    @Published var isLoading = false
    
    private var cancellables = Set<AnyCancellable>()
    private let authModel: UserAuthModel
    
    init(authModel: UserAuthModel) {
        self.authModel = authModel
    }
    
    func generatePairingCode() {
        // 设置加载状态
        isLoading = true
        
        guard let token = authModel.token else {
            pairingError = "Authentication token missing"
            showingError = true
            isLoading = false
            return
        }
        
        // 使用您的API端点
        let url = URL(string: "\(Config.apiBaseURL)/api/pair/generate-code")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: PairingResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink { completion in
                self.isLoading = false
                
                if case .failure(let error) = completion {
                    self.pairingError = error.localizedDescription
                    self.showingError = true
                }
            } receiveValue: { response in
                if response.success {
                    self.pairingCode = response.pairing_code
                } else {
                    self.pairingError = response.error ?? "Failed to generate pairing code"
                    self.showingError = true
                }
            }
            .store(in: &cancellables)
    }
    
    func checkPairingStatus() {
        // 设置加载状态
        isLoading = true
        
        guard let token = authModel.token else {
            pairingError = "Authentication token missing"
            showingError = true
            isLoading = false
            return
        }
        
        let url = URL(string: "\(Config.apiBaseURL)/api/pair/status")!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: PairingStatusResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink { completion in
                self.isLoading = false
                
                if case .failure(let error) = completion {
                    self.pairingError = error.localizedDescription
                    self.showingError = true
                }
            } receiveValue: { response in
                if response.success && response.is_paired {
                    self.isPaired = true
                } else {
                    self.pairingError = "Watch not yet paired. Please enter the code on your watch."
                    self.showingError = true
                }
            }
            .store(in: &cancellables)
    }
}

// 视图结构体
struct WatchPairingView: View {
    @EnvironmentObject private var authModel: UserAuthModel
    @StateObject private var viewModel: WatchPairingViewModel
    
    init() {
        // 创建ViewModel并注入依赖
        _viewModel = StateObject(wrappedValue: WatchPairingViewModel(authModel: UserAuthModel()))
    }
    
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
                    if !viewModel.pairingCode.isEmpty {
                        Text(viewModel.pairingCode)
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
                            
                            if viewModel.isLoading {
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
                if viewModel.pairingCode.isEmpty && !viewModel.isLoading {
                    Button(action: viewModel.generatePairingCode) {
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
                if !viewModel.pairingCode.isEmpty {
                    Button(action: viewModel.checkPairingStatus) {
                        Text("Check Pairing Status")
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.green)
                            .cornerRadius(10)
                            .padding(.horizontal)
                    }
                    .disabled(viewModel.isLoading)
                }
                
                Spacer()
            }
            .padding()
            .onAppear {
                // 在onAppear时调用ViewModel的方法
                viewModel.generatePairingCode()
            }
            .alert(isPresented: $viewModel.showingError) {
                Alert(
                    title: Text("Error"),
                    message: Text(viewModel.pairingError ?? "Unknown error occurred"),
                    dismissButton: .default(Text("OK"))
                )
            }
            .sheet(isPresented: $viewModel.isPaired) {
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
                viewModel.isPaired = false
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
}

// 响应模型
struct PairingResponse: Decodable {
    let success: Bool
    let pairing_code: String
    let expiration_minutes: Int
    let error: String?
}

struct PairingStatusResponse: Decodable {
    let success: Bool
    let is_paired: Bool
    let device_type: String
    let error: String?
}

// preview
struct WatchPairingView_Previews: PreviewProvider {
    static var previews: some View {
        WatchPairingView()
            .environmentObject(UserAuthModel())
    }
}
