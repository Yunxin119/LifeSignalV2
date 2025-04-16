//
//  NetworkMonitor.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/15/25.
//

import Foundation
import Network
import Combine

class NetworkMonitor: ObservableObject {
    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "NetworkMonitor")
    
    @Published var isConnected = true
    @Published var connectionType: ConnectionType = .unknown
    
    enum ConnectionType {
        case wifi
        case cellular
        case ethernet
        case unknown
    }
    
    init() {
        monitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                self?.isConnected = path.status == .satisfied
                self?.determineConnectionType(path)
            }
        }
        monitor.start(queue: queue)
    }
    
    deinit {
        monitor.cancel()
    }
    
    private func determineConnectionType(_ path: NWPath) {
        if path.usesInterfaceType(.wifi) {
            connectionType = .wifi
        } else if path.usesInterfaceType(.cellular) {
            connectionType = .cellular
        } else if path.usesInterfaceType(.wiredEthernet) {
            connectionType = .ethernet
        } else {
            connectionType = .unknown
        }
    }
    
    func checkServerConnection() -> AnyPublisher<Bool, Never> {
        // Use the health check endpoint to verify server connection
        guard let url = URL(string: Config.apiBaseURL + "/health") else {
            return Just(false).eraseToAnyPublisher()
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 5 // Short timeout for health check
        
        return URLSession.shared.dataTaskPublisher(for: request)
            .map { data, response -> Bool in
                guard let httpResponse = response as? HTTPURLResponse else {
                    return false
                }
                return (200...299).contains(httpResponse.statusCode)
            }
            .replaceError(with: false)
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
} 