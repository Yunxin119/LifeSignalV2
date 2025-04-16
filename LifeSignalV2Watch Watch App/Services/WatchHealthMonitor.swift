import Foundation
import HealthKit
import WatchKit

class WatchHealthMonitor: ObservableObject {
    static let shared = WatchHealthMonitor()
    private let healthStore = HKHealthStore()
    private let authService = WatchAuthService.shared
    
    @Published var currentHeartRate: Double?
    @Published var currentBloodOxygen: Double?
    @Published var isHeartRateAbnormal = false
    @Published var isBloodOxygenAbnormal = false
    @Published var monitoringActive = false
    
    // Set of current health data
    private var currentHealthData: HealthData {
        HealthData(
            heartRate: currentHeartRate,
            bloodOxygen: currentBloodOxygen,
            hasFallDetected: false,
            location: nil // Location would be added if needed
        )
    }
    
    // Health monitoring timer
    private var healthUpdateTimer: Timer?
    private let timerInterval: TimeInterval = 300 // 5 minutes
    
    private init() {}
    
    func startMonitoring() {
        requestAuthorization { [weak self] success in
            guard let self = self, success else { return }
            
            DispatchQueue.main.async {
                self.monitoringActive = true
            }
            
            self.startHeartRateMonitoring()
            self.startBloodOxygenMonitoring()
            self.startFallDetection()
            self.startHealthUpdateTimer()
        }
    }
    
    func stopMonitoring() {
        DispatchQueue.main.async {
            self.monitoringActive = false
            self.healthUpdateTimer?.invalidate()
            self.healthUpdateTimer = nil
        }
    }
    
    private func requestAuthorization(completion: @escaping (Bool) -> Void) {
        // Define the health data types your app will read
        let typesToRead: Set<HKSampleType> = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .oxygenSaturation)!
        ]
        
        // Request authorization from HealthKit
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { success, error in
            if !success {
                print("HealthKit authorization failed: \(error?.localizedDescription ?? "unknown error")")
            }
            completion(success)
        }
    }
    
    private func startHeartRateMonitoring() {
        guard let heartRateType = HKObjectType.quantityType(forIdentifier: .heartRate) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: heartRateType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            self?.processHeartRateSamples(samples)
        }
        
        query.updateHandler = { [weak self] query, samples, deleted, anchor, error in
            self?.processHeartRateSamples(samples)
        }
        
        healthStore.execute(query)
    }
    
    private func startBloodOxygenMonitoring() {
        guard let bloodOxygenType = HKObjectType.quantityType(forIdentifier: .oxygenSaturation) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: bloodOxygenType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            self?.processBloodOxygenSamples(samples)
        }
        
        query.updateHandler = { [weak self] query, samples, deleted, anchor, error in
            self?.processBloodOxygenSamples(samples)
        }
        
        healthStore.execute(query)
    }
    
    private func startFallDetection() {
        // For watchOS 9.0 and above we would implement fall detection
        // This is a placeholder for now
        if #available(watchOS 9.0, *) {
            print("Fall detection available on this device")
            // Implementation would go here
        }
    }
    
    private func startHealthUpdateTimer() {
        DispatchQueue.main.async {
            self.healthUpdateTimer = Timer.scheduledTimer(
                withTimeInterval: self.timerInterval,
                repeats: true
            ) { [weak self] _ in
                self?.sendPeriodicHealthUpdate()
            }
        }
    }
    
    private func sendPeriodicHealthUpdate() {
        // Only send if we have at least one health metric
        if let _ = currentHeartRate, authService.isValidAuthentication() {
            Task {
                _ = await authService.uploadHealthData(currentHealthData)
            }
        }
    }
    
    private func processHeartRateSamples(_ samples: [HKSample]?) {
        guard let samples = samples as? [HKQuantitySample], let latestSample = samples.last else { return }
        
        let heartRate = latestSample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
        
        DispatchQueue.main.async {
            self.currentHeartRate = heartRate
            self.isHeartRateAbnormal = HealthThresholds.isHeartRateAbnormal(heartRate)
            
            // Send data to server if abnormal
            if self.isHeartRateAbnormal && self.authService.isValidAuthentication() {
                Task {
                    _ = await self.authService.uploadHealthData(self.currentHealthData)
                }
                
                // Notify the app
                self.notifyAbnormalHeartRate(heartRate)
            }
        }
    }
    
    private func processBloodOxygenSamples(_ samples: [HKSample]?) {
        guard let samples = samples as? [HKQuantitySample], let latestSample = samples.last else { return }
        
        let bloodOxygen = latestSample.quantity.doubleValue(for: HKUnit.percent()) * 100 // Convert to percentage
        
        DispatchQueue.main.async {
            self.currentBloodOxygen = bloodOxygen
            self.isBloodOxygenAbnormal = HealthThresholds.isBloodOxygenAbnormal(bloodOxygen)
            
            // Send data to server if abnormal
            if self.isBloodOxygenAbnormal && self.authService.isValidAuthentication() {
                Task {
                    _ = await self.authService.uploadHealthData(self.currentHealthData)
                }
                
                // Notify the app
                self.notifyAbnormalBloodOxygen(bloodOxygen)
            }
        }
    }
    
    func triggerManualEmergencyAlert() {
        WKInterfaceDevice.current().play(.notification)
        
        if authService.isValidAuthentication() {
            Task {
                _ = await authService.sendEmergencyAlert(healthData: currentHealthData)
            }
        }
        
        NotificationCenter.default.post(
            name: Notification.Name.emergencyAlertTriggered,
            object: nil,
            userInfo: ["healthData": currentHealthData]
        )
    }
    
    private func notifyAbnormalHeartRate(_ value: Double) {
        NotificationCenter.default.post(
            name: Notification.Name.abnormalHeartRateDetected,
            object: nil,
            userInfo: ["heartRate": value]
        )
    }
    
    private func notifyAbnormalBloodOxygen(_ value: Double) {
        NotificationCenter.default.post(
            name: Notification.Name.abnormalBloodOxygenDetected,
            object: nil,
            userInfo: ["bloodOxygen": value]
        )
    }
}

// MARK: - Notification Names
extension Notification.Name {
    static let abnormalHeartRateDetected = Notification.Name("abnormalHeartRateDetected")
    static let abnormalBloodOxygenDetected = Notification.Name("abnormalBloodOxygenDetected")
    static let fallDetected = Notification.Name("fallDetected")
    static let emergencyAlertTriggered = Notification.Name("emergencyAlertTriggered")
} 