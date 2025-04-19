//
//  NotificationService.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/15/25.
//

import SwiftUI
import UserNotifications
import UIKit

class NotificationService: NSObject, ObservableObject, UNUserNotificationCenterDelegate {
    @Published var isPermissionGranted = false
    
    // Add a static shared instance for access from anywhere
    static var shared: NotificationService?
    
    // Add a computed property for notificationsEnabled
    var notificationsEnabled: Bool {
        get { return isPermissionGranted }
    }
    
    private let notificationCenter = UNUserNotificationCenter.current()
    
    override init() {
        super.init()
        
        // Set this class as the delegate for the notification center
        notificationCenter.delegate = self
        
        checkPermission()
        
        // Set the shared instance if it doesn't exist
        if NotificationService.shared == nil {
            NotificationService.shared = self
        }
    }
    
    func checkPermission() {
        UNUserNotificationCenter.current().getNotificationSettings { settings in
            DispatchQueue.main.async {
                self.isPermissionGranted = settings.authorizationStatus == .authorized
            }
        }
    }
    
    func requestPermission() {
        notificationCenter.requestAuthorization(options: [.alert, .badge, .sound]) { success, error in
            DispatchQueue.main.async {
                self.isPermissionGranted = success
                if success {
                    print("✅ Notification permission granted")
                    
                    // Register for remote notifications after getting permission
                    DispatchQueue.main.async {
                        UIApplication.shared.registerForRemoteNotifications()
                    }
                    
                    // Register notification categories
                    self.registerNotificationCategories()
                } else if let error = error {
                    print("❌ Error requesting notification permission: \(error.localizedDescription)")
                } else {
                    print("❌ Notification permission denied")
                }
            }
        }
    }
    
    // Register notification categories and actions
    private func registerNotificationCategories() {
        // Create actions for health alerts
        let dismissAction = UNNotificationAction(
            identifier: "DISMISS_ACTION",
            title: "Dismiss",
            options: .destructive)
        
        let viewDetailsAction = UNNotificationAction(
            identifier: "VIEW_DETAILS_ACTION",
            title: "View Details",
            options: [.foreground])
        
        // Create health alert category
        let healthAlertCategory = UNNotificationCategory(
            identifier: "HEALTH_ALERT",
            actions: [viewDetailsAction, dismissAction],
            intentIdentifiers: [],
            options: [])
        
        // Register the category
        notificationCenter.setNotificationCategories([healthAlertCategory])
        print("🔔 Registered notification categories")
    }
    
    func scheduleNotification(title: String, body: String, timeInterval: TimeInterval = 1) {
        // Check if notification permission is granted
        if !isPermissionGranted {
            print("⚠️ Cannot schedule notification: No permission")
            requestPermission()
            return
        }
        
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = UNNotificationSound.default
        content.categoryIdentifier = "HEALTH_ALERT"
        
        // Create request - use nil trigger for immediate notification
        let identifier = UUID().uuidString
        let request = UNNotificationRequest(
            identifier: identifier, 
            content: content, 
            trigger: timeInterval <= 1 ? nil : UNTimeIntervalNotificationTrigger(timeInterval: timeInterval, repeats: false)
        )
        
        // Add request to notification center
        notificationCenter.add(request) { error in
            if let error = error {
                print("❌ Error scheduling notification: \(error.localizedDescription)")
            } else {
                print("✅ Notification scheduled successfully: \(title)")
            }
        }
    }
    
    func sendLoginSuccessNotification(username: String) {
        scheduleNotification(
            title: "Welcome back!",
            body: "You have successfully logged in as \(username)."
        )
    }
    
    func sendRegistrationSuccessNotification(username: String) {
        scheduleNotification(
            title: "Registration Successful",
            body: "Welcome to LifeSignal, \(username)! Your account has been created successfully."
        )
    }
    
    func sendHealthAnomalyNotification(heartRate: Double, bloodOxygen: Double, riskClass: Int, riskCategory: String) {
        let content = UNMutableNotificationContent()
        
        // Customize title and sound based on risk class
        if riskClass == 2 { // High risk
            content.title = "URGENT Health Alert: High Risk Detected"
            content.sound = UNNotificationSound.defaultCritical
        } else if riskClass == 1 { // Medium risk
            content.title = "Health Alert: Attention Required"
            content.sound = UNNotificationSound.default
        } else { // Low risk
            content.title = "Health Alert: Anomaly Detected"
            content.sound = UNNotificationSound.default
        }
        
        // Create notification body with appropriate urgency
        if riskClass == 2 {
            content.body = "URGENT: Heart Rate: \(Int(heartRate)) bpm, Blood Oxygen: \(Int(bloodOxygen))%. Risk Level: \(riskCategory). Please check detailed AI recommendations immediately."
        } else if riskClass == 1 {
            content.body = "Heart Rate: \(Int(heartRate)) bpm, Blood Oxygen: \(Int(bloodOxygen))%. Risk Level: \(riskCategory). AI recommendations available."
        } else {
            content.body = "Heart Rate: \(Int(heartRate)) bpm, Blood Oxygen: \(Int(bloodOxygen))%. Risk Level: \(riskCategory). Please check the app for details."
        }
        
        content.categoryIdentifier = "HEALTH_ALERT"
        content.userInfo = [
            "heartRate": heartRate,
            "bloodOxygen": bloodOxygen,
            "riskClass": riskClass,
            "riskCategory": riskCategory
        ]
        
        // Create request with nil trigger for immediate delivery
        let identifier = "health-anomaly-\(UUID().uuidString)"
        let request = UNNotificationRequest(identifier: identifier, content: content, trigger: nil)
        
        // Add request to notification center
        notificationCenter.add(request) { error in
            if let error = error {
                print("❌ Error scheduling anomaly notification: \(error.localizedDescription)")
            } else {
                print("✅ Health anomaly notification scheduled successfully")
            }
        }
    }
    
    
    // MARK: - UNUserNotificationCenterDelegate Methods
    
    // This is crucial for showing notifications when the app is in the foreground
    func userNotificationCenter(_ center: UNUserNotificationCenter, 
                               willPresent notification: UNNotification, 
                               withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        // Allow showing notification banners and playing sounds even when app is in foreground
        if #available(iOS 14.0, *) {
            completionHandler([.banner, .sound, .list])
        } else {
            completionHandler([.alert, .sound])
        }
        print("🔔 Notification presented in foreground")
    }
    
    // Handle notification responses
    func userNotificationCenter(_ center: UNUserNotificationCenter, 
                               didReceive response: UNNotificationResponse, 
                               withCompletionHandler completionHandler: @escaping () -> Void) {
        let actionIdentifier = response.actionIdentifier
        
        print("📱 Notification response received: \(actionIdentifier)")
        
        // Handle different actions based on identifier
        completionHandler()
    }
} 
