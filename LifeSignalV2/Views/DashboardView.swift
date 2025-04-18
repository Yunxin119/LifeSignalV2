//
//  DashboardView.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/15/25.
//

import SwiftUI
import Combine

struct DashboardView: View {
    @StateObject private var healthService = HealthService()
    @EnvironmentObject private var authModel: UserAuthModel
    @EnvironmentObject private var notificationService: NotificationService
    
    @State private var showingEmergencySheet = false
    @State private var showingHistorySheet = false
    @State private var refreshTrigger = false
    @State private var hasShownAnomalyNotification = false
    @Binding var selectedTab: Int
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // User Status Section
                    VStack(spacing: 8) {
                        HStack {
                            VStack(alignment: .leading) {
                                Text("Elder's Health")
                                    .font(.headline)
                                    .foregroundColor(.secondary)
                                
                                if let lastUpdated = healthService.latestHealthData?.timestamp {
                                    Text("Last updated: \(timeAgo(from: lastUpdated))")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            
                            Spacer()
                            
                            // Refresh button
                            Button(action: {
                                refreshData()
                            }) {
                                Image(systemName: "arrow.clockwise")
                                    .font(.title3)
                            }
                            .disabled(healthService.isLoading)
                        }
                        .padding(.horizontal)
                        
                        if healthService.isLoading {
                            ProgressView()
                                .padding()
                        }
                    }
                    
                    // Current Health Metrics
                    if let healthData = healthService.latestHealthData {
                        healthMetricsView(for: healthData)
                    } else {
                        noDataView()
                    }
                    
                    // Recommendations Card - Always displayed
                    if let healthData = healthService.latestHealthData, !healthData.recommendations.isEmpty {
                        recommendationsView(for: healthData)
                    }
                    
                    // Quick Actions
                    quickActionsView()
                    
                    // Recent History Preview
                    recentHistoryView()
                    
                    // AI Analysis Section
                    if let healthData = healthService.latestHealthData, let aiAnalysis = healthData.aiAnalysis {
                        aiAnalysisView(analysis: aiAnalysis)
                    }
                }
                .padding()
            }
            .refreshable {
                refreshData()
            }
            .navigationTitle("Health Dashboard")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    // Only show emergency icon for critical risk scores
                    if let healthData = healthService.latestHealthData, healthData.riskScore >= 70 {
                        Button(action: {
                            showingEmergencySheet = true
                        }) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.red)
                        }
                    }
                }
            }
            .onAppear {
                refreshData()
            }
            .onChange(of: healthService.anomalyDetected) { newValue in
                if newValue {
                    checkForAnomalies()
                }
            }
            .sheet(isPresented: $showingEmergencySheet) {
                EmergencyContactView()
            }
            .sheet(isPresented: $showingHistorySheet) {
                HealthHistoryView(healthHistory: healthService.healthHistory)
            }
            .alert("Error", isPresented: Binding<Bool>(
                get: { self.healthService.errorMessage != nil },
                set: { if !$0 { self.healthService.errorMessage = nil } }
            )) {
                Button("OK", role: .cancel) {
                    healthService.errorMessage = nil
                }
                if healthService.healthHistory.isEmpty {
                    Button("Create Test Data") {
                        healthService.errorMessage = nil
                        createTestHealthData()
                    }
                }
            } message: {
                if let errorMessage = healthService.errorMessage {
                    Text(errorMessage)
                }
            }
        }
    }
    
    private func createTestHealthData() {
        // Unchanged test data creation code
        if let token = authModel.token {
            let submissionService = HealthDataSubmissionService()
            
            let alert = UIAlertController(
                title: "Creating Test Data",
                message: "Generating multiple health data points for analysis...",
                preferredStyle: .alert
            )
            
            // Add a progress indicator
            let progressView = UIProgressView(progressViewStyle: .default)
            progressView.progress = 0
            progressView.translatesAutoresizingMaskIntoConstraints = false
            alert.view.addSubview(progressView)
            
            NSLayoutConstraint.activate([
                progressView.leadingAnchor.constraint(equalTo: alert.view.leadingAnchor, constant: 20),
                progressView.trailingAnchor.constraint(equalTo: alert.view.trailingAnchor, constant: -20),
                progressView.topAnchor.constraint(equalTo: alert.view.bottomAnchor, constant: -60),
                progressView.heightAnchor.constraint(equalToConstant: 2)
            ])
            
            // Present the alert
            UIApplication.shared.windows.first?.rootViewController?.present(alert, animated: true)
            
            // Create 10 data points to ensure there's enough for analysis
            submissionService.createMultipleTestDataPoints(token: token, count: 10) { success in
                // Update progress
                DispatchQueue.main.async {
                    // Dismiss the alert
                    alert.dismiss(animated: true) {
                        if success {
                            print("Successfully created multiple test health data points")
                            self.refreshData() // Refresh data after creation
                        } else {
                            print("Failed to create test health data points")
                            // Show error message if needed
                            if let error = submissionService.submissionError {
                                self.healthService.errorMessage = error
                            }
                        }
                    }
                }
            }
            
            // Set up a timer to update the progress bar
            Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { timer in
                DispatchQueue.main.async {
                    if submissionService.totalToSubmit > 0 {
                        let progress = Float(submissionService.currentProgress) / Float(submissionService.totalToSubmit)
                        progressView.progress = progress
                        alert.message = "Generating health data points (\(submissionService.currentProgress)/\(submissionService.totalToSubmit))..."
                    }
                    
                    if !submissionService.isSubmitting || submissionService.currentProgress >= submissionService.totalToSubmit {
                        timer.invalidate()
                    }
                }
            }
        }
    }
    
    private func refreshData() {
        if let token = authModel.token {
            healthService.fetchHealthHistory(token: token)
            
            // Reset anomaly notification flag when refreshing data
            hasShownAnomalyNotification = false
        }
    }
    
    // Check for anomalies and send notifications if needed
    private func checkForAnomalies() {
        if healthService.anomalyDetected && !hasShownAnomalyNotification,
           let healthData = healthService.latestHealthData,
           healthData.riskScore >= 40 { // Only notify for moderate or higher risk
            // Send a notification for the anomaly
            notificationService.sendHealthAnomalyNotification(
                heartRate: healthData.heartRate,
                bloodOxygen: healthData.bloodOxygen,
                riskScore: healthData.riskScore
            )
            hasShownAnomalyNotification = true
            
            // Debug info
            print("🔔 Sending anomaly notification for: Heart Rate \(Int(healthData.heartRate)) BPM, Blood Oxygen \(Int(healthData.bloodOxygen))%, Risk: \(healthData.riskScore)")
        }
    }
    
    
    private func timeAgo(from timestamp: String) -> String {
        let date = HealthData(id: "", heartRate: 0, bloodOxygen: 0, timestamp: timestamp, isAnomaly: false, riskScore: 0, recommendations: []).getDate()
        
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }
    
    private func healthMetricsView(for healthData: HealthData) -> some View {
        VStack(spacing: 16) {
            HStack(spacing: 16) {
                // Heart Rate Card - Only highlight as anomalous for moderate+ risk
                let isHighRiskHR = healthData.isAnomaly && healthData.riskScore >= 40 &&
                                 (healthData.heartRate > 100 || healthData.heartRate < 60)
                
                MetricCard(
                    title: "Heart Rate",
                    value: "\(Int(healthData.heartRate))",
                    unit: "BPM",
                    icon: "heart.fill",
                    color: isHighRiskHR ? .red : .pink,
                    isAnomalous: isHighRiskHR
                )
                
                // Blood Oxygen Card - Only highlight as anomalous for moderate+ risk
                let isHighRiskBO = healthData.isAnomaly && healthData.riskScore >= 40 &&
                                 healthData.bloodOxygen < 95
                
                MetricCard(
                    title: "Blood Oxygen",
                    value: String(format: "%.1f", healthData.bloodOxygen),
                    unit: "%",
                    icon: "lungs.fill",
                    color: isHighRiskBO ? .red : .blue,
                    isAnomalous: isHighRiskBO
                )
            }
            
            RiskScoreView(score: healthData.riskScore)
        }
    }
    
    private func noDataView() -> some View {
        VStack(spacing: 10) {
            Image(systemName: "waveform.path.ecg")
                .font(.system(size: 40))
                .foregroundColor(.secondary)
            Text("No health data available")
                .font(.headline)
            Text("Health data will appear here once the elder's device starts sending information")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    // New recommendation view that's always displayed regardless of risk level
    private func recommendationsView(for healthData: HealthData) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                // Icon and title changes based on risk level
                if healthData.riskScore >= 70 {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.red)
                    Text("Critical Health Alert")
                        .font(.headline)
                        .foregroundColor(.red)
                } else if healthData.riskScore >= 40 {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.yellow)
                    Text("Health Alert")
                        .font(.headline)
                } else {
                    Image(systemName: "list.bullet.clipboard")
                        .foregroundColor(.blue)
                    Text("Recommendations")
                        .font(.headline)
                }
                Spacer()
            }
            
            Divider()
            
            ForEach(healthData.recommendations, id: \.self) { recommendation in
                HStack(alignment: .top) {
                    Image(systemName: "circle.fill")
                        .font(.system(size: 6))
                        .padding(.top, 6)
                    Text(recommendation)
                }
            }
            
            // Only show emergency button for high risk (60+)
            if healthData.riskScore >= 60 {
                Spacer(minLength: 10)
                
                Button(action: {
                    showingEmergencySheet = true
                }) {
                    HStack {
                        Image(systemName: "phone.fill")
                        Text("Contact Emergency Services")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
            }
        }
        .padding()
        .background(backgroundColor(for: healthData.riskScore))
        .cornerRadius(12)
        // Only add red border for higher risk
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(healthData.riskScore >= 60 ? Color.red : Color.clear, lineWidth: healthData.riskScore >= 60 ? 2 : 0)
        )
    }
    
    // Helper function to determine background color based on risk
    private func backgroundColor(for riskScore: Double) -> Color {
        if riskScore >= 70 {
            return Color(.systemRed).opacity(0.1)
        } else if riskScore >= 40 {
            return Color(.systemYellow).opacity(0.1)
        } else {
            return Color(.systemGray6)
        }
    }
    
    private func quickActionsView() -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Quick Actions")
                .font(.headline)
            
            HStack(spacing: 12) {
                // Call Elder
                QuickActionButton(
                    title: "Call Elder",
                    icon: "phone.fill",
                    color: .green
                ) {
                    // Implement call functionality
                }
                
                // View History
                QuickActionButton(
                    title: "View History",
                    icon: "chart.xyaxis.line",
                    color: .blue
                ) {
                    showingHistorySheet = true
                }
                
                // Health Trends
                QuickActionButton(
                    title: "Health Trends",
                    icon: "chart.line.uptrend.xyaxis",
                    color: .purple
                ) {
                    // Navigate to Trends Tab
                    selectedTab = 1
                }

                // AI Analysis
                QuickActionButton(
                    title: "AI Report",
                    icon: "brain",
                    color: .indigo
                ) {
                    selectedTab = 2
                }
                
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func recentHistoryView() -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Recent Health History")
                    .font(.headline)
                
                Spacer()
                
                Button("See All") {
                    showingHistorySheet = true
                }
                .font(.subheadline)
                .foregroundColor(.blue)
            }
            
            if healthService.healthHistory.isEmpty {
                Text("No history available")
                    .foregroundColor(.secondary)
                    .padding()
            } else {
                ForEach(Array(healthService.healthHistory.prefix(3))) { dataPoint in
                    HealthHistoryRow(healthData: dataPoint)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func aiAnalysisView(analysis: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "brain")
                    .foregroundColor(.purple)
                Text("AI Health Analysis")
                    .font(.headline)
            }
            
            Text(analysis)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}


// MARK: Helper Views
struct QuickActionButton: View {
    let title: String
    let icon: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack {
                Image(systemName: icon)
                    .font(.title2)
                    .padding(.bottom, 5)
                Text(title)
                    .font(.caption)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(color.opacity(0.1))
            .foregroundColor(color)
            .cornerRadius(10)
        }
    }
}

struct HealthHistoryRow: View {
    let healthData: HealthData
    
    var body: some View {
        HStack {
            // Date and time
            VStack(alignment: .leading) {
                Text(dateFormatter.string(from: healthData.getDate()))
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(timeFormatter.string(from: healthData.getDate()))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // Heart rate
            HStack {
                Image(systemName: "heart.fill")
                    .foregroundColor(.pink)
                    .font(.caption)
                
                Text("\(Int(healthData.heartRate))")
                    .fontWeight(.medium)
            }
            .frame(width: 60)
            
            // Blood oxygen
            HStack {
                Image(systemName: "lungs.fill")
                    .foregroundColor(.blue)
                    .font(.caption)
                
                Text("\(Int(healthData.bloodOxygen))%")
                    .fontWeight(.medium)
            }
            .frame(width: 60)
            
            // Status indicator
            Circle()
                .fill(healthData.isAnomaly ? Color.red : Color.green)
                .frame(width: 12, height: 12)
        }
        .padding(.vertical, 8)
        .padding(.horizontal)
        .background(Color(.systemBackground))
        .cornerRadius(8)
    }
    
    private var dateFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d, yyyy"
        return formatter
    }
    
    private var timeFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a"
        return formatter
    }
}

struct EmergencyContactView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 60))
                    .foregroundColor(.red)
                    .padding()
                
                Text("Emergency Contacts")
                    .font(.title)
                    .fontWeight(.bold)
                
                Text("Tap a contact below to call for assistance:")
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                
                VStack(spacing: 15) {
                    EmergencyContactButton(
                        name: "Emergency Services",
                        role: "911",
                        icon: "phone.fill",
                        color: .red
                    )
                    
                    EmergencyContactButton(
                        name: "Dr. Johnson",
                        role: "Primary Care Physician",
                        icon: "person.fill",
                        color: .blue
                    )
                    
                    EmergencyContactButton(
                        name: "Jane Smith",
                        role: "Family Member",
                        icon: "person.2.fill",
                        color: .green
                    )
                    
                    EmergencyContactButton(
                        name: "Add New Contact",
                        role: "",
                        icon: "plus",
                        color: .gray
                    )
                }
            }
            .padding()
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Close") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct EmergencyContactButton: View {
    let name: String
    let role: String
    let icon: String
    let color: Color
    
    var body: some View {
        Button(action: {
            // Implement calling functionality
        }) {
            HStack {
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundColor(color)
                    .frame(width: 40, height: 40)
                    .background(color.opacity(0.1))
                    .clipShape(Circle())
                
                VStack(alignment: .leading) {
                    Text(name)
                        .font(.headline)
                    
                    if !role.isEmpty {
                        Text(role)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
                
                if name != "Add New Contact" {
                    Image(systemName: "phone.arrow.right.fill")
                        .foregroundColor(color)
                }
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(12)
            .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
        }
    }
}

struct HealthHistoryView: View {
    let healthHistory: [HealthData]
    @Environment(\.dismiss) private var dismiss
    @State private var selectedTimeRange: TimeRange = .week
    
    enum TimeRange: String, CaseIterable, Identifiable {
        case day = "24 Hours"
        case week = "Week"
        case month = "Month"
        
        var id: String { self.rawValue }
    }
    
    var body: some View {
        NavigationStack {
            VStack {
                // Time range picker
                Picker("Time Range", selection: $selectedTimeRange) {
                    ForEach(TimeRange.allCases) { range in
                        Text(range.rawValue).tag(range)
                    }
                }
                .pickerStyle(.segmented)
                .padding()
                
                if healthHistory.isEmpty {
                    ContentUnavailableView {
                        Label("No Health Data", systemImage: "waveform.path.ecg")
                    } description: {
                        Text("Health data will appear here once collected")
                    }
                } else {
                    List {
                        ForEach(filteredHistory) { dataPoint in
                            HealthHistoryDetailRow(healthData: dataPoint)
                        }
                    }
                    .listStyle(.plain)
                }
            }
            .navigationTitle("Health History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    private var filteredHistory: [HealthData] {
        let now = Date()
        return healthHistory.filter { data in
            // Convert string timestamp to Date for comparison
            let dataDate = data.getDate()
            switch selectedTimeRange {
            case .day:
                return dataDate > now.addingTimeInterval(-86400) // 24 hours
            case .week:
                return dataDate > now.addingTimeInterval(-604800) // 7 days
            case .month:
                return dataDate > now.addingTimeInterval(-2592000) // 30 days
            }
        }
    }
}

struct HealthHistoryDetailRow: View {
    let healthData: HealthData
    @State private var isExpanded = false
    
    var body: some View {
        VStack {
            // Main row (always visible)
            Button(action: {
                withAnimation {
                    isExpanded.toggle()
                }
            }) {
                HStack {
                    VStack(alignment: .leading) {
                        Text(dateFormatter.string(from: healthData.getDate()))
                            .font(.headline)
                        
                        Text(timeFormatter.string(from: healthData.getDate()))
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    // Health metrics summary
                    VStack(alignment: .trailing) {
                        HStack {
                            Image(systemName: "heart.fill")
                                .foregroundColor(.pink)
                            Text("\(Int(healthData.heartRate)) BPM")
                                .fontWeight(.medium)
                        }
                        
                        HStack {
                            Image(systemName: "lungs.fill")
                                .foregroundColor(.blue)
                            Text("\(Int(healthData.bloodOxygen))% O₂")
                                .fontWeight(.medium)
                        }
                    }
                    
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .foregroundColor(.secondary)
                        .padding(.leading)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            
            // Expandable details
            if isExpanded {
                VStack(alignment: .leading, spacing: 10) {
                    Divider()
                    
                    // Risk score
                    HStack {
                        Text("Risk Score:")
                            .fontWeight(.medium)
                        
                        Text("\(Int(healthData.riskScore))%")
                            .foregroundColor(riskColor)
                    }
                    
                    // Recommendations
                    if !healthData.recommendations.isEmpty {
                        Text("Recommendations:")
                            .fontWeight(.medium)
                            .padding(.top, 5)
                        
                        ForEach(healthData.recommendations, id: \.self) { recommendation in
                            HStack(alignment: .top) {
                                Image(systemName: "circle.fill")
                                    .font(.system(size: 6))
                                    .padding(.top, 6)
                                
                                Text(recommendation)
                                    .font(.subheadline)
                            }
                        }
                    }
                    
                    // AI Analysis
                    if let analysis = healthData.aiAnalysis {
                        Text("AI Analysis:")
                            .fontWeight(.medium)
                            .padding(.top, 5)
                        
                        Text(analysis)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.vertical)
                .transition(.opacity)
            }
        }
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(.systemBackground))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(healthData.isAnomaly ? Color.red.opacity(0.5) : Color.clear, lineWidth: healthData.isAnomaly ? 1 : 0)
        )
    }
    
    private var dateFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d, yyyy"
        return formatter
    }
    
    private var timeFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a"
        return formatter
    }
    
    private var riskColor: Color {
        switch healthData.riskScore {
        case 0..<20: return .green
        case 20..<40: return .yellow
        case 40..<60: return .orange
        case 60..<80: return .red
        default: return .purple
        }
    }
}

// MARK: - Preview Provider
struct DashboardView_Previews: PreviewProvider {
    static var previews: some View {
        // Create a custom HealthService for preview
        let healthService = HealthService()
        // Add the preview data to the healthData array instead of computed properties
        healthService.healthData = [HealthService.previewData, HealthService.previewAnomalyData]
        
        return Group {
            DashboardView(selectedTab: .constant(0))
                .environmentObject(UserAuthModel())
                .environmentObject(NotificationService())
                .preferredColorScheme(.light)
            
            DashboardView(selectedTab: .constant(0))
                .environmentObject(UserAuthModel())
                .environmentObject(NotificationService())
                .preferredColorScheme(.dark)
        }
    }
}
