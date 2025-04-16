//
//  AIHealthAnalysisView.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/16/25.
//
import SwiftUI
import Combine

struct AIHealthAnalysisView: View {
    @StateObject private var analysisService = AIHealthAnalysisService()
    @EnvironmentObject private var authModel: UserAuthModel
    
    @State private var timeRange: Int = 30
    
    var body: some View {
        ScrollView {
            VStack(spacing: 25) {
                // Time period picker
                Picker("Time Period", selection: $timeRange) {
                    Text("7 Days").tag(7)
                    Text("30 Days").tag(30)
                    Text("90 Days").tag(90)
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding(.horizontal)
                .onChange(of: timeRange) { _ in
                    refreshData()
                }
                
                if analysisService.isLoading {
                    VStack(spacing: 20) {
                        ProgressView()
                            .scaleEffect(1.5)
                        
                        Text("Analyzing your health data...")
                            .font(.headline)
                        
                        Text("Our AI is processing your health patterns and preparing personalized insights.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 60)
                } else if let analysis = analysisService.analysis {
                    // Status Overview Card
                    statusOverviewCard(analysis)
                    
                    // Assessment Card
                    assessmentCard(analysis)
                    
                    // Recommendations Card
                    recommendationsCard(analysis)
                    
                    // Concerns Card
                    if !analysis.potentialConcerns.isEmpty {
                        concernsCard(analysis)
                    }
                    
                    // Medical Attention Card
                    medicalAttentionCard(analysis)
                    
                    // Follow-up Card
                    followUpCard(analysis)
                } else {
                    // No data view
                    VStack(spacing: 20) {
                        Image(systemName: "brain")
                            .font(.system(size: 60))
                            .foregroundColor(.purple)
                        
                        Text("No AI Analysis Available")
                            .font(.title3)
                            .fontWeight(.bold)
                        
                        Text("AI health analysis will appear here once sufficient data is collected for analysis.")
                            .multilineTextAlignment(.center)
                            .foregroundColor(.secondary)
                            .padding(.horizontal)
                        
                        // Add a more prominent way to add test data
                        Button(action: {
                            // Create test data
                            createTestDataForAnalysis()
                        }) {
                            Label("Generate Test Data", systemImage: "waveform.path.ecg")
                                .padding()
                                .frame(maxWidth: .infinity)
                                .background(Color.purple.opacity(0.1))
                                .foregroundColor(.purple)
                                .cornerRadius(10)
                        }
                        .padding(.top)
                        
                        Button(action: refreshData) {
                            Label("Refresh Analysis", systemImage: "arrow.clockwise")
                                .padding()
                                .background(Color.blue.opacity(0.1))
                                .foregroundColor(.blue)
                                .cornerRadius(10)
                        }
                        .padding(.top, 5)
                    }
                    .padding(.vertical, 60)
                    .padding(.horizontal)
                }
            }
            .padding(.vertical)
            .navigationTitle("AI Health Analysis")
            .onAppear {
                refreshData()
            }
            .alert(isPresented: .init(
                get: { analysisService.errorMessage != nil },
                set: { if !$0 { analysisService.errorMessage = nil }}
            )) {
                Alert(
                    title: Text("Error"),
                    message: Text(analysisService.errorMessage ?? "Unknown error"),
                    dismissButton: .default(Text("OK"))
                )
            }
        }
    }
    
    private func refreshData() {
        if let token = authModel.token {
            print("Fetching AI health analysis with time range: \(timeRange) days...")
            analysisService.fetchAIAnalysis(token: token, days: timeRange)
            
            // Add listener to display raw response for debugging
            if let rawResponseData = analysisService.rawResponse,
               let jsonString = String(data: rawResponseData, encoding: .utf8) {
                print("Raw AI Analysis Response: \(jsonString)")
            }
        }
    }
    
    private func createTestDataForAnalysis() {
        if let token = authModel.token {
            let submissionService = HealthDataSubmissionService()
            
            // Show a loading alert
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
            
            // Set up constraints for the progress view
            NSLayoutConstraint.activate([
                progressView.leadingAnchor.constraint(equalTo: alert.view.leadingAnchor, constant: 20),
                progressView.trailingAnchor.constraint(equalTo: alert.view.trailingAnchor, constant: -20),
                progressView.topAnchor.constraint(equalTo: alert.view.bottomAnchor, constant: -60),
                progressView.heightAnchor.constraint(equalToConstant: 2)
            ])
            
            // Present the alert
            UIApplication.shared.windows.first?.rootViewController?.present(alert, animated: true)
            
            // Create 30 data points distributed over the past month to ensure enough data for AI analysis
            // This is increased from 15 to meet the minimum 10 data points requirement with enough buffer
            print("Creating 30 test data points distributed over time...")
            submissionService.createMultipleTestDataPoints(token: token, count: 30) { success in
                // Update progress
                DispatchQueue.main.async {
                    // Dismiss the alert
                    alert.dismiss(animated: true) {
                        if success {
                            print("Successfully created multiple test health data points")
                            // Wait a moment to ensure server has processed all data
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                                self.refreshData() // Refresh data after creation
                            }
                        } else {
                            print("Failed to create test health data points")
                            // Show error message if needed
                            let errorAlert = UIAlertController(
                                title: "Error Creating Data",
                                message: submissionService.submissionError ?? "Unknown error occurred",
                                preferredStyle: .alert
                            )
                            errorAlert.addAction(UIAlertAction(title: "OK", style: .default))
                            UIApplication.shared.windows.first?.rootViewController?.present(errorAlert, animated: true)
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
    
    private func statusOverviewCard(_ analysis: AITrendAnalysis) -> some View {
        VStack(spacing: 15) {
            // Status indicator with color
            HStack {
                Text("Health Status:")
                    .font(.headline)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text(analysis.healthTrajectory.capitalized)
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundColor(statusColor(analysis.healthTrajectory))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(statusColor(analysis.healthTrajectory).opacity(0.2))
                    .cornerRadius(8)
            }
            
            Divider()
            
            // Time range analyzed
            HStack {
                Text("Period Analyzed:")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text("\(timeRange) days")
                    .font(.subheadline)
                    .fontWeight(.medium)
            }
            
            // Quick assessment
            HStack(alignment: .top) {
                Image(systemName: "brain")
                    .font(.headline)
                    .foregroundColor(.purple)
                    .frame(width: 24)
                
                Text(analysis.assessment)
                    .font(.subheadline)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.top, 4)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
        .padding(.horizontal)
    }
    
    private func assessmentCard(_ analysis: AITrendAnalysis) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Detailed Assessment")
                .font(.headline)
            
            Divider()
            
            Text(analysis.assessment)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            
            if !analysis.trendSummary.isEmpty {
                Text("Trend Analysis")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .padding(.top, 8)
                
                Text(analysis.trendSummary)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
        .padding(.horizontal)
    }
    
    private func recommendationsCard(_ analysis: AITrendAnalysis) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recommendations")
                .font(.headline)
            
            Divider()
            
            if analysis.recommendations.isEmpty {
                Text("No specific recommendations at this time.")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            } else {
                ForEach(analysis.recommendations, id: \.self) { recommendation in
                    HStack(alignment: .top) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                            .font(.subheadline)
                            .frame(width: 22, alignment: .center)
                        
                        Text(recommendation)
                            .font(.subheadline)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.vertical, 2)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
        .padding(.horizontal)
    }
    
    private func concernsCard(_ analysis: AITrendAnalysis) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Potential Concerns")
                .font(.headline)
            
            Divider()
            
            ForEach(analysis.potentialConcerns, id: \.self) { concern in
                HStack(alignment: .top) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.yellow)
                        .font(.subheadline)
                        .frame(width: 22, alignment: .center)
                    
                    Text(concern)
                        .font(.subheadline)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.vertical, 2)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
        .padding(.horizontal)
    }
    
    private func medicalAttentionCard(_ analysis: AITrendAnalysis) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Medical Attention")
                .font(.headline)
            
            Divider()
            
            HStack {
                Image(systemName: medicalAttentionIcon(analysis.medicalConsultation))
                    .foregroundColor(medicalAttentionColor(analysis.medicalConsultation))
                    .font(.title2)
                    .frame(width: 40)
                
                VStack(alignment: .leading) {
                    Text(medicalAttentionTitle(analysis.medicalConsultation))
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(medicalAttentionColor(analysis.medicalConsultation))
                    
                    Text(medicalAttentionDescription(analysis.medicalConsultation))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.vertical, 4)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
        .padding(.horizontal)
    }
    
    private func followUpCard(_ analysis: AITrendAnalysis) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Follow-up Guidance")
                .font(.headline)
            
            Divider()
            
            HStack {
                Image(systemName: "calendar.badge.clock")
                    .foregroundColor(.blue)
                    .font(.title3)
                    .frame(width: 30)
                
                // Use lifestyle adjustments as follow-up guidance
                Text(analysis.lifestyleAdjustments.joined(separator: ". "))
                    .font(.subheadline)
            }
            .padding(.vertical, 4)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
        .padding(.horizontal)
    }
    
    // Helper functions for UI elements
    
    private func statusColor(_ status: String) -> Color {
        switch status.lowercased() {
        case "normal", "stable":
            return .blue
        case "improving":
            return .green
        case "moderate", "concerning":
            return .yellow
        case "elevated", "declining":
            return .orange
        case "severe", "critical":
            return .red
        default:
            return .blue
        }
    }
    
    private func medicalAttentionIcon(_ level: String) -> String {
        switch level.lowercased() {
        case "none":
            return "checkmark.circle.fill"
        case "if_symptoms_persist":
            return "clock.arrow.circlepath"
        case "soon":
            return "exclamationmark.circle.fill"
        case "immediately":
            return "bell.fill"
        default:
            return "questionmark.circle.fill"
        }
    }
    
    private func medicalAttentionColor(_ level: String) -> Color {
        switch level.lowercased() {
        case "none":
            return .green
        case "if_symptoms_persist":
            return .yellow
        case "soon":
            return .orange
        case "immediately":
            return .red
        default:
            return .gray
        }
    }
    
    private func medicalAttentionTitle(_ level: String) -> String {
        switch level.lowercased() {
        case "none":
            return "No Medical Attention Needed"
        case "if_symptoms_persist":
            return "Consult If Symptoms Persist"
        case "soon":
            return "Medical Consultation Recommended Soon"
        case "immediately":
            return "Seek Medical Attention Immediately"
        default:
            return "Consult with Healthcare Provider"
        }
    }
    
    private func medicalAttentionDescription(_ level: String) -> String {
        switch level.lowercased() {
        case "none":
            return "Continue regular monitoring and healthy habits"
        case "if_symptoms_persist":
            return "Watch for changes over the next 24-48 hours"
        case "soon":
            return "Schedule appointment with healthcare provider in the next week"
        case "immediately":
            return "Contact emergency services or visit nearest medical facility"
        default:
            return "Follow up with your doctor at your convenience"
        }
    }
}

struct AIHealthAnalysisView_Previews: PreviewProvider {
    static var previews: some View {
        let service = AIHealthAnalysisService()
        service.analysis = AIHealthAnalysisService.previewAnalysis
        
        return NavigationView {
            AIHealthAnalysisView()
                .environmentObject(UserAuthModel())
        }
    }
}
