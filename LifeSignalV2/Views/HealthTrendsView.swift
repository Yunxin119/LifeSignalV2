//
//  HealthTrendsView.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/16/25.
//

import SwiftUI
import Combine

struct HealthTrendsView: View {
    @StateObject private var trendsService = HealthTrendsService()
    @EnvironmentObject private var authModel: UserAuthModel
    
    @State private var timeRange: Int = 30
    @State private var showAIAnalysis = false
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
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
                
                if trendsService.isLoading {
                    ProgressView()
                        .padding()
                } else if let trends = trendsService.healthTrends {
                    // Trends Overview
                    trendsOverviewCard(trends)
                    
                    // Heart Rate Trends Card
                    metricTrendCard(
                        title: "Heart Rate Trends",
                        metric: trends.heartRate,
                        icon: "heart.fill",
                        color: .pink,
                        unit: "BPM"
                    )
                    
                    // Blood Oxygen Trends Card
                    metricTrendCard(
                        title: "Blood Oxygen Trends",
                        metric: trends.bloodOxygen,
                        icon: "lungs.fill",
                        color: .blue,
                        unit: "%"
                    )
                } else {
                    // Empty View
                    VStack(spacing: 15) {
                        Image(systemName: "chart.xyaxis.line")
                            .font(.system(size: 50))
                            .foregroundColor(.gray)
                        
                        Text("No Trend Data Available")
                            .font(.title3)
                            .fontWeight(.medium)
                        
                        Text("Health trends will appear here once enough data is collected.")
                            .multilineTextAlignment(.center)
                            .foregroundColor(.secondary)
                            .padding(.horizontal)
                        
                        Button(action: refreshData) {
                            Label("Refresh", systemImage: "arrow.clockwise")
                        }
                        .padding(.top)
                    }
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color(.systemBackground))
                    .cornerRadius(12)
                    .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
                    .padding()
                }
            }
            .navigationTitle("Health Trends")
            .alert(isPresented: $showAIAnalysis) {
                if let analysis = trendsService.aiAnalysis {
                    return Alert(
                        title: Text("AI Health Analysis"),
                        message: Text(analysis.trendSummary),
                        primaryButton: .default(Text("View Full Analysis")) {
                            // TODO: In a real app, this would navigate to a detailed view.
                            // Currently we don't have this implemented yet, may work on it in the future
                        },
                        secondaryButton: .cancel()
                    )
                } else if trendsService.isLoading {
                    return Alert(
                        title: Text("Loading AI Analysis"),
                        message: Text("Please wait while we analyze your health data."),
                        dismissButton: .cancel()
                    )
                } else {
                    return Alert(
                        title: Text("Analysis Not Available"),
                        message: Text("Unable to load AI health analysis. Please try again later."),
                        dismissButton: .cancel()
                    )
                }
            }
            .onAppear {
                refreshData()
            }
        }
    }
    
    private func refreshData() {
        if let token = authModel.token {
            trendsService.fetchHealthTrends(token: token, days: timeRange)
        }
    }
    
    private func trendsOverviewCard(_ trends: HealthTrends) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Trends Overview")
                .font(.headline)
            
            Divider()
            
            HStack {
                VStack(alignment: .leading) {
                    Text("Period Analyzed")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(trends.daysAnalyzed) days")
                        .font(.title3)
                        .fontWeight(.medium)
                }
                
                Spacer()
                
                VStack(alignment: .trailing) {
                    Text("Data Points")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(trends.dataPoints)")
                        .font(.title3)
                        .fontWeight(.medium)
                }
            }
            
            HStack {
                // Heart Rate Trend Direction
                trendDirectionIndicator(
                    trend: trends.heartRate.trend,
                    label: "Heart Rate",
                    icon: "heart.fill",
                    color: .pink
                )
                
                Spacer()
                
                // Blood Oxygen Trend Direction
                trendDirectionIndicator(
                    trend: trends.bloodOxygen.trend,
                    label: "Blood Oxygen",
                    icon: "lungs.fill",
                    color: .blue
                )
            }
            .padding(.top, 8)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
        .padding(.horizontal)
    }
    
    private func metricTrendCard(title: String, metric: HealthTrends.MetricTrend, icon: String, color: Color, unit: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                Text(title)
                    .font(.headline)
                Spacer()
            }
            
            Divider()
            
            // Range info
            HStack {
                VStack(alignment: .leading) {
                    Text("Average")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    HStack(alignment: .firstTextBaseline) {
                        Text(String(format: "%.1f", metric.mean))
                            .font(.title2)
                            .fontWeight(.bold)
                        Text(unit)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
                
                VStack(alignment: .center) {
                    Text("Range")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(Int(metric.min)) - \(Int(metric.max)) \(unit)")
                        .font(.subheadline)
                }
                
                Spacer()
                
                VStack(alignment: .trailing) {
                    Text("Variation")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "±%.1f", metric.std))
                        .font(.subheadline)
                }
            }
            
            // Trend Status
            HStack {
                Text("Trend:")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Text(metric.trend.capitalized)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(trendColor(metric.trend))
                
                Spacer()
                
                trendArrow(for: metric.trend)
                    .foregroundColor(trendColor(metric.trend))
            }
            .padding(.top, 4)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
        .padding(.horizontal)
    }
    
    private func trendDirectionIndicator(trend: String, label: String, icon: String, color: Color) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .foregroundColor(color)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                HStack {
                    Text(trend.capitalized)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(trendColor(trend))
                    
                    trendArrow(for: trend)
                        .foregroundColor(trendColor(trend))
                }
            }
        }
    }
    
    private func trendArrow(for trend: String) -> some View {
        Group {
            switch trend.lowercased() {
            case "increasing":
                Image(systemName: "arrow.up")
            case "decreasing":
                Image(systemName: "arrow.down")
            default:
                Image(systemName: "arrow.forward")
            }
        }
    }
    
    private func trendColor(_ trend: String) -> Color {
        switch trend.lowercased() {
        case "increasing":
            return .green
        case "decreasing":
            return .red
        default:
            return .blue
        }
    }
}

struct HealthTrendsView_Previews: PreviewProvider {
    static var previews: some View {
        let service = HealthTrendsService()
        service.healthTrends = HealthTrendsService.previewTrends
        service.aiAnalysis = HealthTrendsService.previewAIAnalysis
        
        return NavigationView {
            HealthTrendsView()
                .environmentObject(UserAuthModel())
                .environmentObject(NotificationService())
        }
    }
}
