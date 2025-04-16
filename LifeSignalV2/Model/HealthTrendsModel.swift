//
//  HealthTrendsModel.swift
//  LifeSignalV2
//
//  Created by Yunxin Liu on 4/16/25.
//

struct HealthTrends: Decodable {
    let daysAnalyzed: Int
    let dataPoints: Int
    let heartRate: MetricTrend
    let bloodOxygen: MetricTrend
    
    struct MetricTrend: Decodable {
        let mean: Double
        let std: Double
        let min: Double
        let max: Double
        let trend: String
        
        // Add explicit initializer for MetricTrend
        init(mean: Double, std: Double, min: Double, max: Double, trend: String) {
            self.mean = mean
            self.std = std
            self.min = min
            self.max = max
            self.trend = trend
        }
    }
    
    enum CodingKeys: String, CodingKey {
        case daysAnalyzed = "days_analyzed"
        case dataPoints = "data_points"
        case heartRate = "heart_rate"
        case bloodOxygen = "blood_oxygen"
    }
    
    // Add explicit memberwise initializer for HealthTrends
    init(daysAnalyzed: Int, dataPoints: Int, heartRate: MetricTrend, bloodOxygen: MetricTrend) {
        self.daysAnalyzed = daysAnalyzed
        self.dataPoints = dataPoints
        self.heartRate = heartRate
        self.bloodOxygen = bloodOxygen
    }
    
    // Custom decoder to handle variable field structures and formats
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        // Try to decode fields directly or with fallbacks
        do {
            daysAnalyzed = try container.decode(Int.self, forKey: .daysAnalyzed)
        } catch {
            print("Warning: could not decode days_analyzed directly: \(error)")
            // Fallback to default value
            daysAnalyzed = 7
        }
        
        do {
            dataPoints = try container.decode(Int.self, forKey: .dataPoints)
        } catch {
            print("Warning: could not decode data_points directly: \(error)")
            // Fallback to default value
            dataPoints = 0
        }
        
        do {
            heartRate = try container.decode(MetricTrend.self, forKey: .heartRate)
        } catch {
            print("Warning: could not decode heart_rate: \(error)")
            // Create an empty metric trend as fallback
            heartRate = MetricTrend(mean: 0, std: 0, min: 0, max: 0, trend: "unknown")
        }
        
        do {
            bloodOxygen = try container.decode(MetricTrend.self, forKey: .bloodOxygen)
        } catch {
            print("Warning: could not decode blood_oxygen: \(error)")
            // Create an empty metric trend as fallback
            bloodOxygen = MetricTrend(mean: 0, std: 0, min: 0, max: 0, trend: "unknown")
        }
    }
}
