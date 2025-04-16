import SwiftUI

struct HeartRateView: View {
    @ObservedObject var healthMonitor: WatchHealthMonitor
    
    var body: some View {
        VStack(spacing: 5) {
            HStack {
                Image(systemName: "heart.fill")
                    .foregroundColor(.red)
                Text("Heart Rate")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
            }
            
            if let heartRate = healthMonitor.currentHeartRate {
                HStack(alignment: .bottom, spacing: 2) {
                    Text("\(Int(heartRate))")
                        .font(.system(size: 28, weight: .semibold))
                        .foregroundColor(healthMonitor.isHeartRateAbnormal ? .red : .primary)
                    
                    Text("BPM")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .padding(.bottom, 4)
                    
                    Spacer()
                    
                    if healthMonitor.isHeartRateAbnormal {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.red)
                            .font(.caption)
                    }
                }
            } else {
                HStack {
                    Text("Waiting...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.gray)
                .shadow(color: Color.black.opacity(0.1), radius: 3, x: 0, y: 1)
        )
    }
}

#Preview {
    let monitor = WatchHealthMonitor.shared
    return HeartRateView(healthMonitor: monitor)
        .padding()
        .previewLayout(.sizeThatFits)
} 
