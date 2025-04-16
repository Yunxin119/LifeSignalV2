import SwiftUI

struct BloodOxygenView: View {
    @ObservedObject var healthMonitor: WatchHealthMonitor
    
    var body: some View {
        VStack(spacing: 5) {
            HStack {
                Image(systemName: "lungs.fill")
                    .foregroundColor(.blue)
                Text("Oxygen")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
            }
            
            if let bloodOxygen = healthMonitor.currentBloodOxygen {
                HStack(alignment: .bottom, spacing: 2) {
                    Text("\(Int(bloodOxygen))")
                        .font(.system(size: 28, weight: .semibold))
                        .foregroundColor(healthMonitor.isBloodOxygenAbnormal ? .red : .primary)
                    
                    Text("%")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .padding(.bottom, 4)
                    
                    Spacer()
                    
                    if healthMonitor.isBloodOxygenAbnormal {
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
                .fill(Color.primary.opacity(0.1))
                .shadow(color: Color.black.opacity(0.1), radius: 3, x: 0, y: 1)
        )
    }
}

#Preview {
    let monitor = WatchHealthMonitor.shared
    return BloodOxygenView(healthMonitor: monitor)
        .padding()
        .previewLayout(.sizeThatFits)
} 
