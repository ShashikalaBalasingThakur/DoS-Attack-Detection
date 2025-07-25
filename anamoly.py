import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Generate synthetic traffic data (normal and anomalous traffic)
def generate_traffic_data(n_normal=900, n_anomalous=100, n_dos=50):
    data = []
    # Normal traffic
    for _ in range(n_normal):
        packet_size = np.random.randint(50, 1500)
        source_ip = f"192.168.1.{np.random.randint(1, 50)}"
        data.append([packet_size, source_ip, 0])  # Label 0 for normal
    # Anomalous traffic (e.g., DoS-like)
    for _ in range(n_anomalous):
        packet_size = np.random.randint(1000, 1500)
        source_ip = f"192.168.1.{np.random.randint(51, 100)}"
        data.append([packet_size, source_ip, 1])  # Label 1 for anomaly
    # Simulate DoS attack (high volume of packets from specific IPs)
    for _ in range(n_dos):
        packet_size = np.random.randint(1500, 3000)  # Large packet size for DoS
        source_ip = f"192.168.1.{np.random.randint(101, 150)}"
        data.append([packet_size, source_ip, 1])  # Label 1 for DoS
    return pd.DataFrame(data, columns=['Packet Size', 'Source IP', 'Label'])

# Create dataset
df = generate_traffic_data()

# Convert Source IP to numerical values for model processing
df['Source IP Numeric'] = df['Source IP'].apply(lambda ip: int(ip.split('.')[-1]))

# Features and scaling
X = df[['Packet Size', 'Source IP Numeric']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Isolation Forest for anomaly detection
model = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
df['Anomaly'] = model.fit_predict(X_scaled)

# Convert anomaly labels (-1 for anomaly, 1 for normal)
df['Anomaly'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)  # 1: anomaly, 0: normal

# DoS Attack Detection: Check for high packet counts per source IP
def detect_dos_attack(df, threshold=10):
    ip_count = df.groupby('Source IP').size()
    dos_ips = ip_count[ip_count > threshold].index.tolist()
    return dos_ips

# Visualize results
def plot_anomaly_detection(df):
    normal_data = df[df['Anomaly'] == 0]
    anomalous_data = df[df['Anomaly'] == 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(normal_data['Source IP Numeric'], normal_data['Packet Size'], label='Normal', alpha=0.7, c='blue')
    plt.scatter(anomalous_data['Source IP Numeric'], anomalous_data['Packet Size'], label='Anomalous', alpha=0.7, c='red')

    plt.title("Anomaly Detection in Network Traffic", fontsize=16)
    plt.xlabel("Source IP (Numeric)", fontsize=12)
    plt.ylabel("Packet Size", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

# Detect potential DoS attacks
dos_ips = detect_dos_attack(df, threshold=10)

# Run the visualization for anomaly-based DoS detection
plot_anomaly_detection(df)

# Summary of detected anomalies and DoS attacks
print("Summary of Detected Anomalies and DoS Attacks:")
print(df[df['Anomaly'] == 1])  # Anomalous data
print("\nDetected DoS Attack IPs:")
print(dos_ips)