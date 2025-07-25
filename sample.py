import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Function to simulate normal network traffic
def generate_normal_traffic(n):
    normal_traffic = []
    for _ in range(n):
        timestamp = time.time()
        packet_size = random.randint(50, 1500)  # Normal packet sizes
        source_ip = f"192.168.1.{random.randint(1, 50)}"
        normal_traffic.append([timestamp, source_ip, packet_size, 'Normal'])
    return normal_traffic

# Function to simulate DoS attack traffic
def generate_dos_attack(n, dos_ip):
    dos_attack_traffic = []
    for _ in range(n):
        timestamp = time.time()
        packet_size = random.randint(500, 1500)  # Larger packet sizes for DoS
        dos_attack_traffic.append([timestamp, dos_ip, packet_size, 'DoS Attack'])
    return dos_attack_traffic

# Generate the dataset with 90% normal traffic and 10% DoS attack traffic
normal_traffic = generate_normal_traffic(900)
dos_ip = "192.168.1.100"  # Simulate all DoS traffic from one malicious IP
dos_attack_traffic = generate_dos_attack(100, dos_ip)

# Combine both normal traffic and DoS traffic into one dataset
traffic_data = normal_traffic + dos_attack_traffic

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(traffic_data, columns=['Timestamp', 'Source IP', 'Packet Size', 'Label'])

# Save the dataset to a CSV file for later use
df.to_csv('improved_network_traffic_data.csv', index=False)

# Print a sample of the dataset
print("Sample Data:\n", df.head())

# Feature Engineering: Calculate the number of packets and the average packet size per Source IP
ip_stats = df.groupby('Source IP').agg(
    packet_count=('Packet Size', 'size'),
    avg_packet_size=('Packet Size', 'mean')
).reset_index()

# Merge stats back into original dataframe for anomaly detection
df = df.merge(ip_stats, on='Source IP', how='left')

# Features for anomaly detection: 'Packet Size', 'Packet Count', 'Avg Packet Size'
X = df[['Packet Size', 'packet_count', 'avg_packet_size']].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Isolation Forest for anomaly detection
iso_forest = IsolationForest(contamination=0.05, random_state=42)  # Set contamination rate for expected outliers (5%)
df['Anomaly'] = iso_forest.fit_predict(X_scaled)

# Convert anomaly labels from {-1, 1} to {0, 1} (1 for anomalies, 0 for normal)
df['Anomaly'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Detection Logic: Check for high packet count from a single IP address (DoS detection)
def detect_dos_attack(df, threshold=50):
    # Group by Source IP and count the number of occurrences (packets)
    ip_count = df.groupby('Source IP').size()
    
    # Identify IPs that exceed the threshold for packet count
    dos_ips = ip_count[ip_count > threshold].index.tolist()
    
    if dos_ips:
        print(f"[ALERT] Potential DoS Attack detected from the following IPs:")
        for ip in dos_ips:
            print(f"- {ip} with {ip_count[ip]} packets")
    else:
        print("[INFO] No DoS Attack detected")
    
    return dos_ips, ip_count

# Visualization: Traffic Volume by Source IP with Anomaly-based DoS Detection
def plot_traffic_volume_with_anomaly_detection(df, threshold=50):
    # Get packet count per Source IP
    traffic_count = df['Source IP'].value_counts()

    # Detect potential DoS attacks based on threshold
    dos_ips, _ = detect_dos_attack(df, threshold)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Highlight DoS attack IPs in red
    colors = ['red' if ip in dos_ips else 'skyblue' for ip in traffic_count.index]

    # Create bar plot
    traffic_count.plot(kind='bar', color=colors, alpha=0.8)

    # Draw the threshold line
    plt.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold = {threshold} packets')

    # Title and labels
    plt.title('Traffic Volume by Source IP with Threshold-based DoS Detection', fontsize=16)
    plt.xlabel('Source IP', fontsize=12)
    plt.ylabel('Number of Packets', fontsize=12)
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Run the visualization with anomaly-based DoS attack detection
plot_traffic_volume_with_anomaly_detection(df, threshold=50)