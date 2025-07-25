import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate synthetic traffic data (normal and anomalous traffic)
def generate_traffic_data(n_normal=900, n_anomalous=100, n_dos=500):
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

# Features and scaling for anomaly detection
X = df[['Packet Size', 'Source IP Numeric']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Isolation Forest for anomaly detection
model = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
df['Anomaly'] = model.fit_predict(X_scaled)

# Convert anomaly labels (-1 for anomaly, 1 for normal)
df['Anomaly'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)  # 1: anomaly, 0: normal

# DoS Attack Detection: Check for high packet counts per source IP
def detect_dos_attack(df, threshold=2):  # Lowered threshold to 2
    ip_count = df.groupby('Source IP').size()
    dos_ips = ip_count[ip_count > threshold].index.tolist()
    return dos_ips

# Detect DoS attacks for both methods
threshold_dos_ips = detect_dos_attack(df, threshold=2)  # Lower threshold

# Anomaly-based DoS detection (labels anomalies as 1, normal as 0)
actual_dos = df[df['Label'] == 1]

# For Threshold-based method, check which true DoS attacks exceed the threshold
threshold_dos_actual = df[df['Source IP'].isin(threshold_dos_ips) & (df['Label'] == 1)]

# Compute metrics for **Threshold-based Detection**
threshold_true_positive = len(threshold_dos_actual)
threshold_false_positive = len(df[(df['Source IP'].isin(threshold_dos_ips)) & (df['Label'] == 0)])
threshold_false_negative = len(actual_dos) - threshold_true_positive
threshold_true_negative = len(df) - threshold_true_positive - threshold_false_positive - threshold_false_negative

# Metrics for **Threshold-based Detection**
threshold_precision = threshold_true_positive / (threshold_true_positive + threshold_false_positive) if (threshold_true_positive + threshold_false_positive) > 0 else 0
threshold_recall = threshold_true_positive / (threshold_true_positive + threshold_false_negative) if (threshold_true_positive + threshold_false_negative) > 0 else 0
threshold_f1 = 2 * (threshold_precision * threshold_recall) / (threshold_precision + threshold_recall) if (threshold_precision + threshold_recall) > 0 else 0
threshold_accuracy = (threshold_true_positive + threshold_true_negative) / len(df)

# Compute metrics for **Anomaly-based Detection**
anomaly_true_positive = len(df[(df['Anomaly'] == 1) & (df['Label'] == 1)])
anomaly_false_positive = len(df[(df['Anomaly'] == 1) & (df['Label'] == 0)])
anomaly_false_negative = len(actual_dos) - anomaly_true_positive
anomaly_true_negative = len(df) - anomaly_true_positive - anomaly_false_positive - anomaly_false_negative

# Metrics for **Anomaly-based Detection**
anomaly_precision = anomaly_true_positive / (anomaly_true_positive + anomaly_false_positive) if (anomaly_true_positive + anomaly_false_positive) > 0 else 0
anomaly_recall = anomaly_true_positive / (anomaly_true_positive + anomaly_false_negative) if (anomaly_true_positive + anomaly_false_negative) > 0 else 0
anomaly_f1 = 2 * (anomaly_precision * anomaly_recall) / (anomaly_precision + anomaly_recall) if (anomaly_precision + anomaly_recall) > 0 else 0
anomaly_accuracy = (anomaly_true_positive + anomaly_true_negative) / len(df)

# Print the results for comparison
print("Threshold-based Detection Metrics:")
print(f"Precision: {threshold_precision:.2f}")
print(f"Recall: {threshold_recall:.2f}")
print(f"F1 Score: {threshold_f1:.2f}")
print(f"Accuracy: {threshold_accuracy:.2f}")

print("\nAnomaly-based Detection Metrics:")
print(f"Precision: {anomaly_precision:.2f}")
print(f"Recall: {anomaly_recall:.2f}")
print(f"F1 Score: {anomaly_f1:.2f}")
print(f"Accuracy: {anomaly_accuracy:.2f}")

# Calculate and display the percentage of improvement in F1 score and accuracy
if threshold_f1 > 0:
    f1_improvement = (anomaly_f1 - threshold_f1) / threshold_f1 * 100
else:
    f1_improvement = 0  # Prevent division by zero if F1 is zero for threshold method

if threshold_accuracy > 0:
    accuracy_improvement = (anomaly_accuracy - threshold_accuracy) / threshold_accuracy * 100
else:
    accuracy_improvement = 0  # Prevent division by zero if accuracy is zero for threshold method

print("\nPercentage of Improvement (Anomaly-based over Threshold-based):")
print(f"F1 Score Improvement: {f1_improvement:.2f}%")
print(f"Accuracy Improvement: {accuracy_improvement:.2f}%")

# Visualize the distribution of packets by Source IP
def plot_packet_distribution(df):
    ip_count = df.groupby('Source IP').size()
    plt.figure(figsize=(12, 6))
    ip_count.plot(kind='bar', color='skyblue', alpha=0.8)
    plt.title('Packet Count by Source IP', fontsize=16)
    plt.xlabel('Source IP', fontsize=12)
    plt.ylabel('Number of Packets', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Call function to visualize packet distribution
plot_packet_distribution(df)