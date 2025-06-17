import pandas as pd
import matplotlib.pyplot as plt

# Load spike data
df = pd.read_csv("spike_log.txt", sep=" ", names=["timestamp", "itcopy_latency", "oncopy_latency", "kernel_latency"])

# Normalize time to start from 0
df["timestamp"] = df["timestamp"] - df["timestamp"].iloc[0]

# Apply filtering: below 100 → 0, above 2000 → 2000
for col in ["itcopy_latency", "oncopy_latency", "kernel_latency"]:
    df[col] = df[col].clip(lower=100, upper=2000)
    df[col] = df[col].where(df[col] != 100, 0)

# Plotting
plt.figure(figsize=(15, 6))
plt.plot(df["timestamp"], df["itcopy_latency"], label="itcopy latency", color="blue")
plt.plot(df["timestamp"], df["oncopy_latency"], label="oncopy latency", color="orange")
plt.plot(df["timestamp"], df["kernel_latency"], label="kernel latency", color="green")
plt.xlabel("Time (seconds)")
plt.ylabel("Latency (cycles)")
plt.title("Prime+Probe Latency Over Time (Filtered)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

