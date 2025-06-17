import pandas as pd

# Load spike log
df = pd.read_csv("spike_log.txt", sep=" ", names=["timestamp", "itcopy_latency", "oncopy_latency", "kernel_latency"])

# Define a spike threshold (adjust as needed)
SPIKE_THRESHOLD = 100

# Count spikes for each function
itcopy_spike_count = (df["itcopy_latency"] >= SPIKE_THRESHOLD).sum()
oncopy_spike_count = (df["oncopy_latency"] >= SPIKE_THRESHOLD).sum()
kernel_spike_count = (df["kernel_latency"] >= SPIKE_THRESHOLD).sum()

# Print results
print(f"Total itcopy spikes: {itcopy_spike_count}")
print(f"Total oncopy spikes: {oncopy_spike_count}")
print(f"Total kernel spikes: {kernel_spike_count}")
