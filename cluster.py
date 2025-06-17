import pandas as pd

# Inference 3 time window from layer_log.txt (in nanoseconds)
START_TSC = 5624980142955251
END_TSC = 5624981559696709

CPU_FREQ_HZ = 2239693000  # 2.24 GHz
# Load spike log (still reading all columns for compatibility)
df = pd.read_csv("spike_log.txt", sep=" ", names=["timestamp", "itcopy_latency", "oncopy_latency", "kernel_latency"])


# Filter spike data to the time window of inference 7
df_window = df[(df["timestamp"] >= START_TSC) & (df["timestamp"] <= END_TSC)].copy()

df_window["relative_time_cycles"] = df_window["timestamp"] - START_TSC
df_window["relative_time_s"] = df_window["relative_time_cycles"] / CPU_FREQ_HZ

# Define latency threshold above which we consider a spike
THRESHOLD = 600

# Create binary spike columns (only itcopy and oncopy)
df_window["itcopy_spike"] = (df_window["itcopy_latency"] > THRESHOLD).astype(int)
df_window["oncopy_spike"] = (df_window["oncopy_latency"] > THRESHOLD).astype(int)

# Define time window size in seconds for clustering
WINDOW_SIZE_CYCLES = 2000

# Assign a window index to each row
df_window["window"] = (df_window["relative_time_cycles"] / WINDOW_SIZE_CYCLES).astype(int)

# Aggregate spikes in each window using logical OR (max)
window_vectors = df_window.groupby("window")[["itcopy_spike", "oncopy_spike"]].max()

# Print clustered spike vectors
#for idx, row in window_vectors.iterrows():
    #print(f"Window {idx:04d} → [{row.itcopy_spike} {row.oncopy_spike}]")

# Save to CSV for further analysis
# Also get the first TSC timestamp for each window
window_timestamps = df_window.groupby("window")["timestamp"].first()

# Merge timestamps with spike vectors
window_vectors["timestamp"] = window_timestamps

# Reorder columns so timestamp comes first
window_vectors = window_vectors[["timestamp", "itcopy_spike", "oncopy_spike"]]

# Save to CSV for further analysis
window_vectors.to_csv("spike_vectors.csv", index=False)


