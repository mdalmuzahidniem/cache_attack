import pandas as pd
import matplotlib.pyplot as plt

# Load
latency_df = pd.read_csv("latency_log.csv")

# Filter: keep only measurements >= 200 cycles
filtered_df = latency_df[latency_df['Latency'] >= 300]

# Group: for each cache set, pick maximum latency across 100 iterations
max_latency_per_set = filtered_df.groupby('SetIndex')['Latency'].max()


# Sort the cache sets by maximum latency descending (highest first)
sorted_sets = max_latency_per_set.sort_values(ascending=False)

# Select top N hot sets (N=100 or 200)
TOP_N = 100  # or 200
top_hot_sets = sorted_sets.head(TOP_N)

# Save the hot set indices into a text file
with open('hot_sets.txt', 'w') as f:
    for idx in top_hot_sets.index:
        f.write(f"{idx}\n")

print(f"✅ Top {TOP_N} hot cache sets selected and saved successfully!")



# Plot
plt.figure(figsize=(18, 6))
plt.scatter(max_latency_per_set.index, max_latency_per_set.values, s=5)
plt.title("Cache Set Index vs Maximum Latency (Spikes)")
plt.xlabel("Cache Set Index (0 to 16383)")
plt.ylabel("Maximum Latency (cycles)")
plt.grid(True)
plt.show()
