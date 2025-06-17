"""
import pandas as pd

# Load the spike vectors (only itcopy and oncopy should be present now)
df = pd.read_csv("spike_vectors.csv", index_col=0)

# Templates from Cache Telepathy (based on iter1–iter4)
conv_template = [1, 1]       # iter1
fc_templates = [[1, 0], [0, 1]]  # iter2 and iter3

# Initialize tracking
current_layer = []
last_match = None

# Phase 1: Pattern Matching (based only on itcopy + oncopy)
for idx, row in df.iterrows():
    vec = [row["itcopy_spike"], row["oncopy_spike"]]

    if vec == conv_template:
        if last_match == "conv":
            current_layer[-1]["tiles"] += 1
        else:
            current_layer.append({"type": "conv", "start": idx, "tiles": 1})
            last_match = "conv"

    elif vec in fc_templates:
        if last_match == "fc":
            current_layer[-1]["tiles"] += 1
        else:
            current_layer.append({"type": "fc", "start": idx, "tiles": 1})
            last_match = "fc"

    else:
        last_match = None  # No spike → break layer grouping

# Phase 2: Heuristic correction based on tile count
for layer in current_layer:
    if layer["tiles"] == 1 and layer["type"] == "fc":
        layer["type"] = "conv"  # Too short for FC, likely conv
    elif layer["tiles"] >= 3 and layer["type"] == "conv":
        layer["type"] = "fc"    # Too long for conv, likely FC

# Report results
for i, layer in enumerate(current_layer):
    print(f"Layer {i:02d}: Type = {layer['type']}, Start Window = {layer['start']}, Tiles = {layer['tiles']}")

# Optional: save to CSV
pd.DataFrame(current_layer).to_csv("reconstructed_layers.csv", index=False)
"""

import pandas as pd

# === Load spike vector data ===
try:
    df = pd.read_csv("spike_vectors.csv")
except FileNotFoundError:
    print("❌ Error: 'spike_vectors.csv' not found.")
    exit(1)

# === Ensure timestamp column exists ===
if "timestamp" not in df.columns:
    print("❌ 'timestamp' column is missing in spike_vectors.csv.")
    exit(1)

# === Step 1: Sort by timestamp just in case ===
df = df.sort_values(by="timestamp").reset_index(drop=True)

# === Step 2: Compute spike-to-spike timing gap (in TSC cycles) ===
df["delta_tsc"] = df["timestamp"].diff().fillna(0)

# === Step 3: Group into layers based on GAP threshold ===
GAP_THRESHOLD = 200000  # You can tune this: 50_000–200_000 usually good
layer_ids = []
current_layer = 0
for delta in df["delta_tsc"]:
    if delta > GAP_THRESHOLD:
        current_layer += 1
    layer_ids.append(current_layer)
df["layer_index"] = layer_ids

# === Step 4: Summarize layers by tile count ===
layer_summary = df.groupby("layer_index").size().reset_index(name="tile_count")
layer_summary["layer_type"] = ["conv" if count >= 3 else "fc" for count in layer_summary["tile_count"]]

# === Step 5: Save output ===
layer_summary.to_csv("layer_tiles.csv", index=False)

# === Print result ===
print("✅ GEMM-loop-based segmentation complete. Saved to 'layer_tiles.csv':\n")
print(layer_summary)







# === MERGE LOGIC: Group GEMM segments into final layers ===


print("\n Merging GEMM tile segments into final layers...")

# Load layer_tiles.csv (produced earlier in this script)
df = pd.read_csv("layer_tiles.csv")

# Optional: Filter out noise
MIN_TILES = 1000
df = df[df["tile_count"] >= MIN_TILES].reset_index(drop=True)

# Merge logic
MERGED_LAYERS = []
current_group = []
TILE_JUMP_THRESHOLD = 100  # adjust to control merging

for i in range(len(df)):
    if not current_group:
        current_group.append(df.iloc[i])
    else:
        last_tile_count = current_group[-1]["tile_count"]
        current_tile_count = df.iloc[i]["tile_count"]

        if abs(current_tile_count - last_tile_count) <= TILE_JUMP_THRESHOLD:
            current_group.append(df.iloc[i])
        else:
            total_tiles = sum(layer["tile_count"] for layer in current_group)
            layer_type = "conv" if total_tiles > 2000 else "fc"
            MERGED_LAYERS.append({
                "merged_layer_index": len(MERGED_LAYERS),
                "tile_count": total_tiles,
                "layer_type": layer_type
            })
            current_group = [df.iloc[i]]

# Add final group
if current_group:
    total_tiles = sum(layer["tile_count"] for layer in current_group)
    layer_type = "conv" if total_tiles > 2000 else "fc"
    MERGED_LAYERS.append({
        "merged_layer_index": len(MERGED_LAYERS),
        "tile_count": total_tiles,
        "layer_type": layer_type
    })

# Save merged result
merged_df = pd.DataFrame(MERGED_LAYERS)
merged_df.to_csv("merged_layers.csv", index=False)

# Print result
print(" Final merged layers saved to 'merged_layers.csv':")
print(merged_df)




