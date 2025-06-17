import re
import pandas as pd

# --- File paths ---
layer_file = "layer_log.txt"
spike_file = "spike_log.txt"

# --- Step 1: Load spike_log.txt ---
try:
    spike_df = pd.read_csv(spike_file, sep=' ', header=None, names=['tsc', 'itcopy', 'oncopy', 'kernel'])
except FileNotFoundError:
    print(f"❌ Error: File '{spike_file}' not found.")
    exit(1)

spike_min = spike_df['tsc'].min()
spike_max = spike_df['tsc'].max()

print(f"🕒 Spike log TSC range: {spike_min} - {spike_max}")

# --- Step 2: Parse layer_log.txt ---
inference_windows = []
with open(layer_file, 'r') as f:
    current_inference = None
    start_tsc = end_tsc = None

    for line in f:
        if "Inference" in line:
            match = re.search(r'Inference\s+(\d+)', line)
            if match:
                current_inference = int(match.group(1))
        elif "START_TSC" in line:
            start_match = re.search(r'(\d+)', line)
            if start_match:
                start_tsc = int(start_match.group(1))
        elif "END_TSC" in line:
            end_match = re.search(r'(\d+)', line)
            if end_match and start_tsc is not None and current_inference is not None:
                end_tsc = int(end_match.group(1))
                inference_windows.append((current_inference, start_tsc, end_tsc))
                # reset
                current_inference, start_tsc, end_tsc = None, None, None

# --- Step 3: Check which inferences are fully inside spike log ---
fully_contained = []
for i, start, end in inference_windows:
    if start >= spike_min and end <= spike_max:
        fully_contained.append((i, start, end))

# --- Step 4: Output result ---
if fully_contained:
    print("\n✅ Fully contained inferences inside spike_log.txt:")
    for i, start, end in fully_contained:
        print(f"Inference {i}: START_TSC={start}, END_TSC={end}")
else:
    print("⚠️ No inferences fully contained within spike_log.txt.")

