import pandas as pd

# Constants
TILE_SIZE = 128  # You can try TILE_SIZE = 64 if needed

# Load CSV (type, start, tiles)
df = pd.read_csv("reconstructed_layers.csv")

print("== Reconstructed VGG-16 Layer Shapes from Tile Counts ==")
print("--------------------------------------------------------")

for idx, row in df.iterrows():
    layer_type = row["type"]
    start = int(row["start"])
    tiles = int(row["tiles"])

    print(f"\nLayer {idx:02d} @ window {start} | Type: {layer_type} | Tiles: {tiles}")

    # Infer possible (N, M) combinations: tiles = ceil(N/T) * ceil(M/T)
    candidates = []
    for i in range(1, tiles + 1):
        if tiles % i == 0:
            n_tiles = i
            m_tiles = tiles // i
            N = n_tiles * TILE_SIZE
            M = m_tiles * TILE_SIZE
            candidates.append((N, M))

    if not candidates:
        print("  → No (N, M) candidates found.")
        continue

    print(f"  → Possible (N, M) pairs (assuming {TILE_SIZE}×{TILE_SIZE} tiles):")
    for N, M in candidates:
        print(f"    - N = {N}, M = {M}")

    if layer_type == "fc":
        for N, M in candidates:
            print(f"    → FC Layer: Input features = {N}, Output = {M}")
    elif layer_type == "conv":
        for N, M in candidates:
            h_w = int(N**0.5)
            print(f"    → Conv Layer: C_out = {M}, H_out × W_out ≈ {N} ({h_w}×{h_w}), assuming 3×3 kernel")
    else:
        print("  → Unknown layer type")

