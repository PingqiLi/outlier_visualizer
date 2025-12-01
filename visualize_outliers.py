import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import glob

def split_qkv(matrix, num_heads, num_kv_heads, head_dim):
    # matrix shape: [Tokens, Hidden_Size] where Hidden_Size = (Num_Heads + 2 * Num_KV_Heads) * Head_Dim
    # We need to split along the last dimension
    
    q_dim = num_heads * head_dim
    k_dim = num_kv_heads * head_dim
    v_dim = num_kv_heads * head_dim
    
    total_dim = q_dim + k_dim + v_dim
    if matrix.shape[1] != total_dim:
        print(f"  Warning: Matrix shape {matrix.shape} does not match config Q({q_dim})+K({k_dim})+V({v_dim})={total_dim}. Skipping split.")
        return None
    
    q = matrix[:, :q_dim]
    k = matrix[:, q_dim:q_dim+k_dim]
    v = matrix[:, q_dim+k_dim:]
    
    return {"q_proj": q, "k_proj": k, "v_proj": v}

def visualize_outliers(data_dir, output_dir, layer_pattern, io_type, qkv_config=None):
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse QKV config if provided: "num_heads,num_kv_heads,head_dim"
    qkv_params = None
    if qkv_config:
        try:
            parts = [int(x) for x in qkv_config.split(',')]
            if len(parts) == 3:
                qkv_params = tuple(parts)
                print(f"QKV Splitting Enabled: Heads={parts[0]}, KV_Heads={parts[1]}, Dim={parts[2]}")
        except:
            print("Invalid QKV config format. Use 'num_heads,num_kv_heads,head_dim'")

    # Find matching layer directories
    all_layers = [d for d in data_path.iterdir() if d.is_dir()]
    matched_layers = [d for d in all_layers if layer_pattern in d.name]
    
    if not matched_layers:
        print(f"No layers found matching pattern: {layer_pattern}")
        return

    print(f"Found {len(matched_layers)} matching layers. Generating plots...")

    for layer_dir in matched_layers:
        layer_name = layer_dir.name
        print(f"Processing {layer_name}...")
        
        # Find all token files
        file_pattern = f"token_*_{io_type}.npy"
        files = list(layer_dir.glob(file_pattern))
        
        if not files:
            print(f"  No {io_type} files found in {layer_name}")
            continue
            
        files.sort(key=lambda x: int(x.name.split('_')[1]))
        
        data_list = []
        token_ids = []
        
        for f in files:
            try:
                arr = np.load(f)
                if arr.ndim > 1:
                    arr = arr.flatten()
                data_list.append(arr)
                token_ids.append(int(f.name.split('_')[1]))
            except Exception as e:
                print(f"  Error loading {f}: {e}")
        
        if not data_list:
            continue
            
        matrix = np.stack(data_list)
        
        # Prepare items to plot: {name: matrix}
        items_to_plot = {layer_name: matrix}
        
        # Special handling for QKV splitting
        if "qkv_proj" in layer_name and qkv_params and io_type == "output":
            splits = split_qkv(matrix, *qkv_params)
            if splits:
                # Add split components to plot list
                for k, v in splits.items():
                    # Construct new name: layers.10.self_attn.q_proj
                    base_name = layer_name.replace("qkv_proj", k)
                    items_to_plot[base_name] = v
        
        # Special handling for MoE Experts
        if "experts" in layer_name:
            print(f"  Note: '{layer_name}' represents the aggregated output of the MoE block (128 experts).")

        for name, mat in items_to_plot.items():
            # 1. Heatmap
            plt.figure(figsize=(12, 8))
            plt.imshow(np.abs(mat), aspect='auto', cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Absolute Magnitude')
            plt.title(f"Activation Heatmap: {name} ({io_type})")
            plt.xlabel("Channel Index")
            plt.ylabel("Token ID")
            
            safe_name = name.replace(".", "_")
            plt.savefig(output_path / f"{safe_name}_{io_type}_heatmap.png")
            plt.close()
            
            # 2. Max-Abs
            max_vals = np.max(np.abs(mat), axis=1)
            plt.figure(figsize=(10, 6))
            plt.plot(token_ids, max_vals, marker='.', linestyle='-', linewidth=0.5, markersize=2)
            plt.title(f"Max-Abs Activation per Token: {name} ({io_type})")
            plt.xlabel("Token ID")
            plt.ylabel("Max Absolute Value")
            plt.grid(True)
            
            plt.savefig(output_path / f"{safe_name}_{io_type}_max_abs.png")
            plt.close()
            
        print(f"  Saved plots for {layer_name} (and splits if applicable)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize activation outliers from NPY files.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to STITCHED NPY directory")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Output directory for plots")
    parser.add_argument("--layer_pattern", type=str, default="mlp", help="Substring to filter layers")
    parser.add_argument("--io_type", type=str, default="output", choices=["input", "output"], help="Input or Output activations")
    parser.add_argument("--qkv_config", type=str, default=None, help="Optional: 'num_heads,num_kv_heads,head_dim' to split qkv_proj")
    
    args = parser.parse_args()
    
    visualize_outliers(args.data_dir, args.output_dir, args.layer_pattern, args.io_type, args.qkv_config)
