import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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

def load_config_from_path(model_path):
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found in {model_path}")
        return None
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        num_heads = config.get("num_attention_heads")
        num_kv_heads = config.get("num_key_value_heads")
        hidden_size = config.get("hidden_size")
        
        if num_heads and num_kv_heads and hidden_size:
            # Check if head_dim is explicitly defined
            head_dim = config.get("head_dim")
            if head_dim is None:
                head_dim = hidden_size // num_heads
                
            print(f"Loaded config from {model_path}: Heads={num_heads}, KV_Heads={num_kv_heads}, Hidden={hidden_size}, Head_Dim={head_dim}")
            return (num_heads, num_kv_heads, head_dim)
        else:
            print(f"Error: Missing required fields in config.json (num_attention_heads, num_key_value_heads, hidden_size)")
            return None
            
    except Exception as e:
        print(f"Error reading config.json: {e}")
        return None

def plot_layer(layer_dir, output_path, io_type, qkv_params):
    layer_name = layer_dir.name
    
    # Filter out self_attn.attn as requested
    if "self_attn.attn" in layer_name:
        return
        
    # Look for merged file: input.npy or output.npy
    file_path = layer_dir / f"{io_type}.npy"
    
    if not file_path.exists():
        # print(f"  No {io_type}.npy found in {layer_name}")
        return
        
    try:
        matrix = np.load(file_path)
        # matrix shape should be [Num_Tokens, Hidden_Dim]
        if matrix.ndim == 1:
            # Single token case?
            matrix = matrix.reshape(1, -1)
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return
    
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
    
    # Special handling for Gate-Up splitting
    if "gate_up_proj" in layer_name and io_type == "output":
        # gate_up_proj output is [Gate | Up] concatenated along last dim
        hidden_dim = matrix.shape[-1]
        if hidden_dim % 2 == 0:
            split_dim = hidden_dim // 2
            gate_proj = matrix[..., :split_dim]
            up_proj = matrix[..., split_dim:]
            
            items_to_plot[layer_name.replace("gate_up_proj", "gate_proj")] = gate_proj
            items_to_plot[layer_name.replace("gate_up_proj", "up_proj")] = up_proj
            # print(f"  Split gate_up_proj into gate_proj and up_proj")
    
    # Special handling for MoE Experts
    if "experts" in layer_name:
        pass
        # print(f"  Note: '{layer_name}' represents the aggregated output of the MoE block (128 experts).")

    for name, mat in items_to_plot.items():
        # Filter out norm layers as requested
        if "norm" in name.lower():
            continue
            
        safe_name = name.replace(".", "_")
        
        # 3. 3D Surface Plot
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # User requested moderate downsampling (e.g., target ~500x500 grid)
            # This is faster than full resolution but detailed enough
            rows, cols = mat.shape
            row_step = max(1, rows // 500)
            col_step = max(1, cols // 500)
            
            mat_ds = np.abs(mat)[::row_step, ::col_step]
            rows_ds, cols_ds = mat_ds.shape
            x = np.arange(0, cols, col_step)[:cols_ds]
            y = np.arange(0, rows, row_step)[:rows_ds]
            X, Y = np.meshgrid(x, y)
            
            fig = plt.figure(figsize=(16, 12)) # Larger figure for detail
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot surface
            # stride parameters control the sampling of the rstride/cstride for the surface construction itself
            # Setting them to 1 means full resolution of the downsampled grid
            surf = ax.plot_surface(X, Y, mat_ds, cmap='coolwarm', edgecolor='none', alpha=0.8, rstride=1, cstride=1)
            
            ax.set_title(f"3D Activation Magnitude: {name}")
            ax.set_xlabel('Channel Index')
            ax.set_ylabel('Token ID')
            ax.set_zlabel('Magnitude')
            
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            plt.savefig(output_path / f"{safe_name}_{io_type}_3d.png", dpi=150) # Higher DPI
            plt.close()
            print(f"  Saved 3D plot for {name}")
        except Exception as e:
            print(f"  Warning: Failed to generate 3D plot for {name}: {e}")
        
    # print(f"  Processed {layer_name}")

def visualize_outliers(data_dir, output_dir, layer_pattern, io_type, qkv_config=None, model_path=None, workers=4):
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse QKV config
    qkv_params = None
    
    # Priority 1: Explicit config string
    if qkv_config:
        try:
            parts = [int(x) for x in qkv_config.split(',')]
            if len(parts) == 3:
                qkv_params = tuple(parts)
                print(f"Using provided QKV config: Heads={parts[0]}, KV_Heads={parts[1]}, Dim={parts[2]}")
        except:
            print("Invalid QKV config format. Use 'num_heads,num_kv_heads,head_dim'")
            
    # Priority 2: Auto-parse from model path
    elif model_path:
        qkv_params = load_config_from_path(model_path)
        
    if not qkv_params and "qkv_proj" in layer_pattern:
        print("Warning: No QKV config provided. QKV splitting will be skipped.")

    # Find matching layer directories
    all_layers = [d for d in data_path.iterdir() if d.is_dir()]
    matched_layers = [d for d in all_layers if layer_pattern in d.name]
    
    if not matched_layers:
        print(f"No layers found matching pattern: {layer_pattern}")
        return

    print(f"Found {len(matched_layers)} matching layers. Generating plots with {workers} workers...")

    # Parallel processing
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Create partial function with fixed arguments
        func = partial(plot_layer, output_path=output_path, io_type=io_type, qkv_params=qkv_params)
        
        # Submit tasks
        futures = [executor.submit(func, layer_dir) for layer_dir in matched_layers]
        
        # Wait for completion (optional: use tqdm if desired, but simple wait is fine)
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Task failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize activation outliers from NPY files.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to STITCHED NPY directory")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Output directory for plots")
    parser.add_argument("--layer_pattern", type=str, default="mlp", help="Substring to filter layers")
    parser.add_argument("--io_type", type=str, default="output", choices=["input", "output"], help="Input or Output activations")
    parser.add_argument("--qkv_config", type=str, default=None, help="Optional: 'num_heads,num_kv_heads,head_dim' to split qkv_proj")
    parser.add_argument("--model_path", type=str, default=None, help="Optional: Path to model directory containing config.json for auto-parsing QKV config")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    visualize_outliers(args.data_dir, args.output_dir, args.layer_pattern, args.io_type, args.qkv_config, args.model_path, args.workers)
