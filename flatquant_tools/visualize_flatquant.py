import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
import torch_npu


def unpack_int32_to_int4_signed(x):
    assert x.dtype == torch.int32
    E, K, N_ = x.shape # N_ = N/8

    out = torch.stack([(x >> (4 * i)) & 0xF for i in range(8)], dim=-1)
    out = out.to(torch.int8)
    out = torch.where(out >= 8, out - 16, out)
    out = out.reshape(E, K, N_ * 8).to(torch.int8)
    return out

def run_npu_kronecker_quant(
    x: torch.Tensor,
    left_trans: torch.Tensor,
    right_trans: torch.Tensor,
    clip_ratio: float = 1.0,
    device: str = "npu"
) -> torch.Tensor:
    """
    Executes torch_npu.npu_kronecker_quant and unpacks the result.
    """
    # Move to device
    x = x.to(device)
    left_trans = left_trans.to(device)
    right_trans = right_trans.to(device)

    # Handle 3D input (e.g. [S, G1, G2]) -> [S, D]
    if x.dim() == 3:
        s, g1, g2 = x.shape
        x = x.reshape(s, g1 * g2)
        d = g1 * g2
    else:
        s, d = x.shape

    g1 = left_trans.shape[0]
    g2 = right_trans.shape[0]
    
    if d != g1 * g2:
         raise ValueError(f"Hidden dim {d} does not match Kronecker factors {g1}x{g2}={g1*g2}")

    # Reshape for the op
    x = x.reshape(-1, g1, g2)
    
    print("Calling torch_npu.npu_kronecker_quant...")
    x_quantized_int32, scale = torch_npu.npu_kronecker_quant(
        x,
        left_trans,
        right_trans,
        clip_ratio=clip_ratio,
        dst_dtype=torch.int32
    )
    
    # Unpack
    print("Unpacking int32 result...")
    # Ensure 3D input for unpack_int32_to_int4_signed: [S, D//8] -> [S, 1, D//8]
    if x_quantized_int32.dim() == 2:
        x_quantized_int32 = x_quantized_int32.unsqueeze(1)
        
    x_unpacked = unpack_int32_to_int4_signed(x_quantized_int32)
    
    # Result is [S, 1, D], flatten to [S, D] and convert to float
    x_unpacked = x_unpacked.flatten(1).float()
    
    # Reshape back to [S, D] (redundant if flattened above, but safe)
    x_unpacked = x_unpacked.reshape(s, d)
    
    # Return unpacked int4 values (float type)
    return x_unpacked.cpu()

def plot_activation(orig_activation, transformed_activation, output_dir, name_prefix=""):
    """
    Generates plots for original and transformed activations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Data Preparation ---
    orig_flat = orig_activation.flatten().abs().float().numpy()
    trans_flat = transformed_activation.flatten().abs().float().numpy()
    
    # Downsample for 3D plot (Target ~500x500)
    def get_downsampled_data(tensor):
        mat = tensor.abs().float().numpy()
        rows, cols = mat.shape
        row_step = max(1, rows // 500)
        col_step = max(1, cols // 500)
        
        mat_ds = mat[::row_step, ::col_step]
        
        # Create meshgrid matching original coordinates
        rows_ds, cols_ds = mat_ds.shape
        x = np.arange(0, cols, col_step)[:cols_ds]
        y = np.arange(0, rows, row_step)[:rows_ds]
        X, Y = np.meshgrid(x, y)
        
        return X, Y, mat_ds, (rows, cols)

    X1, Y1, orig_ds, orig_shape = get_downsampled_data(orig_activation)
    X2, Y2, trans_ds, trans_shape = get_downsampled_data(transformed_activation)
    
    # --- 3D Surface Plot Comparison ---
    fig = plt.figure(figsize=(24, 10)) # Wider figure
    
    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X1, Y1, orig_ds, cmap='coolwarm', edgecolor='none', alpha=0.8, rstride=1, cstride=1)
    ax1.set_title(f"Original Activation (BF16)\nMax: {orig_flat.max():.2f}", fontsize=14)
    ax1.set_xlabel('Channel Index', fontsize=12)
    ax1.set_ylabel('Token ID', fontsize=12)
    ax1.set_zlabel('Magnitude', fontsize=12)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
    
    # Transformed
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X2, Y2, trans_ds, cmap='coolwarm', edgecolor='none', alpha=0.8, rstride=1, cstride=1)
    ax2.set_title(f"FlatQuant Output (Int4 Levels)\nMax: {trans_flat.max():.2f}", fontsize=14)
    ax2.set_xlabel('Channel Index', fontsize=12)
    ax2.set_ylabel('Token ID', fontsize=12)
    ax2.set_zlabel('Magnitude', fontsize=12)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
    
    # Clean filename generation
    # Try to extract layer info: root.model.layers.1.mlp.experts... -> L1_experts
    import re
    match = re.search(r'layers\.(\d+)', name_prefix)
    layer_idx = match.group(1) if match else "unknown"
    
    component = "layer"
    if "experts" in name_prefix:
        component = "experts"
    elif "mlp" in name_prefix:
        component = "mlp"
    elif "attn" in name_prefix:
        component = "attn"
        
    short_name = f"L{layer_idx}_{component}"
    filename = f"{short_name}_flatquant_comparison.png"
    
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to {output_dir}/{filename}")

def main():
    parser = argparse.ArgumentParser(description="Visualize FlatQuant activations (NPU Environment).")
    
    parser.add_argument("--dump_dir", type=str, required=True, help="Directory containing input_0.pth and (optional) output_0.pth")
    parser.add_argument("--output_dir", type=str, default="./flatquant_plots", help="Output directory")
    parser.add_argument("--device", type=str, default="npu:0", help="Device to run on")
    
    args = parser.parse_args()
    
    try:
        dump_path = Path(args.dump_dir)
        if not dump_path.exists():
            print(f"Error: Directory {dump_path} does not exist.")
            return

        # 1. Load Original Input (BF16)
        input_path = dump_path / "input_0.pth"
        if not input_path.exists():
             print(f"Error: input_0.pth not found in {dump_path}")
             return
             
        print(f"Loading input from {input_path}...")
        act = torch.load(input_path, map_location='cpu')
        
        # Ensure input is 2D for plotting/processing
        if act.dim() == 3:
             s, g1, g2 = act.shape
             act = act.reshape(s, g1 * g2)
        
        # 2. Try Load Output (Int32)
        output_path = dump_path / "output_0.pth"
        act_unpacked = None
        
        if output_path.exists():
            print(f"Found output file: {output_path}")
            out_int32 = torch.load(output_path, map_location='cpu')
            print(f"Output shape (Int32): {out_int32.shape}")
            
            print("Unpacking...")
            # Ensure 3D input: [S, D//8] -> [S, 1, D//8]
            if out_int32.dim() == 2:
                out_int32 = out_int32.unsqueeze(1)
            elif out_int32.dim() == 3:
                # If already [S, G1, G2//8], flatten last two to match [S, 1, D//8] logic or just pass as is?
                # unpack expects E, K, N_. If we pass [S, G1, G2//8], then E=S, K=G1, N_=G2//8.
                # Output will be [S, G1, G2]. This works perfectly!
                pass
                
            act_unpacked = unpack_int32_to_int4_signed(out_int32).float()
            
        else:
            print(f"output_0.pth not found. Attempting to run operator from inputs...")
            left_path = dump_path / "input_1.pth"
            right_path = dump_path / "input_2.pth"
            
            if not (left_path.exists() and right_path.exists()):
                 print(f"Error: Missing input_1.pth or input_2.pth for op execution.")
                 return

            left = torch.load(left_path, map_location='cpu')
            right = torch.load(right_path, map_location='cpu')
            
            print(f"Running on {args.device}...")
            act_unpacked = run_npu_kronecker_quant(act, left, right, device=args.device)

        # 3. Post-processing (Reshape & Slice)
        if act_unpacked is not None:
            # Reshape unpacked if needed (3D -> 2D)
            if act_unpacked.dim() == 3:
                s, g1, g2 = act_unpacked.shape
                act_unpacked = act_unpacked.reshape(s, g1 * g2)
            
            print(f"Unpacked shape: {act_unpacked.shape}")

            # Slice to first 2000 tokens for clearer visualization
            if act.shape[0] > 2000:
                print(f"Slicing first 2000 tokens from {act.shape[0]}...")
                act = act[:2000]
                act_unpacked = act_unpacked[:2000]
            
            plot_activation(act, act_unpacked, args.output_dir, name_prefix=dump_path.name)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
