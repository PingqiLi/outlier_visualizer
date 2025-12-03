import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
import torch_npu

def unpack_int32_to_int4(packed_tensor):
    """
    Unpacks [..., N] int32 tensor to [..., N*8] float tensor (representing int4 values).
    Assumes 8x4-bit packing in int32.
    """
    # packed: [..., D//8] int32
    # output: [..., D] float
    
    unpacked = []
    # Extract 8 4-bit values
    for i in range(8):
        shift = i * 4
        # Extract 4 bits
        val = (packed_tensor >> shift) & 0xF
        # Sign extension (assuming 2's complement 4-bit: -8..7)
        # 0..7 -> 0..7
        # 8..15 -> -8..-1
        val = torch.where(val >= 8, val - 16, val)
        unpacked.append(val)
        
    # Stack along the last dimension
    # Assuming linear packing order: [0, 1, 2, 3, 4, 5, 6, 7] -> int32
    result = torch.stack(unpacked, dim=-1) # [..., D//8, 8]
    result = result.flatten(-2, -1) # [..., D]
    return result.float()

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
    x_unpacked = unpack_int32_to_int4(x_quantized_int32)
    
    # Reshape back to [S, D]
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
    fig = plt.figure(figsize=(20, 8))
    
    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X1, Y1, orig_ds, cmap='coolwarm', edgecolor='none', alpha=0.8, rstride=1, cstride=1)
    ax1.set_title(f"Original Input (BF16)\nMax: {orig_flat.max():.2f}\nShape: {orig_shape}")
    ax1.set_xlabel('Channel (Dim 1)')
    ax1.set_ylabel('Token (Dim 0)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    # Transformed
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X2, Y2, trans_ds, cmap='coolwarm', edgecolor='none', alpha=0.8, rstride=1, cstride=1)
    ax2.set_title(f"FlatQuant Output (Int4 Levels)\nMax: {trans_flat.max():.2f}\nShape: {trans_shape}")
    ax2.set_xlabel('Channel (Dim 1)')
    ax2.set_ylabel('Token (Dim 0)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    plt.savefig(output_dir / f"{name_prefix}_comparison_3d.png", dpi=150)
    plt.close()
    
    print(f"Saved comparison plots to {output_dir}")

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
            act_unpacked = unpack_int32_to_int4(out_int32)
            
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
