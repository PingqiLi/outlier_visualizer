import torch
import numpy as np
import matplotlib.pyplot as plt
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

def plot_activation(activation, output_dir, name_prefix=""):
    """
    Generates 3D plot and histogram for a single activation tensor.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Data Preparation ---
    # Flatten for histogram
    act_flat = activation.flatten().abs().numpy()
    
    # Downsample for 3D plot
    def downsample(mat, target_size=500):
        rows, cols = mat.shape
        r_step = max(1, rows // target_size)
        c_step = max(1, cols // target_size)
        return mat[::r_step, ::c_step]

    act_2d = activation.numpy()
    act_ds = downsample(np.abs(act_2d))
    
    # --- 1. 3D Surface Plot ---
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        pass

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(act_ds.shape[1])
    y = np.arange(act_ds.shape[0])
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, act_ds, cmap='coolwarm', edgecolor='none', alpha=0.8)
    
    ax.set_title(f"FlatQuant Activation (Int4 Levels)\nMax: {act_flat.max():.2f}")
    ax.set_xlabel('Channel')
    ax.set_ylabel('Token')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    plt.savefig(output_dir / f"{name_prefix}_3d.png", dpi=150)
    plt.close()
    
    # --- 2. Histogram ---
    plt.figure(figsize=(10, 6))
    plt.hist(act_flat, bins=100, alpha=0.7, log=True, color='orange')
    plt.title(f"Activation Distribution (Log Scale) - {name_prefix}")
    plt.xlabel("Absolute Value (Int4 Level)")
    plt.ylabel("Count (Log)")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"{name_prefix}_hist.png")
    plt.close()
    
    print(f"Saved plots to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize FlatQuant activations (NPU Environment).")
    
    # Option 1: Load existing Int32 output
    parser.add_argument("--output_file", type=str, help="Path to output_0.pth (Int32 tensor)")
    
    # Option 2: Run Op from Inputs
    parser.add_argument("--dump_dir", type=str, help="Directory containing input_0.pth, input_1.pth, input_2.pth")
    
    parser.add_argument("--output_dir", type=str, default="./flatquant_plots", help="Output directory")
    parser.add_argument("--device", type=str, default="npu:0", help="Device to run on")
    
    args = parser.parse_args()
    
    try:
        if args.output_file:
            # Path 1: Load and Unpack Output
            print(f"Loading output file: {args.output_file}")
            out_int32 = torch.load(args.output_file, map_location='cpu')
            print(f"Output shape (Int32): {out_int32.shape}")
            
            print("Unpacking...")
            # Unpack expects [..., N]
            # If shape is [B, S, D//8], flatten to [B*S, D//8] or keep as is?
            # unpack_int32_to_int4 handles arbitrary leading dims.
            act_unpacked = unpack_int32_to_int4(out_int32)
            
            # If original shape was [S, D//8], new is [S, D]
            print(f"Unpacked shape: {act_unpacked.shape}")
            
            plot_activation(act_unpacked, args.output_dir, name_prefix=Path(args.output_file).parent.name)
            
        elif args.dump_dir:
            # Path 2: Run Op from Inputs
            dump_path = Path(args.dump_dir)
            act_path = dump_path / "input_0.pth"
            left_path = dump_path / "input_1.pth"
            right_path = dump_path / "input_2.pth"
            
            if not (act_path.exists() and left_path.exists() and right_path.exists()):
                 print(f"Error: Missing input files in {dump_path}")
                 return

            print(f"Loading inputs from {dump_path}...")
            act = torch.load(act_path, map_location='cpu')
            left = torch.load(left_path, map_location='cpu')
            right = torch.load(right_path, map_location='cpu')
            
            print(f"Running on {args.device}...")
            act_unpacked = run_npu_kronecker_quant(act, left, right, device=args.device)
            
            plot_activation(act_unpacked, args.output_dir, name_prefix=dump_path.name)
            
        else:
            print("Error: Must provide either --output_file or --dump_dir")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
