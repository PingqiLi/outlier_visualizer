import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path

def simulate_kronecker_transform(x, left, right):
    """
    Simulates the Kronecker transformation: X_transformed = L @ X_reshaped @ R
    
    Args:
        x: Input tensor (BF16/FP32). Shape [Sequences, Hidden_Dim]
        left: Left transformation matrix. Shape [G1, G1]
        right: Right transformation matrix. Shape [G2, G2]
        
    Returns:
        x_transformed: Transformed tensor.
    """
    # Ensure inputs are float32 for calculation precision
    x = x.float()
    left = left.float()
    right = right.float()
    
    # 1. Reshape X
    # Target shape: [Batch*Seq, Left_Dim, Right_Dim]
    # We assume Hidden_Dim = Left_Dim * Right_Dim
    # And we treat the first dimension as the batch/sequence dimension
    
    seq_len, hidden_dim = x.shape
    g1 = left.shape[0]
    g2 = right.shape[0]
    
    if hidden_dim != g1 * g2:
        raise ValueError(f"Hidden dim {hidden_dim} does not match Kronecker factors {g1}x{g2}={g1*g2}")
        
    x_reshaped = x.reshape(-1, g1, g2)
    
    # 2. Apply Left Transform: L @ X
    # L: [G1, G1], X: [B, G1, G2] -> [B, G1, G2]
    # We use einsum for clarity: 'ij, bjk -> bik'
    x_left = torch.einsum('ij, bjk -> bik', left, x_reshaped)
    
    # 3. Apply Right Transform: X_left @ R
    # X_left: [B, G1, G2], R: [G2, G2] -> [B, G1, G2]
    # Einsum: 'bij, jk -> bik'
    x_transformed = torch.einsum('bij, jk -> bik', x_left, right)
    
    return x_transformed

def plot_comparison(original, transformed, output_dir, name_prefix=""):
    """
    Generates side-by-side 3D plots and histograms.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Data Preparation ---
    # Flatten for histogram
    orig_flat = original.flatten().abs().numpy()
    trans_flat = transformed.flatten().abs().numpy()
    
    # Downsample for 3D plot
    def downsample(mat, target_size=500):
        rows, cols = mat.shape
        r_step = max(1, rows // target_size)
        c_step = max(1, cols // target_size)
        return mat[::r_step, ::c_step]

    # Reshape original to match transformed 3D structure [B*G1, G2] for visualization
    # Or just visualize as is. 
    # Let's visualize the transformed data in its native [B, G1, G2] flattened to [B*G1, G2] 
    # to see the block structure, or [B, G1*G2] to match original.
    # To compare apples to apples, let's look at the flattened hidden dim.
    
    orig_2d = original.numpy()
    trans_2d = transformed.reshape(original.shape).numpy()
    
    orig_ds = downsample(np.abs(orig_2d))
    trans_ds = downsample(np.abs(trans_2d))
    
    # --- 1. 3D Surface Plot Comparison ---
    fig = plt.figure(figsize=(20, 8))
    
    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    x1 = np.arange(orig_ds.shape[1])
    y1 = np.arange(orig_ds.shape[0])
    X1, Y1 = np.meshgrid(x1, y1)
    surf1 = ax1.plot_surface(X1, Y1, orig_ds, cmap='coolwarm', edgecolor='none', alpha=0.8)
    ax1.set_title(f"Original Activation (Abs)\nMax: {orig_flat.max():.2f}")
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Token')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    # Transformed
    ax2 = fig.add_subplot(122, projection='3d')
    x2 = np.arange(trans_ds.shape[1])
    y2 = np.arange(trans_ds.shape[0])
    X2, Y2 = np.meshgrid(x2, y2)
    surf2 = ax2.plot_surface(X2, Y2, trans_ds, cmap='coolwarm', edgecolor='none', alpha=0.8)
    ax2.set_title(f"Transformed Activation (Abs)\nMax: {trans_flat.max():.2f}")
    ax2.set_xlabel('Channel (Transformed)')
    ax2.set_ylabel('Token')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    plt.savefig(output_dir / f"{name_prefix}_comparison_3d.png", dpi=150)
    plt.close()
    
    # --- 2. Histogram Comparison ---
    plt.figure(figsize=(12, 6))
    plt.hist(orig_flat, bins=100, alpha=0.5, label='Original', log=True, color='blue')
    plt.hist(trans_flat, bins=100, alpha=0.5, label='Transformed', log=True, color='orange')
    plt.title(f"Activation Distribution (Log Scale) - {name_prefix}")
    plt.xlabel("Absolute Value")
    plt.ylabel("Count (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"{name_prefix}_comparison_hist.png")
    plt.close()
    
    print(f"Saved plots to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Simulate FlatQuant transformation and visualize outliers.")
    parser.add_argument("--dump_dir", type=str, help="Directory containing input_0.pth, input_1.pth, input_2.pth")
    parser.add_argument("--input_act", type=str, help="Path to activation (input_0.pth)")
    parser.add_argument("--left_mat", type=str, help="Path to left matrix (input_1.pth)")
    parser.add_argument("--right_mat", type=str, help="Path to right matrix (input_2.pth)")
    parser.add_argument("--output_dir", type=str, default="./flatquant_plots", help="Output directory")
    
    args = parser.parse_args()
    
    # Resolve paths
    if args.dump_dir:
        dump_path = Path(args.dump_dir)
        act_path = dump_path / "input_0.pth"
        left_path = dump_path / "input_1.pth"
        right_path = dump_path / "input_2.pth"
    elif args.input_act and args.left_mat and args.right_mat:
        act_path = Path(args.input_act)
        left_path = Path(args.left_mat)
        right_path = Path(args.right_mat)
    else:
        print("Error: Must provide either --dump_dir or all of --input_act, --left_mat, --right_mat")
        return

    if not (act_path.exists() and left_path.exists() and right_path.exists()):
        print(f"Error: One or more files not found:\n{act_path}\n{left_path}\n{right_path}")
        return
        
    print(f"Loading data...")
    try:
        # Load tensors
        # Note: msit dump might save them as simple tensors or dicts. 
        # Usually torch.save saves the tensor directly.
        act = torch.load(act_path, map_location='cpu')
        left = torch.load(left_path, map_location='cpu')
        right = torch.load(right_path, map_location='cpu')
        
        print(f"Shapes: Act={act.shape}, Left={left.shape}, Right={right.shape}")
        
        # Simulate
        print("Simulating Kronecker transformation...")
        transformed = simulate_kronecker_transform(act, left, right)
        
        # Visualize
        print("Generating plots...")
        plot_comparison(act, transformed, args.output_dir, name_prefix=act_path.parent.name)
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
