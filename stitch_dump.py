import os
import torch
import numpy as np
import argparse
from pathlib import Path
import shutil

def stitch_tensors(base_dir, output_dir, tp_size=4):
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all npu directories
    npu_dirs = sorted(list(base_path.glob("npu*")))
    if len(npu_dirs) < tp_size:
        print(f"Warning: Found {len(npu_dirs)} NPU directories, expected {tp_size}. Stitching might fail.")
    
    # Assuming npu0 is the reference for structure
    ref_npu_dir = npu_dirs[0]
    
    # Iterate over token directories
    token_dirs = sorted([d for d in ref_npu_dir.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda x: int(x.name))
    
    print(f"Found {len(token_dirs)} tokens. Starting stitching...")

    for token_dir in token_dirs:
        token_id = token_dir.name
        # print(f"Processing Token ID: {token_id}")
        
        # Iterate over layer directories
        for layer_dir in sorted(token_dir.iterdir()):
            if not layer_dir.is_dir():
                continue
            
            layer_full_name = layer_dir.name
            # Simplify layer name: root.model.layers.0.mlp -> layers.0.mlp
            layer_short_name = layer_full_name.replace("root.model.", "")
            
            # Create layer directory in output
            layer_out_dir = output_path / layer_short_name
            layer_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine stitching strategy
            # ColumnParallel: Output is split, needs concat. Input is replicated.
            # RowParallel: Input is split, Output is reduced (replicated).
            # We handle both input and output files if they exist.
            
            for file_name in ["input.pth", "output.pth"]:
                file_type = file_name.split(".")[0] # input or output
                
                # Check if we need to stitch this specific file type for this layer type
                needs_concat = False
                
                # Logic for Output
                if file_type == "output":
                    if "qkv_proj" in layer_full_name or "gate_up_proj" in layer_full_name:
                        needs_concat = True
                
                # Logic for Input (Optional, usually we care about output)
                # If RowParallel (o_proj, down_proj), input is split.
                if file_type == "input":
                    if "o_proj" in layer_full_name or "down_proj" in layer_full_name:
                        needs_concat = True

                tensors = []
                missing_file = False
                
                for i in range(tp_size):
                    # Find the directory for this rank
                    rank_dir = list(base_path.glob(f"npu{i}*"))
                    if not rank_dir:
                        missing_file = True
                        break
                    
                    file_path = rank_dir[0] / token_id / layer_full_name / file_name
                    if not file_path.exists():
                        # Try checking if it's missing just for this rank or all
                        missing_file = True
                        break
                    
                    try:
                        t = torch.load(file_path, map_location="cpu")
                        tensors.append(t)
                    except Exception:
                        missing_file = True
                        break
                
                if missing_file:
                    continue
                
                try:
                    if needs_concat:
                        # Concatenate along the last dimension
                        stitched_tensor = torch.cat(tensors, dim=-1)
                    else:
                        # Replicated, just take rank 0
                        stitched_tensor = tensors[0]
                    
                    # Convert to numpy and save
                    npy_path = layer_out_dir / f"token_{token_id}_{file_type}.npy"
                    np.save(npy_path, stitched_tensor.float().numpy())
                    
                except Exception as e:
                    print(f"Error processing {layer_short_name} {file_type}: {e}")

    print(f"Stitching complete. Data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch vLLM TP dump files to NPY.")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to torch_tensors directory")
    parser.add_argument("--output_dir", type=str, default="./stitched_npy", help="Output directory")
    parser.add_argument("--tp_size", type=int, default=4, help="Tensor Parallel size")
    
    args = parser.parse_args()
    
    stitch_tensors(args.base_dir, args.output_dir, args.tp_size)
