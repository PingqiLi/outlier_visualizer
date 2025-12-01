import os
import torch
import numpy as np
import argparse
from pathlib import Path
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

def process_layer(layer_dir, token_id, tp_size, base_path, output_path):
    """
    Process a single layer for a single token.
    Returns a dictionary of {layer_short_name: {file_type: numpy_array}}
    """
    results = {}
    
    layer_full_name = layer_dir.name
    # Simplify layer name
    layer_short_name = layer_full_name.replace("root.model.", "")
    
    # Determine stitching strategy
    # ColumnParallel: Output is split, needs concat. Input is replicated.
    # RowParallel: Input is split, Output is reduced (replicated).
    
    for file_name in ["input.pth", "output.pth"]:
        file_type = file_name.split(".")[0] # input or output
        
        # Check if we need to stitch this specific file type for this layer type
        needs_concat = False
        
        # Logic for Output
        if file_type == "output":
            if "qkv_proj" in layer_full_name or "gate_up_proj" in layer_full_name:
                needs_concat = True
            elif "q_norm" in layer_full_name or "k_norm" in layer_full_name:
                # Q/K Norm outputs are [Seq, Heads_Per_Rank, Head_Dim]
                # We need to concat along the Heads dimension (dim=-2)
                needs_concat = True
                concat_dim = -2
            else:
                concat_dim = -1 # Default
        
        # Logic for Input
        if file_type == "input":
            if "o_proj" in layer_full_name or "down_proj" in layer_full_name:
                needs_concat = True
                concat_dim = -1

        tensors = []
        missing_file = False
        
        for i in range(tp_size):
            # Find the directory for this rank
            # We assume the structure is consistent across ranks
            rank_dir_pattern = f"npu{i}*"
            rank_dirs = list(base_path.glob(rank_dir_pattern))
            
            if not rank_dirs:
                missing_file = True
                break
            
            # Use the first match
            rank_dir = rank_dirs[0]
            
            file_path = rank_dir / str(token_id) / layer_full_name / file_name
            if not file_path.exists():
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
            
        # Skip q_norm and k_norm as requested by user to avoid stitching issues
        if "q_norm" in layer_full_name or "k_norm" in layer_full_name:
            continue
        
        try:
            if needs_concat:
                stitched_tensor = torch.cat(tensors, dim=concat_dim)
            else:
                stitched_tensor = tensors[0]
            
            # Flatten if necessary (usually [1, Hidden] -> [Hidden])
            # BUT: Prefill phase might have [Seq_Len, Hidden], Decode has [1, Hidden]
            # We should keep the batch dimension if it exists and is > 1
            if stitched_tensor.dim() > 1 and stitched_tensor.shape[0] == 1:
                stitched_tensor = stitched_tensor.squeeze(0)
            
            # For QK Norm, flatten [Seq, Heads, Head_Dim] -> [Seq, Hidden]
            if ("q_norm" in layer_full_name or "k_norm" in layer_full_name) and stitched_tensor.dim() == 3:
                 s = stitched_tensor.shape
                 # [Seq, Heads, Head_Dim] -> [Seq, Heads*Head_Dim]
                 stitched_tensor = stitched_tensor.view(s[0], -1)
                
            if layer_short_name not in results:
                results[layer_short_name] = {}
            results[layer_short_name][file_type] = stitched_tensor.float().numpy()
            
        except Exception as e:
            # print(f"Error processing {layer_short_name} {file_type}: {e}")
            pass
            
    return results

def stitch_tensors(base_dir, output_dir, tp_size=4, num_workers=4):
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all npu directories to verify TP size
    npu_dirs = sorted(list(base_path.glob("npu*")))
    if len(npu_dirs) < tp_size:
        print(f"Warning: Found {len(npu_dirs)} NPU directories, expected {tp_size}.")
    
    ref_npu_dir = npu_dirs[0]
    
    # Find all tokens
    token_dirs = sorted([d for d in ref_npu_dir.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda x: int(x.name))
    token_ids = [int(d.name) for d in token_dirs]
    
    print(f"Found {len(token_ids)} tokens. Starting parallel stitching with {num_workers} workers...")
    
    layer_data = {} # {layer_name: {'input': {token_id: array}, 'output': {token_id: array}}}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for t_dir in token_dirs:
            futures.append(executor.submit(process_token_wrapper, t_dir, tp_size, base_path))
            
        for future in tqdm(futures, total=len(futures), desc="Stitching Tokens"):
            token_id, token_results = future.result()
            
            for layer_name, files in token_results.items():
                if layer_name not in layer_data:
                    layer_data[layer_name] = {'input': {}, 'output': {}}
                
                for f_type, arr in files.items():
                    layer_data[layer_name][f_type][token_id] = arr

    print("Merging and saving data...")
    
    for layer_name, data_types in layer_data.items():
        layer_out_dir = output_path / layer_name
        layer_out_dir.mkdir(parents=True, exist_ok=True)
        
        for f_type, token_map in data_types.items():
            if not token_map:
                continue
            
            # Sort by token ID
            sorted_ids = sorted(token_map.keys())
            arrays = [token_map[tid] for tid in sorted_ids]
            
            # Merge strategy: Flatten all to [N, Hidden] and concatenate
            # Prefill: [Seq_Len, Hidden] -> [Seq_Len, Hidden]
            # Decode: [Hidden] or [1, Hidden] -> [1, Hidden]
            
            processed_arrays = []
            for arr in arrays:
                if arr.ndim == 1:
                    # [Hidden] -> [1, Hidden]
                    processed_arrays.append(arr.reshape(1, -1))
                elif arr.ndim == 2:
                    # [Seq_Len, Hidden] or [1, Hidden] -> Keep as is
                    processed_arrays.append(arr)
                else:
                    print(f"  Warning: Unexpected shape {arr.shape} in {layer_name} {f_type}. Skipping.")
                    continue
            
            if not processed_arrays:
                continue
                
            try:
                # Concatenate along axis 0 (Time dimension)
                merged_array = np.concatenate(processed_arrays, axis=0)
                
                save_path = layer_out_dir / f"{f_type}.npy"
                np.save(save_path, merged_array)
                # print(f"    Saved {save_path.name} with shape {merged_array.shape}")
                
            except Exception as e:
                print(f"Error merging {layer_name} {f_type}: {e}")

    print(f"Stitching complete. Data saved to {output_dir}")

def process_token_wrapper(token_dir, tp_size, base_path):
    token_id = int(token_dir.name)
    results = {} # {layer_name: {type: arr}}
    
    # Iterate over layers in this token dir
    for layer_dir in token_dir.iterdir():
        if not layer_dir.is_dir(): continue
        
        # Only process if it contains .pth files
        if not (list(layer_dir.glob("*.pth"))):
            continue
            
        layer_res = process_layer(layer_dir, token_id, tp_size, base_path, None)
        # Merge layer_res into results
        for lname, lfiles in layer_res.items():
            if lname not in results: results[lname] = {}
            results[lname].update(lfiles)
            
    return token_id, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch vLLM TP dump files to NPY.")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to torch_tensors directory")
    parser.add_argument("--output_dir", type=str, default="./stitched_npy", help="Output directory")
    parser.add_argument("--tp_size", type=int, default=4, help="Tensor Parallel size")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    
    args = parser.parse_args()
    
    stitch_tensors(args.base_dir, args.output_dir, args.tp_size, args.workers)
