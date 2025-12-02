import torch
import numpy as np
from pathlib import Path
import shutil
import os

# Import the module to test
# We need to add the directory to sys.path or just import by path if possible, 
# but since it's in the same dir, we can try importing if we run from there.
# For simplicity, I'll just copy the function here or use run_command to run the script itself.

def generate_dummy_data(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    B, S = 1, 128
    G1, G2 = 64, 64
    Hidden = G1 * G2
    
    # Create random activation with some outliers
    x = torch.randn(S, Hidden)
    # Add outliers
    x[0, 0] = 100.0
    x[10, 500] = -80.0
    
    # Create random transform matrices (orthogonal-ish)
    # For testing, let's make them Identity to verify correctness easily first
    # left = torch.eye(G1)
    # right = torch.eye(G2)
    
    # Or random
    left = torch.randn(G1, G1)
    right = torch.randn(G2, G2)
    
    torch.save(x, output_dir / "input_0.pth")
    torch.save(left, output_dir / "input_1.pth")
    torch.save(right, output_dir / "input_2.pth")
    
    print(f"Generated dummy data in {output_dir}")
    return output_dir

if __name__ == "__main__":
    test_dir = Path("./test_dump/layer_0/npu_kronecker_quant")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    generate_dummy_data(test_dir)
    
    # Now run the visualization script
    # We assume the script is in the same directory or we know the path
    script_path = Path(__file__).parent / "visualize_flatquant.py"
    
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        exit(1)
        
    cmd = f"python3 {script_path} --dump_dir {test_dir} --output_dir ./test_plots"
    print(f"Running: {cmd}")
    os.system(cmd)
    
    # Check if outputs exist
    expected_plot = Path("./test_plots/npu_kronecker_quant_comparison_3d.png")
    if expected_plot.exists():
        print("SUCCESS: Plot generated.")
    else:
        print("FAILURE: Plot not found.")
