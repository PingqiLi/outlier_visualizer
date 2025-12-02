# FlatQuant Visualization Tools

This directory contains tools to simulate and visualize the effect of FlatQuant (Kronecker Decomposition) on activation outliers.

## Scripts

*   `visualize_flatquant.py`: The main visualization script.
*   `test_flatquant_sim.py`: A test script to verify the simulation logic (requires `torch`).

## Usage

### Prerequisites

*   Python 3
*   PyTorch
*   Matplotlib
*   NumPy

### 1. Prepare Dump Data

Ensure you have dumped the `npu_kronecker_quant` operator data using `msit_llm`.
The dump directory should contain:
*   `input_0.pth`: Original Activation (BF16/FP16)
*   `input_1.pth`: Left Transformation Matrix
*   `input_2.pth`: Right Transformation Matrix

### 2. Run Visualization

You can run the script by pointing to the dump directory:

```bash
python3 visualize_flatquant.py --dump_dir /path/to/dump/layer_X/npu_kronecker_quant/
```

Or by specifying individual files:

```bash
python3 visualize_flatquant.py \
    --input_act /path/to/input_0.pth \
    --left_mat /path/to/input_1.pth \
    --right_mat /path/to/input_2.pth
```

### 3. Output

The script will generate plots in `./flatquant_plots` (default) or the directory specified by `--output_dir`.

*   **3D Comparison**: Side-by-side 3D surface plots showing the activation magnitude before and after transformation.
*   **Histogram**: A log-scale histogram comparing the distribution of values, highlighting outlier suppression.
